"""JAX adapter: dense, conv, elementwise, indexing and control-flow jaxpr primitives."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from statistics import median
from time import perf_counter_ns
from typing import Any

from ..coverage import Coverage
from ..operations import Operation, numel
from ..profiler import Measurement
from .base import FrameworkAdapter

# Primitives that only change layout or precision: a parameter stays a parameter.
_PASSTHROUGH_PRIMITIVES = frozenset(
    {
        "transpose",
        "reshape",
        "convert_element_type",
        "squeeze",
        "expand_dims",
        "copy",
        "broadcast_in_dim",
    }
)

_ELEMENTWISE_PRIMITIVES = frozenset(
    {
        "add",
        "mul",
        "max",
        "exp",
        "tanh",
        "logistic",
        "sub",
        "neg",
        "sin",
        "cos",
        "rsqrt",
        "integer_pow",
        "abs",
        "log",
        "sqrt",
        "reduce_sum",
        "reduce_max",
        "div",
        "erf",
        "min",
        "pow",
        "square",
        # Comparisons, selection and integer/bit ops common in RL environments.
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
        "not",
        "xor",
        "select_n",
        "clamp",
        "sign",
        "floor",
        "ceil",
        "round",
        "rem",
        "exp2",
        "log1p",
        "expm1",
        "is_finite",
        "atan2",
        "shift_left",
        "shift_right_logical",
        "shift_right_arithmetic",
        "population_count",
        "reduce_min",
        "reduce_prod",
        "reduce_and",
        "reduce_or",
        "argmax",
        "argmin",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "iota",
    }
)

_REDUCTION_PRIMITIVES = frozenset(
    {
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_and",
        "reduce_or",
        "argmax",
        "argmin",
    }
)

# Fused scaled-dot-product attention primitives that backends substitute for
# jax.nn.dot_product_attention; inputs are (q, k, v, ...) in (B, H, T, D).
_FUSED_ATTENTION_PRIMITIVES = frozenset({"mps.sdpa"})

# Pure data movement: read the inputs, write the output, no arithmetic.
_DATA_MOVEMENT_PRIMITIVES = frozenset({"concatenate", "pad", "rev"})

# In-place indexed updates whose ``update_jaxpr`` combines old and new values.
_COMBINING_SCATTERS = frozenset(
    {
        "scatter-add",
        "scatter_add",
        "scatter-mul",
        "scatter_mul",
        "scatter-min",
        "scatter_min",
        "scatter-max",
        "scatter_max",
    }
)


def _dot_general_flops(lhs: tuple[int, ...], rhs: tuple[int, ...], dimension_numbers: Any) -> int:
    """Exact FLOPs of a general tensor contraction.

    MACs = batch × lhs_free × rhs_free × contract = |lhs| · |rhs| / (batch · contract).
    """
    (lhs_contract, _rhs_contract), (lhs_batch, _rhs_batch) = dimension_numbers
    shared = 1
    for axis in (*lhs_contract, *lhs_batch):
        shared *= lhs[axis]
    return 2 * (numel(lhs) * numel(rhs)) // shared if shared else 0


def _sub_jaxprs(equation: Any) -> list[tuple[Any, int]]:
    """Return ``(jaxpr, outer_invar_offset)`` pairs for higher-order primitives."""
    params = equation.params
    name = equation.primitive.name
    if name == "while":
        body = params["body_jaxpr"]
        return [(body, int(params.get("cond_nconsts", 0)))]
    if name == "cond":
        return [(branch, 1) for branch in params["branches"]]
    found = []
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in params and params[key] is not None:
            found.append((params[key], 0))
            break
    return found


def _join_scope(prefix: str, name: str) -> str:
    """Join name-scope paths, skipping empty parts (``jax.named_scope``, Flax module names)."""
    return "/".join(part for part in (prefix, name) if part)


def _is_literal(var: Any) -> bool:
    return type(var).__name__ == "Literal" and hasattr(var, "val")


def _literal_int(var: Any) -> int | None:
    if not _is_literal(var):
        return None
    try:
        return int(var.val)
    except (TypeError, ValueError):
        return None


def _infer_while_trip_count(equation: Any) -> int | None:
    """Infer the trip count of ``while i < N: i += step`` with literal ``i0``, ``N``, ``step``.

    Matches the pattern ``lax.while_loop`` produces for counted loops: the
    condition is a single ``lt`` of a carry against a literal bound, the
    carry starts from a literal, and the body adds a literal step to it.
    Returns ``None`` for anything else (data-dependent loops).
    """
    params = equation.params
    cond = _as_jaxpr(params["cond_jaxpr"])
    body = _as_jaxpr(params["body_jaxpr"])
    cond_nconsts = int(params.get("cond_nconsts", 0))
    body_nconsts = int(params.get("body_nconsts", 0))
    if len(cond.eqns) != 1 or cond.eqns[0].primitive.name != "lt":
        return None
    counter, bound_var = cond.eqns[0].invars
    bound = _literal_int(bound_var)
    carry = list(cond.invars)[cond_nconsts:]
    if bound is None or counter not in carry:
        return None
    position = carry.index(counter)
    init = list(equation.invars)[cond_nconsts + body_nconsts :]
    start = _literal_int(init[position]) if position < len(init) else None
    if start is None:
        return None
    body_counter = list(body.invars)[body_nconsts + position]
    body_out = list(body.outvars)[position]
    for body_eqn in body.eqns:
        if body_eqn.outvars and body_eqn.outvars[0] is body_out:
            if body_eqn.primitive.name != "add" or len(body_eqn.invars) != 2:
                return None
            left, right = body_eqn.invars
            other = right if left is body_counter else left if right is body_counter else None
            step = _literal_int(other) if other is not None else None
            if step is None or step <= 0:
                return None
            return max(0, math.ceil((bound - start) / step))
    return None


def _aval_bytes(var: Any) -> int:
    aval = getattr(var, "aval", None)
    if aval is None or not hasattr(aval, "shape"):
        return 0
    return numel(tuple(int(d) for d in aval.shape)) * int(aval.dtype.itemsize)


def _as_jaxpr(value: Any) -> Any:
    return getattr(value, "jaxpr", value)


def _import_jax() -> Any:
    try:
        import jax
    except ImportError as error:  # pragma: no cover - optional package
        raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error
    return jax


# (platform, device string, quick) -> HardwareCalibration, filled by calibrate().
_CALIBRATION_CACHE: dict[tuple[Any, str, bool], Any] = {}


def _measure_calibration(jax: Any, device: Any, quick: bool) -> Any:
    """Run the dispatch, matmul and streaming microbenchmarks on ``device``."""
    import jax.numpy as jnp

    from ..attribution import HardwareCalibration

    def on_device(value: Any) -> Any:
        return jax.device_put(value, device)

    key = jax.random.key(0)

    def materialized(shape: tuple[int, ...]) -> Any:
        # Random data, not jnp.ones: backends such as jax-mps (MLX) keep
        # constant-filled arrays as broadcast scalars, so a streaming probe on
        # them reads 4 bytes and reports roughly twice the real bandwidth.
        return on_device(jax.random.uniform(key, shape, dtype=jnp.float32))

    def best_time(fn: Any, *args: Any, trials: int = 7) -> float:
        fn(*args).block_until_ready()
        best = float("inf")
        for _ in range(trials):
            start = perf_counter_ns()
            fn(*args).block_until_ready()
            best = min(best, (perf_counter_ns() - start) / 1e9)
        return best

    tiny = on_device(jnp.zeros(1))
    add_one = jax.jit(lambda x: x + 1)
    n_calls = 500
    add_one(tiny).block_until_ready()
    best_loop = float("inf")
    for _ in range(3):
        start = perf_counter_ns()
        value = tiny
        for _ in range(n_calls):
            value = add_one(value)
        value.block_until_ready()
        best_loop = min(best_loop, (perf_counter_ns() - start) / 1e9)
    dispatch = best_loop / n_calls
    sync_latency = max(0.0, best_time(add_one, tiny, trials=20) - dispatch)

    def kernel_time(seconds: float) -> float:
        # Each probe is one jitted call: remove the per-call cost the model adds.
        return max(seconds - dispatch - sync_latency, 0.01 * seconds)

    matmul = jax.jit(jnp.matmul)
    flops_curve = []
    for n in (64, 256, 1024) if quick else (32, 64, 128, 256, 512, 1024, 2048, 4096):
        a = materialized((n, n))
        flops_curve.append((2.0 * n**3, 2.0 * n**3 / kernel_time(best_time(matmul, a, a))))

    bump = jax.jit(lambda x: x + 1.0)
    mib = 1024 * 1024
    bandwidth_curve = []
    for size in (
        (mib // 16, 4 * mib, 64 * mib)
        if quick
        else (mib // 16, mib // 4, mib, 4 * mib, 16 * mib, 64 * mib, 256 * mib)
    ):
        x = materialized((size // 4,))
        bandwidth_curve.append((2.0 * size, 2.0 * size / kernel_time(best_time(bump, x))))

    platform = getattr(device, "platform", "unknown")
    return HardwareCalibration(
        achievable_flops=max(rate for _, rate in flops_curve),
        achievable_bandwidth=bandwidth_curve[-1][1],
        dispatch_seconds=dispatch,
        call_overhead_seconds=sync_latency,
        source=f"jax microbenchmarks ({platform}, jit)",
        flops_curve=tuple(flops_curve),
        bandwidth_curve=tuple(bandwidth_curve),
    )


class JaxAdapter(FrameworkAdapter):
    """Trace a JAX function and translate common primitives to operations.

    The tracer recurses into higher-order primitives (``jit``/``pjit``,
    ``scan``, ``while``, ``cond``, ``custom_jvp_call``, ``remat``) so models
    built from ``lax.scan`` or nested ``jax.jit`` calls are fully counted.
    ``scan`` bodies are multiplied by their static length.  ``while`` bodies
    use, in order of precedence: ``trip_counts[k]`` for the ``k``-th while
    loop in trace order (0-based), a trip count inferred from a counted
    ``i < N`` loop with literal bounds, ``default_trip_count``, or 1 with a
    warning.  The resolved counts are recorded in :attr:`while_trip_counts`.
    For ``cond`` the most expensive branch is used as a conservative estimate.

    Indexing primitives are modeled by the elements they touch: ``gather``,
    ``dynamic_slice`` and ``slice`` become ``gather`` operations; ``scatter``
    (``-add``/``-mul``/``-min``/``-max``) and ``dynamic_update_slice`` become
    in-place ``scatter`` operations.

    ``parameter_argnums`` selects which positional arguments hold weights.
    By default every argument after the first is treated as a parameter,
    together with any arrays closed over by the traced function.
    """

    name = "jax"

    def __init__(
        self,
        parameter_argnums: Sequence[int] | None = None,
        *,
        trip_counts: Mapping[int, int] | None = None,
        default_trip_count: int | None = None,
    ) -> None:
        self.parameter_argnums = tuple(parameter_argnums) if parameter_argnums is not None else None
        self.trip_counts = dict(trip_counts or {})
        if any(count < 0 for count in self.trip_counts.values()):
            raise ValueError("trip_counts must be non-negative")
        if default_trip_count is not None and default_trip_count < 0:
            raise ValueError("default_trip_count must be non-negative")
        self.default_trip_count = default_trip_count
        self.warnings: list[str] = []
        self.while_trip_counts: list[tuple[int, str]] = []
        self.coverage: Coverage | None = None

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import jax
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error
        closed = jax.make_jaxpr(model)(*example_inputs)
        jaxpr = closed.jaxpr
        self.warnings = []
        self.while_trip_counts = []
        self._modeled = self._free = self._unmodeled_bytes = 0
        self._unmodeled: list[str] = []

        # Flattened invars follow the pytree leaves of each positional argument.
        leaf_counts = [len(jax.tree.leaves(arg)) for arg in example_inputs]
        argnums = (
            self.parameter_argnums
            if self.parameter_argnums is not None
            else tuple(range(1, len(example_inputs)))
        )
        # id(var) -> (parameter_id, parameter_bytes of the original array).  Layout
        # ops propagate the entry so ``x @ w.T`` or a broadcast bias still resolves.
        param_vars: dict[int, tuple[int, int]] = {}
        cursor = 0
        for position, count in enumerate(leaf_counts):
            for var in jaxpr.invars[cursor : cursor + count]:
                if position in argnums:
                    param_vars[id(var)] = (id(var), _aval_bytes(var))
            cursor += count
        # Closed-over arrays are keyed by the array object itself, so a weight
        # captured by several nested ``jit`` functions is still counted once.
        for var, value in zip(jaxpr.constvars, closed.consts, strict=False):
            param_vars[id(var)] = (id(value), _aval_bytes(var))

        operations: list[Operation] = []
        counter = [0]
        self._walk(jaxpr, param_vars, 1, operations, counter)
        self.coverage = Coverage.from_counts(
            self._modeled, self._free, self._unmodeled, self._unmodeled_bytes
        )
        return operations

    def _walk(
        self,
        jaxpr: Any,
        param_vars: dict[int, tuple[int, int]],
        repeat: int,
        operations: list[Operation],
        counter: list[int],
        scope: str = "",
    ) -> None:
        for equation in jaxpr.eqns:
            counter[0] += 1
            index = counter[0]
            primitive = equation.primitive.name
            eqn_scope = _join_scope(scope, str(getattr(equation.source_info, "name_stack", "")))
            subs = _sub_jaxprs(equation)
            if subs:
                inner_repeat = repeat
                if primitive == "scan":
                    inner_repeat = repeat * int(equation.params.get("length", 1))
                elif primitive == "while":
                    inner_repeat = repeat * self._while_trip_count(equation, index)
                    if inner_repeat == 0:
                        continue
                branch_results: list[list[Operation]] = []
                for sub, offset in subs:
                    inner = _as_jaxpr(sub)
                    inner_params: dict[int, tuple[int, int]] = {}
                    outer_vars = list(equation.invars)[offset:]
                    for inner_var, outer_var in zip(inner.invars, outer_vars, strict=False):
                        if id(outer_var) in param_vars:
                            inner_params[id(inner_var)] = param_vars[id(outer_var)]
                    consts = getattr(sub, "consts", ())
                    for position, const in enumerate(getattr(inner, "constvars", ())):
                        key = id(consts[position]) if position < len(consts) else id(const)
                        inner_params[id(const)] = (key, _aval_bytes(const))
                    collected: list[Operation] = []
                    inner_scope = eqn_scope
                    if primitive in {"pjit", "jit", "closed_call", "core_call"}:
                        inner_scope = _join_scope(eqn_scope, str(equation.params.get("name", "")))
                    self._walk(inner, inner_params, inner_repeat, collected, counter, inner_scope)
                    branch_results.append(collected)
                if primitive == "cond" and branch_results:
                    from ..estimate import estimate_operations

                    def cost(ops: list[Operation]) -> tuple[int, int]:
                        total = estimate_operations(ops)
                        return total.flops, total.total_bytes

                    operations.extend(max(branch_results, key=cost))
                else:
                    for collected in branch_results:
                        operations.extend(collected)
                continue

            before = len(operations)
            if self._translate(equation, index, param_vars, repeat, operations) == "free":
                self._free += 1
            elif len(operations) > before:
                self._modeled += 1
                if eqn_scope:
                    operations[before:] = [
                        replace(op, name=f"{eqn_scope}/{op.name}") for op in operations[before:]
                    ]
            else:
                self._unmodeled.append(primitive)
                self._unmodeled_bytes += repeat * sum(_aval_bytes(v) for v in equation.outvars)

    def _translate(
        self,
        equation: Any,
        index: int,
        param_vars: dict[int, tuple[int, int]],
        repeat: int,
        operations: list[Operation],
    ) -> str | None:
        """Append operations for one leaf equation; return ``"free"`` for zero-cost ones."""
        primitive = equation.primitive.name
        if primitive in _PASSTHROUGH_PRIMITIVES and equation.invars:
            source = equation.invars[0]
            if id(source) in param_vars:
                for out in equation.outvars:
                    param_vars[id(out)] = param_vars[id(source)]
            return "free"

        in_avals = [value.aval for value in equation.invars if hasattr(value, "aval")]
        out_avals = [value.aval for value in equation.outvars if hasattr(value, "aval")]
        if not out_avals or not hasattr(out_avals[0], "shape"):
            return "free"
        input_shapes = tuple(
            tuple(int(d) for d in aval.shape) for aval in in_avals if hasattr(aval, "shape")
        )
        output_shape = tuple(int(d) for d in out_avals[0].shape)
        if any(d <= 0 for shape in (*input_shapes, output_shape) for d in shape):
            return "free"
        dtype_bytes = int(out_avals[0].dtype.itemsize)
        base_attrs: dict[str, Any] = {"repeat": repeat} if repeat > 1 else {}

        if primitive == "dot_general" and len(input_shapes) == 2:
            attrs = dict(base_attrs)
            attrs["flops"] = _dot_general_flops(
                input_shapes[0], input_shapes[1], equation.params["dimension_numbers"]
            )
            for position in (1, 0):
                entry = param_vars.get(id(equation.invars[position]))
                if entry is not None:
                    attrs["parameter_id"], attrs["parameter_bytes"] = entry
                    break
            operations.append(
                Operation(f"dot_{index}", "matmul", input_shapes, output_shape, dtype_bytes, attrs)
            )
        elif primitive == "conv_general_dilated" and len(input_shapes) == 2:
            lhs, rhs = input_shapes
            if len(lhs) != 4 or len(rhs) != 4 or len(output_shape) != 4:
                return None
            numbers = equation.params.get("dimension_numbers")
            if numbers is not None and hasattr(numbers, "lhs_spec"):
                lhs = tuple(lhs[i] for i in numbers.lhs_spec)
                rhs = tuple(rhs[i] for i in numbers.rhs_spec)
                out = tuple(output_shape[i] for i in numbers.out_spec)
            else:
                out = output_shape
            attrs = dict(base_attrs)
            attrs["groups"] = int(equation.params.get("feature_group_count", 1))
            entry = param_vars.get(id(equation.invars[1]))
            if entry is not None:
                attrs["parameter_id"], attrs["parameter_bytes"] = entry
            operations.append(
                Operation(f"conv_{index}", "conv2d", (lhs, rhs), out, dtype_bytes, attrs)
            )
        elif primitive in {"gather", "dynamic_slice", "slice"}:
            operand = input_shapes[0] if input_shapes else output_shape
            indices = input_shapes[1] if primitive == "gather" and len(input_shapes) > 1 else ()
            attrs = dict(base_attrs)
            attrs["index_bytes"] = sum(_aval_bytes(var) for var in equation.invars[1:])
            operations.append(
                Operation(
                    f"{primitive}_{index}",
                    "gather",
                    (operand, indices or (1,)),
                    output_shape,
                    dtype_bytes,
                    attrs,
                )
            )
        elif primitive == "dynamic_update_slice" or primitive.startswith("scatter"):
            if primitive == "dynamic_update_slice":
                if len(input_shapes) < 2:
                    return None
                operand, updates = input_shapes[0], input_shapes[1] or (1,)
                index_vars = equation.invars[2:]
            else:
                if len(input_shapes) < 3:
                    return None
                operand, updates = input_shapes[0], input_shapes[2] or (1,)
                index_vars = equation.invars[1:2]
            attrs = dict(base_attrs)
            attrs["index_bytes"] = sum(_aval_bytes(var) for var in index_vars)
            if primitive in _COMBINING_SCATTERS:
                attrs["combine"] = 1
            operations.append(
                Operation(
                    f"{primitive}_{index}",
                    "scatter",
                    (operand, (1,), updates),
                    output_shape,
                    dtype_bytes,
                    attrs,
                )
            )
        elif primitive in _FUSED_ATTENTION_PRIMITIVES and len(input_shapes) >= 3:
            # Backend-substituted fused attention, e.g. jax-mps replaces
            # jax.nn.dot_product_attention with ``mps.sdpa`` on (B, H, T, D).
            q, k = input_shapes[0], input_shapes[1]
            if len(q) != 4 or len(k) != 4:
                return None
            batch, heads, q_len, head_dim = q
            kv_len = k[2]
            attrs = dict(base_attrs)
            name = f"{primitive}_{index}"
            if q_len == kv_len:
                attrs.update(
                    num_heads=heads, seq_len=q_len, head_dim=head_dim, include_projections=0
                )
                operations.append(
                    Operation(name, "attention", input_shapes[:3], output_shape, dtype_bytes, attrs)
                )
            else:
                # QKᵀ and AV (2 FLOP per MAC each) plus a 5-FLOP softmax per score.
                scores = batch * heads * q_len * kv_len
                attrs["flops"] = 4 * scores * head_dim + 5 * scores
                operations.append(
                    Operation(name, "custom", input_shapes[:3], output_shape, dtype_bytes, attrs)
                )
        elif primitive in _DATA_MOVEMENT_PRIMITIVES:
            attrs = dict(base_attrs)
            attrs["flops_per_element"] = 0
            operations.append(
                Operation(
                    f"{primitive}_{index}",
                    "elementwise",
                    input_shapes,
                    output_shape,
                    dtype_bytes,
                    attrs,
                )
            )
        elif primitive in _ELEMENTWISE_PRIMITIVES:
            attrs = dict(base_attrs)
            if primitive in _REDUCTION_PRIMITIVES and input_shapes:
                # A reduction does one operation per *input* element.
                attrs["flops_per_element"] = numel(input_shapes[0]) / numel(output_shape)
            for var in equation.invars:  # e.g. a bias add or norm scale
                entry = param_vars.get(id(var))
                if entry is not None:
                    attrs["parameter_id"], attrs["parameter_bytes"] = entry
                    break
            operations.append(
                Operation(
                    f"{primitive}_{index}",
                    "elementwise",
                    input_shapes,
                    output_shape,
                    dtype_bytes,
                    attrs,
                )
            )
        return None

    def _while_trip_count(self, equation: Any, index: int) -> int:
        ordinal = len(self.while_trip_counts)
        if ordinal in self.trip_counts:
            count, source = int(self.trip_counts[ordinal]), "user"
        elif (inferred := _infer_while_trip_count(equation)) is not None:
            count, source = inferred, "inferred"
        elif self.default_trip_count is not None:
            count, source = int(self.default_trip_count), "default"
        else:
            count, source = 1, "assumed"
            self.warnings.append(
                f"while_loop #{ordinal} at equation {index}: trip count is data dependent; "
                f"body counted once (pass trip_counts={{{ordinal}: n}} or default_trip_count)"
            )
        self.while_trip_counts.append((count, source))
        return count

    def hardware(
        self, device: Any = None, *, measure: str = "auto", quick: bool = False, **kwargs: Any
    ):
        """Return a ``(HardwareSpec, detection)`` pair for the JAX device in use.

        ``device`` defaults to ``jax.devices()[0]``, i.e. the platform selected
        by ``JAX_PLATFORMS``, so a CPU run on a machine with a GPU is scored
        against the CPU rather than the GPU.  Extra keyword arguments are
        forwarded to :func:`neural_cost.detect_hardware`.

        ``measure`` chooses where the peaks come from:

        * ``"auto"`` (default): published-spec tables, falling back to
          on-device measurement (:meth:`calibrate`, a short sweep with
          ``quick=True``) for devices without a table entry such as TPUs or
          unknown GPUs.
        * ``"always"``: measured achievable peaks on the device.  The
          detection is ``None``.
        * ``"never"``: tables only; raise ``ValueError`` when none applies.
        """
        if measure not in {"auto", "always", "never"}:
            raise ValueError("measure must be 'auto', 'always' or 'never'")
        jax = _import_jax()
        from ..hardware_detect import detect_hardware, normalize_device_type

        device = device if device is not None else jax.devices()[0]
        if measure != "always":
            try:
                return detect_hardware(device=device, **kwargs)
            except ValueError:
                if measure == "never":
                    raise
        calibration = self.calibrate(device=device, quick=quick)
        name = f"{getattr(device, 'device_kind', None) or device} (measured)"
        return calibration.to_hardware_spec(name, normalize_device_type(device)), None

    def calibrate(self, *, quick: bool = False, device: Any = None, refresh: bool = False):
        """Measure size-dependent FLOP/s, bandwidth and per-call dispatch for jitted JAX.

        Runs a matmul sweep (FLOP/s) and an ``x + 1`` streaming sweep
        (bandwidth: one read plus one write per element) through JAX itself
        on ``device`` (default: ``jax.devices()[0]``).  This works for any
        backend, including ones with no published-spec table, and shows how
        far the achievable peak sits below the datasheet figure.

        Returns a :class:`neural_cost.attribution.HardwareCalibration` with
        matmul and elementwise throughput curves.  Results are cached per
        device and mode for the process; pass ``refresh=True`` to re-measure.
        Turn the result into a spec with ``calibration.to_hardware_spec()``.
        Under ``jax.jit`` a whole model is one dispatch, so pass
        ``kernel_count=1`` to :func:`neural_cost.attribute_gap` for jitted
        functions.
        """
        jax = _import_jax()
        device = device if device is not None else jax.devices()[0]
        key = (getattr(device, "platform", None), str(device), quick)
        if not refresh and key in _CALIBRATION_CACHE:
            return _CALIBRATION_CACHE[key]
        result = _measure_calibration(jax, device, quick)
        _CALIBRATION_CACHE[key] = result
        return result

    def benchmark(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Benchmark a JAX function while waiting for dispatched device work."""
        try:
            import jax
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error

        def wait(value: Any) -> None:
            for leaf in jax.tree.leaves(value):
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()

        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")
        for _ in range(warmup):
            wait(function(*args, **kwargs))
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            wait(function(*args, **kwargs))
            samples.append((perf_counter_ns() - start) / 1_000_000_000)
        # The first array leaf in the arguments (e.g. inside a Flax variables dict).
        first = next((leaf for leaf in jax.tree.leaves(args) if hasattr(leaf, "device")), None)
        device = str(first.device) if first is not None else None
        peak = None
        allocated = None
        reserved = None
        if first is not None:
            dev = first.device
            if hasattr(dev, "memory_stats") and callable(dev.memory_stats):
                try:
                    stats = dev.memory_stats()
                    if stats:
                        peak = stats.get("peak_bytes_in_use")
                        allocated = stats.get("bytes_in_use")
                        reserved = stats.get("bytes_limit") or stats.get("bytes_reserved")
                except (AttributeError, RuntimeError, ValueError):
                    stats = None
        return Measurement(
            median(samples),
            tuple(samples),
            peak_memory_bytes=peak,
            device=device,
            allocated_memory_bytes=allocated,
            reserved_memory_bytes=reserved,
        )
