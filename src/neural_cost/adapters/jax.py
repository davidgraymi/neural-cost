"""JAX adapter for common jaxpr dot and elementwise primitives."""

from collections.abc import Sequence
from statistics import median
from time import perf_counter_ns
from typing import Any

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


def _aval_bytes(var: Any) -> int:
    aval = getattr(var, "aval", None)
    if aval is None or not hasattr(aval, "shape"):
        return 0
    return numel(tuple(int(d) for d in aval.shape)) * int(aval.dtype.itemsize)


def _as_jaxpr(value: Any) -> Any:
    return getattr(value, "jaxpr", value)


class JaxAdapter(FrameworkAdapter):
    """Trace a JAX function and translate common primitives to operations.

    The tracer recurses into higher-order primitives (``jit``/``pjit``,
    ``scan``, ``while``, ``cond``, ``custom_jvp_call``, ``remat``) so models
    built from ``lax.scan`` or nested ``jax.jit`` calls are fully counted.
    ``scan`` bodies are multiplied by their trip count; ``while`` bodies are
    counted once because the trip count is data dependent; for ``cond`` the
    most expensive branch is used as a conservative estimate.

    ``parameter_argnums`` selects which positional arguments hold weights.
    By default every argument after the first is treated as a parameter,
    together with any arrays closed over by the traced function.
    """

    name = "jax"

    def __init__(self, parameter_argnums: Sequence[int] | None = None) -> None:
        self.parameter_argnums = tuple(parameter_argnums) if parameter_argnums is not None else None
        self.warnings: list[str] = []

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import jax
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error
        closed = jax.make_jaxpr(model)(*example_inputs)
        jaxpr = closed.jaxpr
        self.warnings = []

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
        return operations

    def _walk(
        self,
        jaxpr: Any,
        param_vars: dict[int, tuple[int, int]],
        repeat: int,
        operations: list[Operation],
        counter: list[int],
    ) -> None:
        for equation in jaxpr.eqns:
            counter[0] += 1
            index = counter[0]
            primitive = equation.primitive.name
            subs = _sub_jaxprs(equation)
            if subs:
                inner_repeat = repeat
                if primitive == "scan":
                    inner_repeat = repeat * int(equation.params.get("length", 1))
                elif primitive == "while":
                    self.warnings.append(
                        f"while_loop at equation {index}: trip count is data dependent; "
                        "body counted once"
                    )
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
                    self._walk(inner, inner_params, inner_repeat, collected, counter)
                    branch_results.append(collected)
                if primitive == "cond" and branch_results:
                    from ..estimate import estimate_operations

                    operations.extend(
                        max(branch_results, key=lambda ops: estimate_operations(ops).flops)
                    )
                else:
                    for collected in branch_results:
                        operations.extend(collected)
                continue

            if primitive in _PASSTHROUGH_PRIMITIVES and equation.invars:
                source = equation.invars[0]
                if id(source) in param_vars:
                    for out in equation.outvars:
                        param_vars[id(out)] = param_vars[id(source)]
                continue

            in_avals = [value.aval for value in equation.invars if hasattr(value, "aval")]
            out_avals = [value.aval for value in equation.outvars if hasattr(value, "aval")]
            if not out_avals or not hasattr(out_avals[0], "shape"):
                continue
            input_shapes = tuple(
                tuple(int(d) for d in aval.shape) for aval in in_avals if hasattr(aval, "shape")
            )
            output_shape = tuple(int(d) for d in out_avals[0].shape)
            if any(d <= 0 for shape in (*input_shapes, output_shape) for d in shape):
                continue
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
                    Operation(
                        f"dot_{index}", "matmul", input_shapes, output_shape, dtype_bytes, attrs
                    )
                )
            elif primitive == "conv_general_dilated" and len(input_shapes) == 2:
                lhs, rhs = input_shapes
                if len(lhs) != 4 or len(rhs) != 4 or len(output_shape) != 4:
                    continue
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
            elif primitive in _ELEMENTWISE_PRIMITIVES:
                attrs = dict(base_attrs)
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

    def hardware(self, device: Any = None, **kwargs: Any):
        """Detect a :class:`~neural_cost.HardwareSpec` for the JAX device in use.

        ``device`` defaults to ``jax.devices()[0]``, i.e. the platform selected
        by ``JAX_PLATFORMS``, so a CPU run on a machine with a GPU is scored
        against the CPU rather than the GPU.  Extra keyword arguments are
        forwarded to :func:`neural_cost.detect_hardware`.  Returns the
        ``(spec, detection)`` pair.
        """
        try:
            import jax
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error
        from ..hardware_detect import detect_hardware

        return detect_hardware(device=device if device is not None else jax.devices()[0], **kwargs)

    def calibrate(self, *, quick: bool = False):
        """Measure size-dependent FLOP/s, bandwidth and per-call dispatch for jitted JAX.

        Returns a :class:`neural_cost.attribution.HardwareCalibration` with
        matmul and elementwise throughput curves.  Under ``jax.jit`` a whole
        model is one dispatch, so pass ``kernel_count=1`` to
        :func:`neural_cost.attribute_gap` for jitted functions.
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error
        from ..attribution import HardwareCalibration

        def best_time(fn: Any, *args: Any, trials: int = 7) -> float:
            fn(*args).block_until_ready()
            best = float("inf")
            for _ in range(trials):
                start = perf_counter_ns()
                fn(*args).block_until_ready()
                best = min(best, (perf_counter_ns() - start) / 1e9)
            return best

        tiny = jnp.zeros(1)
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
        for n in (64, 256, 1024) if quick else (32, 64, 128, 256, 512, 1024, 2048):
            a = jnp.ones((n, n))
            flops_curve.append((2.0 * n**3, 2.0 * n**3 / kernel_time(best_time(matmul, a, a))))

        bump = jax.jit(lambda x: x + 1.0)
        mib = 1024 * 1024
        bandwidth_curve = []
        for size in (
            (mib // 16, 4 * mib, 64 * mib)
            if quick
            else (mib // 16, mib // 4, mib, 4 * mib, 16 * mib, 64 * mib, 256 * mib)
        ):
            x = jnp.ones(size // 4)
            bandwidth_curve.append((2.0 * size, 2.0 * size / kernel_time(best_time(bump, x))))

        platform = jax.devices()[0].platform
        return HardwareCalibration(
            achievable_flops=max(rate for _, rate in flops_curve),
            achievable_bandwidth=bandwidth_curve[-1][1],
            dispatch_seconds=dispatch,
            call_overhead_seconds=sync_latency,
            source=f"jax microbenchmarks ({platform}, jit)",
            flops_curve=tuple(flops_curve),
            bandwidth_curve=tuple(bandwidth_curve),
        )

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
        device = str(args[0].device) if args and hasattr(args[0], "device") else None
        peak = None
        allocated = None
        reserved = None
        if args and hasattr(args[0], "device"):
            dev = args[0].device
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
