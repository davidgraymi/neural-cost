"""PyTorch module adapter (available only when PyTorch is installed)."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns
from typing import Any

from ..coverage import Coverage
from ..operations import Operation
from ..profiler import Measurement
from .base import FrameworkAdapter

# Leaf modules with no compulsory cost in inference (eval-mode dropout is identity).
_FREE_MODULES = frozenset(
    {
        "Identity",
        "Flatten",
        "Unflatten",
        "Dropout",
        "Dropout1d",
        "Dropout2d",
        "Dropout3d",
        "AlphaDropout",
        "FeatureAlphaDropout",
    }
)

_FUNCTIONAL_CONVS = frozenset(
    {
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    }
)


def _conv_types(torch: Any) -> tuple[type, ...]:
    nn = torch.nn
    return (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )


def _pool_types(torch: Any) -> tuple[type, ...]:
    nn = torch.nn
    return tuple(
        getattr(nn, f"{prefix}Pool{dims}d")
        for prefix in ("Max", "Avg", "AdaptiveMax", "AdaptiveAvg")
        for dims in (1, 2, 3)
    )


def _pool_window(module: Any, source: tuple[int, ...], output: tuple[int, ...]) -> tuple[int, ...]:
    """Pooling window per spatial dimension (``ceil(in / out)`` for adaptive pools)."""
    dims = int(module.__class__.__name__[-2])
    kernel = getattr(module, "kernel_size", None)
    if kernel is not None:
        return tuple(kernel) if isinstance(kernel, (tuple, list)) else (int(kernel),) * dims
    spatial_in, spatial_out = source[-dims:], output[-dims:]
    return tuple(-(-i // max(o, 1)) for i, o in zip(spatial_in, spatial_out, strict=True))


def _conv_shapes(
    data: tuple[int, ...],
    weight: tuple[int, ...],
    output: tuple[int, ...],
    groups: int,
    transposed: bool,
) -> tuple[str, tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...], dict[str, Any]]:
    """Portable (kind, inputs, output, attrs) for a PyTorch convolution.

    PyTorch stores transposed-convolution weights as ``(C_in, C_out / groups, *k)``;
    the portable record wants ``(C_out, C_in / groups, *k)``.  Unbatched inputs
    ``(C, *spatial)`` get a batch dimension of 1.  Plain 2-D convolutions keep
    the ``conv2d`` kind; everything else is ``conv``.
    """
    if transposed:
        weight = (weight[1] * groups, weight[0] // groups, *weight[2:])
    if len(data) == len(weight) - 1:
        data, output = (1, *data), (1, *output)
    attrs: dict[str, Any] = {"groups": groups}
    if transposed:
        attrs["transposed"] = True
    kind = "conv2d" if len(weight) == 4 and not transposed else "conv"
    return kind, (data, weight), output, attrs


def _sdpa_attrs(
    q_shape: tuple[int, ...], k_shape: tuple[int, ...], is_causal: bool
) -> dict[str, Any]:
    """Attention attrs for ``scaled_dot_product_attention`` on (..., L, E) / (..., S, E)."""
    if len(q_shape) == 4:
        _, heads, q_len, head_dim = q_shape
    elif len(q_shape) == 3:
        _, q_len, head_dim = q_shape
        heads = 1
    else:
        heads, q_len, head_dim = 1, 1, 1
    kv_len = k_shape[-2] if len(k_shape) >= 2 else q_len
    attrs: dict[str, Any] = {
        "num_heads": heads,
        "seq_len": q_len,
        "kv_len": kv_len,
        "head_dim": head_dim,
        "include_projections": False,
    }
    if is_causal:
        # PyTorch aligns the causal mask to the top-left corner when L != S.
        attrs.update(causal=True, q_offset=0)
    return attrs


# ATen kernels that correspond to operations the hook adapter captures
# (contractions, normalization, attention, embedding, pooling, RNN cells).
_HEAVY_KERNEL_PREFIXES = (
    "addmm",
    "mm",
    "bmm",
    "baddbmm",
    "matmul",
    "linear",
    "conv",
    "convolution",
    "mkldnn_convolution",
    "mps_convolution",
    "cudnn",
    "miopen",
    "native_layer_norm",
    "native_batch_norm",
    "native_group_norm",
    "batch_norm",
    "rms_norm",
    "fused_rms_norm",
    "embedding",
    "native_multi_head_attention",
    "scaled_dot_product",
    "efficient_attention",
    "flash_attention",
    "lstm",
    "gru",
    "rnn",
    "thnn_fused",
    "adaptive_avg_pool",
    "avg_pool",
    "max_pool",
)


@dataclass(frozen=True, slots=True)
class KernelRecord:
    """One dispatched ATen kernel: name, bytes read + written, and dispatch level."""

    name: str
    bytes: int
    heavy: bool
    python_dispatch: bool


class TorchAdapter(FrameworkAdapter):
    """Capture Linear and Conv2d module calls through forward hooks.

    :meth:`benchmark` always times a loop with no profiler attached.  With
    ``use_profiler=True`` (the default) it then runs a separate profiled pass
    for trace statistics and CPU peak memory; ``use_profiler=False`` skips
    that pass and halves the benchmark's run time.
    """

    name = "pytorch"

    def __init__(self, use_profiler: bool = True) -> None:
        self.use_profiler = use_profiler
        self.coverage: Coverage | None = None

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on optional package
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error

        _CONV_TYPES = _conv_types(torch)
        _POOL_TYPES = _pool_types(torch)
        captured: list[Operation] = []
        hooks = []

        def dtype_bytes(tensor: Any) -> int:
            return tensor.element_size()

        # MultiheadAttention key length and causal flag, recorded by a pre-hook
        # because forward hooks do not see keyword arguments.
        mha_calls: dict[int, tuple[int | None, bool]] = {}

        def hook(name: str, module: Any):
            def record(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                if not inputs:
                    return
                source = inputs[0]
                if not isinstance(source, torch.Tensor):
                    return
                if isinstance(module, torch.nn.Linear):
                    if not isinstance(output, torch.Tensor):
                        return
                    captured.append(
                        Operation(
                            name,
                            "linear",
                            (tuple(source.shape), tuple(module.weight.T.shape)),
                            tuple(output.shape),
                            dtype_bytes(source),
                            {
                                "parameter_bytes": sum(
                                    p.numel() * p.element_size()
                                    for p in module.parameters(recurse=False)
                                ),
                                "parameter_id": id(module.weight),
                            },
                        )
                    )
                elif isinstance(module, _CONV_TYPES):
                    if not isinstance(output, torch.Tensor):
                        return
                    kind, conv_inputs, conv_output, conv_attrs = _conv_shapes(
                        tuple(source.shape),
                        tuple(module.weight.shape),
                        tuple(output.shape),
                        module.groups,
                        bool(module.transposed),
                    )
                    captured.append(
                        Operation(
                            name,
                            kind,
                            conv_inputs,
                            conv_output,
                            dtype_bytes(source),
                            {
                                "parameter_bytes": sum(
                                    p.numel() * p.element_size()
                                    for p in module.parameters(recurse=False)
                                ),
                                "parameter_id": id(module.weight),
                                **conv_attrs,
                            },
                        )
                    )
                elif isinstance(module, _POOL_TYPES):
                    if not isinstance(output, torch.Tensor):
                        return
                    captured.append(
                        Operation(
                            name,
                            "pooling",
                            (tuple(source.shape),),
                            tuple(output.shape),
                            dtype_bytes(source),
                            {
                                "kernel_size": _pool_window(
                                    module, tuple(source.shape), tuple(output.shape)
                                )
                            },
                        )
                    )
                elif isinstance(module, torch.nn.Embedding):
                    if not isinstance(output, torch.Tensor):
                        return
                    captured.append(
                        Operation(
                            name,
                            "embedding",
                            (tuple(source.shape),),
                            tuple(output.shape),
                            dtype_bytes(module.weight),
                            {
                                "parameter_bytes": module.weight.numel()
                                * module.weight.element_size(),
                                "parameter_id": id(module.weight),
                            },
                        )
                    )
                elif isinstance(module, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)):
                    # Unroll as input×hidden + hidden×hidden linears per layer per direction.
                    if not isinstance(output, (tuple, list)):
                        return
                    directions = 2 if module.bidirectional else 1
                    gates = (
                        4
                        if isinstance(module, torch.nn.LSTM)
                        else (3 if isinstance(module, torch.nn.GRU) else 1)
                    )
                    if source.dim() >= 3:
                        steps = source.shape[1] if module.batch_first else source.shape[0]
                        batch = source.shape[0] if module.batch_first else source.shape[1]
                    else:  # unbatched (L, H_in)
                        steps, batch = source.shape[0], 1
                    in_size = source.shape[-1]
                    h = module.hidden_size
                    for layer in range(module.num_layers):
                        layer_in = in_size if layer == 0 else h * directions
                        for direction in range(directions):
                            suffix = f"l{layer}" + ("_reverse" if direction else "")
                            tag = f"{name}.layer{layer}" + (".reverse" if direction else "")
                            # The input projection is one GEMM over all timesteps; the
                            # recurrent projection is T dependent (B, H) GEMMs.
                            for proj, width, rows, repeat in (
                                ("ih", layer_in, batch * steps, 1),
                                ("hh", h, batch, steps),
                            ):
                                weight = getattr(module, f"weight_{proj}_{suffix}")
                                bias = getattr(module, f"bias_{proj}_{suffix}", None)
                                tensors = [weight] + ([bias] if bias is not None else [])
                                attrs: dict[str, Any] = {
                                    "parameter_bytes": sum(
                                        t.numel() * t.element_size() for t in tensors
                                    ),
                                    "parameter_id": id(weight),
                                }
                                if repeat > 1:
                                    attrs["repeat"] = repeat
                                captured.append(
                                    Operation(
                                        f"{tag}.{proj}",
                                        "linear",
                                        ((rows, width), (width, gates * h)),
                                        (rows, gates * h),
                                        source.element_size(),
                                        attrs,
                                    )
                                )
                elif isinstance(module, torch.nn.MultiheadAttention):
                    if not isinstance(output, (tuple, list, torch.Tensor)):
                        return
                    out_tensor = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(out_tensor, torch.Tensor):
                        return
                    embed_dim = module.embed_dim
                    num_heads = module.num_heads
                    if source.dim() >= 3:
                        batch_first = bool(getattr(module, "batch_first", False))
                        seq_len = source.shape[1] if batch_first else source.shape[0]
                        batch = source.shape[0] if batch_first else source.shape[1]
                    else:  # unbatched (L, E)
                        seq_len, batch = source.shape[0], 1
                    kv_len, is_causal = mha_calls.pop(id(module), (None, False))
                    mha_attrs: dict[str, Any] = {}
                    if kv_len is not None and kv_len != seq_len:
                        mha_attrs["kv_len"] = kv_len
                    if is_causal:
                        mha_attrs["causal"] = True
                    captured.append(
                        Operation(
                            name,
                            "attention",
                            (tuple(source.shape),),
                            (seq_len, batch, embed_dim),
                            dtype_bytes(source),
                            {
                                "num_heads": num_heads,
                                "seq_len": seq_len,
                                # out_proj is a child module, so recurse.
                                "parameter_bytes": sum(
                                    p.numel() * p.element_size() for p in module.parameters()
                                ),
                                "parameter_id": id(module.in_proj_weight)
                                if module.in_proj_weight is not None
                                else id(module),
                                **mha_attrs,
                            },
                        )
                    )
                elif (
                    isinstance(
                        module,
                        (
                            torch.nn.LayerNorm,
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                        ),
                    )
                    or (hasattr(torch.nn, "RMSNorm") and isinstance(module, torch.nn.RMSNorm))
                    or module.__class__.__name__ == "RMSNorm"
                ):
                    if not isinstance(output, torch.Tensor):
                        return
                    if (
                        hasattr(torch.nn, "RMSNorm") and isinstance(module, torch.nn.RMSNorm)
                    ) or module.__class__.__name__ == "RMSNorm":
                        kind = "rmsnorm"
                    elif isinstance(module, torch.nn.LayerNorm):
                        kind = "layernorm"
                    else:
                        kind = "batchnorm"
                    attrs: dict[str, int | float | tuple[int, ...]] = {}
                    if hasattr(module, "weight") and module.weight is not None:
                        attrs["parameter_bytes"] = sum(
                            p.numel() * p.element_size() for p in module.parameters(recurse=False)
                        )
                        attrs["parameter_id"] = id(module.weight)
                    captured.append(
                        Operation(
                            name,
                            kind,
                            (tuple(source.shape),),
                            tuple(output.shape),
                            dtype_bytes(source),
                            attrs,
                        )
                    )

            return record

        _tracked_list = [
            torch.nn.Linear,
            *_CONV_TYPES,
            *_POOL_TYPES,
            torch.nn.Embedding,
            torch.nn.RNN,
            torch.nn.GRU,
            torch.nn.LSTM,
            torch.nn.MultiheadAttention,
            torch.nn.LayerNorm,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
        ]
        if hasattr(torch.nn, "RMSNorm"):
            _tracked_list.append(torch.nn.RMSNorm)
        _TRACKED = tuple(_tracked_list)
        # MultiheadAttention may dispatch to scaled_dot_product_attention internally;
        # its hook already accounts for the full attention cost, so suppress the
        # functional capture while inside one.
        mha_depth = [0]

        def enter_mha(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            mha_depth[0] += 1
            key = args[1] if len(args) > 1 else kwargs.get("key")
            kv_len = None
            if isinstance(key, torch.Tensor):
                if key.dim() >= 3:
                    kv_len = key.shape[1] if getattr(module, "batch_first", False) else key.shape[0]
                elif key.dim() == 2:
                    kv_len = key.shape[0]
            is_causal = args[7] if len(args) > 7 else kwargs.get("is_causal", False)
            mha_calls[id(module)] = (kv_len, bool(is_causal))

        def exit_mha(*_args: Any) -> None:
            mha_depth[0] -= 1

        # Coverage: every tracked module, plus leaf modules outside tracked
        # subtrees, is a unit; record which ones actually execute.
        units: dict[int, tuple[str, bool]] = {}
        called: set[int] = set()
        tracked_prefixes: list[str] = []
        for name, module in model.named_modules():
            tracked = isinstance(module, _TRACKED) or module.__class__.__name__ == "RMSNorm"
            if any(name.startswith(prefix + ".") for prefix in tracked_prefixes):
                continue
            if tracked:
                tracked_prefixes.append(name)
            if tracked or not any(True for _ in module.children()):
                units[id(module)] = (module.__class__.__name__, tracked)
                hooks.append(module.register_forward_hook(lambda mod, *_: called.add(id(mod))))

        for name, module in model.named_modules():
            if isinstance(module, _TRACKED) or module.__class__.__name__ == "RMSNorm":
                if isinstance(module, torch.nn.MultiheadAttention):
                    hooks.append(module.register_forward_pre_hook(enter_mha, with_kwargs=True))
                    hooks.append(module.register_forward_hook(exit_mha))
                hooks.append(
                    module.register_forward_hook(hook(name or module.__class__.__name__, module))
                )

        was_training = model.training
        orig_sdpa = getattr(torch.nn.functional, "scaled_dot_product_attention", None)
        sdpa_counter = [0]
        if orig_sdpa is not None:

            def wrapped_sdpa(query, key, value, *args, **kwargs):
                out = orig_sdpa(query, key, value, *args, **kwargs)
                if (
                    mha_depth[0] == 0
                    and isinstance(query, torch.Tensor)
                    and isinstance(out, torch.Tensor)
                ):
                    sdpa_counter[0] += 1
                    q_shape = tuple(query.shape)
                    k_shape = tuple(key.shape) if isinstance(key, torch.Tensor) else q_shape
                    v_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else q_shape
                    is_causal = kwargs.get("is_causal", args[2] if len(args) > 2 else False)
                    captured.append(
                        Operation(
                            f"scaled_dot_product_attention_{sdpa_counter[0]}",
                            "attention",
                            (q_shape, k_shape, v_shape),
                            tuple(out.shape),
                            dtype_bytes(query),
                            _sdpa_attrs(q_shape, k_shape, bool(is_causal)),
                        )
                    )
                return out

            torch.nn.functional.scaled_dot_product_attention = wrapped_sdpa

        try:
            model.eval()
            with torch.no_grad():
                model(*example_inputs)
        finally:
            if orig_sdpa is not None:
                torch.nn.functional.scaled_dot_product_attention = orig_sdpa
            for registered_hook in hooks:
                registered_hook.remove()
            model.train(was_training)
        modeled = free = 0
        unmodeled: list[str] = []
        for key, (class_name, tracked) in units.items():
            if key not in called:
                continue
            if tracked:
                modeled += 1
            elif class_name in _FREE_MODULES:
                free += 1
            else:
                unmodeled.append(class_name)
        self.coverage = Coverage.from_counts(modeled, free, unmodeled, unit="modules")
        return captured

    def count_kernels(self, model: Any, example_inputs: Sequence[Any]) -> int:
        """Count ATen operator dispatches in one forward pass (≈ kernels launched in eager)."""
        return len(self.trace_kernels(model, example_inputs))

    def trace_kernels(self, model: Any, example_inputs: Sequence[Any]) -> list[KernelRecord]:
        """Record every ATen kernel one forward pass dispatches, with its traffic.

        A ``TorchDispatchMode`` sees each operator that reaches a backend
        kernel, including activations, residual adds and copies that forward
        hooks never capture; view/metadata ops (no kernel) are skipped.  A
        ``TorchFunctionMode`` marks which kernels were the first issued by a
        Python-level call: only those pay Python dispatch overhead, while
        kernels a C++ composite issues internally (e.g. the per-timestep ops of
        an eager CPU LSTM) do not.
        """
        try:
            import torch
            from torch.overrides import TorchFunctionMode
            from torch.utils._python_dispatch import TorchDispatchMode
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error

        records: list[KernelRecord] = []
        call_state = {"depth": 0, "fresh": False}

        def tensor_bytes(value: Any) -> int:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, (list, tuple)):
                return sum(tensor_bytes(v) for v in value)
            return 0

        def is_view(func: Any) -> bool:
            schema = getattr(func, "_schema", None)
            returns = schema.returns if schema is not None else ()
            return bool(returns) and all(
                r.alias_info is not None and not r.alias_info.is_write for r in returns
            )

        class _Calls(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                call_state["depth"] += 1
                if call_state["depth"] == 1:
                    call_state["fresh"] = True
                try:
                    return func(*args, **(kwargs or {}))
                finally:
                    call_state["depth"] -= 1

        class _Kernels(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if not is_view(func):
                    name = func.name() if hasattr(func, "name") else str(func)
                    short = name.split("::")[-1].split(".")[0]
                    records.append(
                        KernelRecord(
                            name,
                            tensor_bytes(list(args)) + tensor_bytes(out),
                            short.lstrip("_").startswith(_HEAVY_KERNEL_PREFIXES),
                            python_dispatch=call_state["fresh"] or call_state["depth"] == 0,
                        )
                    )
                    call_state["fresh"] = False
                return out

        with torch.no_grad(), _Calls(), _Kernels():
            model(*example_inputs)
        return records

    def replay(self, operations: Sequence[Operation], device: Any = None) -> list[float | None]:
        """Time each operation in isolation on ``device`` (see ``_torch_replay``).

        Returns per-operation seconds (multiplied by ``repeat``) or ``None``
        where an operation kind cannot be synthesized.  Pass the result to
        :func:`neural_cost.attribute_gap` as ``isolated_seconds`` to split
        kernel inefficiency from composition effects.
        """
        from ._torch_replay import replay_operations

        return replay_operations(operations, device)

    def calibrate(self, device: Any = None, *, dtype: Any = None, quick: bool = False):
        """Measure size-dependent FLOP/s, bandwidth, dispatch and sync cost on ``device``.

        Returns a :class:`neural_cost.attribution.HardwareCalibration` with
        ``flops_curve`` (square matmuls, 32² … 4096²) and ``bandwidth_curve``
        (copies, 64 KiB … 256 MiB) so cache-resident and small kernels are
        modeled at the rate they actually achieve.  Per-kernel dispatch is
        subtracted from every probe, matching the additive model
        ``T = sync + kernels × dispatch + Σ kernel time``.
        """
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error
        from ..attribution import HardwareCalibration

        device = torch.device(device or "cpu")
        dtype = dtype or torch.float32

        from ._torch_replay import make_sync
        from ._torch_replay import per_call as _per_call

        sync = make_sync(torch, device)

        def per_call(fn: Any, **kwargs: Any) -> float:
            return _per_call(fn, sync, **kwargs)

        # 1. Dispatch: steady-state cost of a tiny kernel.
        tiny = torch.zeros(1, device=device, dtype=dtype)
        out = torch.empty_like(tiny)
        dispatch = per_call(lambda: torch.add(tiny, 1, out=out), reps=2000)
        asynchronous = device.type in {"cuda", "mps"}

        # 2. Per-call synchronization latency: one tiny kernel + sync, minus dispatch.
        sync_latency = float("inf")
        for _ in range(20):
            start = perf_counter_ns()
            torch.add(tiny, 1, out=out)
            sync()
            sync_latency = min(sync_latency, (perf_counter_ns() - start) / 1e9)
        sync_latency = max(0.0, sync_latency - dispatch)

        def kernel_time(seconds: float) -> float:
            # Synchronous dispatch adds to every kernel; asynchronous launches overlap
            # execution, so the back-to-back time already is the kernel time.
            return seconds if asynchronous else max(seconds - dispatch, 0.01 * seconds)

        # 3. Compute curve: square matmuls.
        sizes = (64, 256, 1024) if quick else (32, 64, 128, 256, 512, 1024, 2048)
        if device.type != "cpu" and not quick:
            sizes += (4096,)
        flops_curve = []
        for n in sizes:
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)
            seconds = kernel_time(per_call(lambda a=a, b=b: torch.mm(a, b)))
            flops_curve.append((2.0 * n**3, 2.0 * n**3 / seconds))

        # 4. Bandwidth curve: copies (read + write) from cache-resident to DRAM-sized.
        mib = 1024 * 1024
        copy_sizes = (
            (mib // 16, 4 * mib, 64 * mib)
            if quick
            else (mib // 16, mib // 4, mib, 4 * mib, 16 * mib, 64 * mib, 256 * mib)
        )
        bandwidth_curve = []
        for size in copy_sizes:
            src = torch.ones(size // tiny.element_size(), device=device, dtype=dtype)
            dst = torch.empty_like(src)
            seconds = kernel_time(per_call(lambda s=src, d=dst: d.copy_(s)))
            bandwidth_curve.append((2.0 * size, 2.0 * size / seconds))
            del src, dst

        return HardwareCalibration(
            achievable_flops=max(rate for _, rate in flops_curve),
            achievable_bandwidth=bandwidth_curve[-1][1],
            dispatch_seconds=dispatch,
            call_overhead_seconds=sync_latency,
            source=f"torch microbenchmarks ({device}, {str(dtype).replace('torch.', '')})",
            flops_curve=tuple(flops_curve),
            bandwidth_curve=tuple(bandwidth_curve),
            async_dispatch=asynchronous,
        )

    def benchmark(
        self,
        function: Any,
        *args: Any,
        warmup: int = 3,
        repeats: int = 10,
        min_sample_seconds: float | None = None,
        **kwargs: Any,
    ) -> Measurement:
        """Time ``function`` and collect device memory and profiler telemetry.

        The timed samples always come from a loop with no profiler attached:
        ``torch.profiler`` adds per-operator overhead (15-20 % on small models)
        that would otherwise inflate the latency.  With ``use_profiler=True``
        a second, separate profiled pass of ``repeats`` iterations fills in
        ``trace_event_count``, ``trace_device_time_seconds`` and, on CPU, the
        peak memory.  ``min_sample_seconds`` batches short calls into longer
        samples (see :func:`neural_cost.profiler.timed_samples`).
        """
        from dataclasses import replace

        timed = self._time(
            function,
            *args,
            warmup=warmup,
            repeats=repeats,
            min_sample_seconds=min_sample_seconds,
            **kwargs,
        )
        if not getattr(self, "use_profiler", True):
            return timed
        traced = self.trace(function, *args, warmup=0, repeats=repeats, **kwargs)
        return replace(
            timed,
            peak_memory_bytes=_first(timed.peak_memory_bytes, traced.peak_memory_bytes),
            allocated_memory_bytes=_first(
                timed.allocated_memory_bytes, traced.allocated_memory_bytes
            ),
            reserved_memory_bytes=_first(timed.reserved_memory_bytes, traced.reserved_memory_bytes),
            trace_event_count=traced.trace_event_count,
            trace_device_time_seconds=_first(
                traced.trace_device_time_seconds, timed.trace_device_time_seconds
            ),
        )

    def _time(
        self,
        function: Any,
        *args: Any,
        warmup: int = 3,
        repeats: int = 10,
        min_sample_seconds: float | None = None,
        **kwargs: Any,
    ) -> Measurement:
        """Synchronized wall-clock timing with no profiler attached.

        CUDA reports allocator counters reset after warmup.  MPS reports the
        GPU time of each sample from ``torch.mps.Event`` pairs; it has no peak
        counter, so only current and driver memory are reported.
        """
        import torch

        from ..profiler import timed_samples

        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")
        device = _tensor_device(torch, args, kwargs)
        kind = device.type if device is not None else "cpu"
        sync = _synchronizer(torch, device)

        for _ in range(warmup):
            function(*args, **kwargs)
        sync()
        if kind == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timer = _MpsTimer(torch) if kind == "mps" else None
        samples, calls, overhead = timed_samples(
            lambda: function(*args, **kwargs),
            lambda output: sync(),
            warmup=0,
            repeats=repeats,
            min_sample_seconds=min_sample_seconds,
            hooks=timer,
        )
        peak = allocated = reserved = None
        if kind == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
        elif kind == "mps":
            allocated = torch.mps.current_allocated_memory()
            reserved = torch.mps.driver_allocated_memory()
        return Measurement(
            median(samples),
            tuple(samples),
            peak,
            str(device or "cpu"),
            allocated,
            reserved,
            # GPU time for ``repeats`` calls, like the profiler's figure.
            trace_device_time_seconds=timer.total / calls if timer is not None else None,
            calls_per_sample=calls,
            sync_overhead_seconds=overhead,
        )

    def trace(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Profile ``repeats`` iterations with ``torch.profiler``.

        The returned samples include profiler overhead; :meth:`benchmark`
        times a separate, unprofiled loop.  Device time is the self time of
        each op, so nested ops are counted once.  CPU peak memory is the
        high-water mark of the profiler's allocation timeline on top of the
        parameters and inputs that were live before the first iteration.
        """
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error
        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")
        device = _tensor_device(torch, args, kwargs)
        kind = device.type if device is not None else "cpu"
        sync = _synchronizer(torch, device)
        for _ in range(warmup):
            function(*args, **kwargs)
        sync()
        if kind == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if kind == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        samples: list[float] = []
        with torch.profiler.profile(
            activities=activities, profile_memory=kind in {"cuda", "cpu"}
        ) as trace:
            for _ in range(repeats):
                start = perf_counter_ns()
                function(*args, **kwargs)
                sync()
                samples.append((perf_counter_ns() - start) / 1_000_000_000)
        events = trace.events()
        peak = allocated = reserved = None
        if kind == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            device_time, event_count = _trace_device_totals(events)
        elif kind == "mps":
            allocated = torch.mps.current_allocated_memory()
            reserved = torch.mps.driver_allocated_memory()
            device_time = _trace_cpu_seconds(events)  # host-side op time; benchmark() uses GPU time
            event_count = len(events)
        else:
            high_water = _cpu_allocation_high_water(trace)
            if high_water is not None:
                peak = _live_bytes(torch, function, args, kwargs) + high_water
            allocated = reserved = peak
            device_time = _trace_cpu_seconds(events)
            event_count = len(events)
        return Measurement(
            median(samples),
            tuple(samples),
            peak,
            str(device) if device else "cpu",
            allocated,
            reserved,
            event_count,
            device_time,
        )


def _first(*values: Any) -> Any:
    """The first value that is not ``None``."""
    return next((value for value in values if value is not None), None)


def _tensor_device(torch: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
    """Device of the first tensor among the arguments, searching nested containers."""
    stack = list(args) + list(kwargs.values())
    while stack:
        value = stack.pop(0)
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return None


def _synchronizer(torch: Any, device: Any) -> Any:
    """A function that waits for all queued work on ``device``."""
    kind = device.type if device is not None else "cpu"
    if kind == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if kind == "mps":
        return torch.mps.synchronize
    return lambda: None


class _MpsTimer:
    """GPU time of each sample from a pair of ``torch.mps.Event`` markers.

    Used as :func:`neural_cost.profiler.timed_samples` hooks: the events
    bracket the calls of one sample, and ``done`` reads the elapsed GPU time
    once the sample's work has finished.
    """

    def __init__(self, torch: Any) -> None:
        self._begin = torch.mps.Event(enable_timing=True)
        self._end = torch.mps.Event(enable_timing=True)
        self.total = 0.0

    def start(self) -> None:
        self._begin.record()

    def stop(self) -> None:
        self._end.record()

    def done(self) -> None:
        self._end.synchronize()
        self.total += self._begin.elapsed_time(self._end) / 1_000


def _cpu_allocation_high_water(trace: Any) -> int | None:
    """Largest net CPU allocation reached during a ``profile_memory`` trace.

    The profiler records every allocation and free as a ``[memory]`` event
    with a signed byte count.  Walking them in time order gives the bytes
    live at each moment relative to the start of the trace.
    """
    try:
        raw = trace.profiler.kineto_results.events()
    except AttributeError:  # pragma: no cover - very old torch
        return None
    deltas = sorted(
        (event.start_ns(), event.nbytes())
        for event in raw
        if event.name() == "[memory]" and str(event.device_type()).endswith("CPU")
    )
    if not deltas:
        return None
    live = high = 0
    for _, nbytes in deltas:
        live += nbytes
        high = max(high, live)
    return high


def _live_bytes(torch: Any, function: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> int:
    """Bytes held by parameters, buffers and tensor arguments before a run."""
    seen: set[int] = set()
    total = 0

    def add(tensor: Any) -> None:
        nonlocal total
        key = tensor.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += tensor.untyped_storage().nbytes()

    for name in ("parameters", "buffers"):
        method = getattr(function, name, None)
        if callable(method):
            for tensor in method():
                add(tensor)
    stack = list(args) + list(kwargs.values())
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            add(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return total


def _trace_cpu_seconds(events: Any) -> float:
    """Total CPU time in a profiler trace, counting each op once.

    ``cpu_time_total`` is inclusive: ``aten::linear`` already contains its
    ``aten::addmm`` child, so summing it over every event counts nested time
    repeatedly.  Self time partitions the trace without overlap.
    """
    return sum(event.self_cpu_time_total for event in events) / 1_000_000


def _trace_device_totals(events: Any) -> tuple[float, int]:
    """Device seconds and kernel-launching ops in a CUDA profiler trace.

    A CPU op's ``device_time_total`` includes the kernels of all its children,
    and the kernels may also appear as events of their own.  Summing the self
    device time of CPU-side ops attributes each kernel to exactly one op.
    """
    import torch

    cpu = torch.autograd.DeviceType.CPU
    seconds = 0.0
    launching_ops = 0
    for event in events:
        if getattr(event, "device_type", cpu) != cpu:
            continue  # the kernel itself; already counted under the op that launched it
        own = getattr(event, "self_device_time_total", None)
        if own is None:  # torch < 2.4
            own = event.self_cuda_time_total
        seconds += own
        launching_ops += own > 0
    return seconds / 1_000_000, launching_ops


class TorchFxAdapter(TorchAdapter):
    """Capture PyTorch module and inline functional operations via torch.fx graph tracing."""

    name = "pytorch-fx"

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import torch
            import torch.fx
            from torch.fx.passes.shape_prop import ShapeProp
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchFxAdapter") from error

        _CONV_TYPES = _conv_types(torch)
        _POOL_TYPES = _pool_types(torch)

        try:
            gm = torch.fx.symbolic_trace(model)
            # Without no_grad, autograd keeps every activation alive for a
            # backward pass that never comes: ~7x the memory of a forward pass.
            with torch.no_grad():
                ShapeProp(gm).propagate(*example_inputs)
        except Exception:  # noqa: BLE001
            # Fall back gracefully to hook-based capture if symbolic tracing fails
            return super().operations(model, example_inputs)

        captured: list[Operation] = []
        modules = dict(model.named_modules())
        # FX treats torch.nn modules as leaves.  Recurrent and attention modules
        # have no functional decomposition here, so take their operations from
        # the hook-based capture (one forward pass, computed lazily).
        delegated_types = (
            torch.nn.RNN,
            torch.nn.GRU,
            torch.nn.LSTM,
            torch.nn.MultiheadAttention,
        )
        hook_ops: list[Operation] | None = None

        def delegated(target: str) -> list[Operation]:
            nonlocal hook_ops
            if hook_ops is None:
                hook_ops = list(TorchAdapter.operations(self, model, example_inputs))
            return [op for op in hook_ops if op.name == target or op.name.startswith(target + ".")]

        def norm_attrs(mod: Any) -> dict[str, Any]:
            if getattr(mod, "weight", None) is None:
                return {}
            return {
                "parameter_bytes": sum(
                    p.numel() * p.element_size() for p in mod.parameters(recurse=False)
                ),
                "parameter_id": id(mod.weight),
            }

        def get_shape(val: Any) -> tuple[int, ...] | None:
            if hasattr(val, "shape"):
                return tuple(int(d) for d in val.shape)
            if isinstance(val, torch.fx.Node) and "tensor_meta" in val.meta:
                meta = val.meta["tensor_meta"]
                if hasattr(meta, "shape"):
                    return tuple(int(d) for d in meta.shape)
            return None

        def get_dtype_bytes(val: Any) -> int:
            if hasattr(val, "dtype"):
                return val.element_size() if hasattr(val, "element_size") else 4
            if isinstance(val, torch.fx.Node) and "tensor_meta" in val.meta:
                meta = val.meta["tensor_meta"]
                if hasattr(meta, "dtype"):
                    return torch.empty((), dtype=meta.dtype).element_size()
            return 4

        for node in gm.graph.nodes:
            if node.op == "call_module" and isinstance(
                modules.get(str(node.target)), delegated_types
            ):
                # Checked first: these return tuples, which carry no single output shape.
                captured.extend(delegated(str(node.target)))
                continue
            out_shape = get_shape(node)
            if not out_shape:
                continue
            dtype_b = get_dtype_bytes(node)

            if node.op == "call_module":
                mod = modules.get(str(node.target))
                if mod is None:
                    continue
                in_shapes = tuple(s for s in (get_shape(arg) for arg in node.args) if s is not None)
                if isinstance(mod, torch.nn.Linear):
                    if in_shapes:
                        captured.append(
                            Operation(
                                node.name,
                                "linear",
                                (in_shapes[0], tuple(mod.weight.T.shape)),
                                out_shape,
                                dtype_b,
                                {
                                    "parameter_bytes": sum(
                                        p.numel() * p.element_size()
                                        for p in mod.parameters(recurse=False)
                                    ),
                                    "parameter_id": id(mod.weight),
                                },
                            )
                        )
                elif isinstance(mod, _CONV_TYPES):
                    if in_shapes:
                        kind, conv_inputs, conv_output, conv_attrs = _conv_shapes(
                            in_shapes[0],
                            tuple(mod.weight.shape),
                            out_shape,
                            mod.groups,
                            bool(mod.transposed),
                        )
                        captured.append(
                            Operation(
                                node.name,
                                kind,
                                conv_inputs,
                                conv_output,
                                dtype_b,
                                {
                                    "parameter_bytes": sum(
                                        p.numel() * p.element_size()
                                        for p in mod.parameters(recurse=False)
                                    ),
                                    "parameter_id": id(mod.weight),
                                    **conv_attrs,
                                },
                            )
                        )
                elif isinstance(mod, torch.nn.Embedding):
                    captured.append(
                        Operation(
                            node.name,
                            "embedding",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            {
                                "parameter_bytes": mod.weight.numel() * mod.weight.element_size(),
                                "parameter_id": id(mod.weight),
                            },
                        )
                    )
                elif isinstance(
                    mod,
                    (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Tanh),
                ):
                    captured.append(
                        Operation(
                            node.name, "elementwise", in_shapes or (out_shape,), out_shape, dtype_b
                        )
                    )
                elif isinstance(mod, torch.nn.LayerNorm):
                    captured.append(
                        Operation(
                            node.name,
                            "layernorm",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            norm_attrs(mod),
                        )
                    )
                elif (
                    hasattr(torch.nn, "RMSNorm") and isinstance(mod, torch.nn.RMSNorm)
                ) or mod.__class__.__name__ == "RMSNorm":
                    attrs = {}
                    if hasattr(mod, "weight") and mod.weight is not None:
                        attrs["parameter_bytes"] = sum(
                            p.numel() * p.element_size() for p in mod.parameters(recurse=False)
                        )
                        attrs["parameter_id"] = id(mod.weight)
                    captured.append(
                        Operation(
                            node.name,
                            "rmsnorm",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            attrs,
                        )
                    )
                elif isinstance(
                    mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
                ):
                    captured.append(
                        Operation(
                            node.name,
                            "batchnorm",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            norm_attrs(mod),
                        )
                    )
                elif isinstance(mod, _POOL_TYPES):
                    source_shape = in_shapes[0] if in_shapes else out_shape
                    captured.append(
                        Operation(
                            node.name,
                            "pooling",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            {"kernel_size": _pool_window(mod, source_shape, out_shape)},
                        )
                    )

            elif node.op in {"call_function", "call_method"}:
                fn = node.target
                fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
                in_shapes = tuple(s for s in (get_shape(arg) for arg in node.args) if s is not None)

                if fn_name in {"matmul", "mm", "bmm", "linear"} or fn in {
                    torch.matmul,
                    torch.mm,
                    torch.bmm,
                    torch.nn.functional.linear,
                }:
                    if len(in_shapes) >= 2:
                        captured.append(
                            Operation(node.name, "matmul", in_shapes[:2], out_shape, dtype_b)
                        )
                elif fn_name in {
                    "add",
                    "mul",
                    "sub",
                    "truediv",
                    "div",
                    "relu",
                    "gelu",
                    "silu",
                    "sigmoid",
                    "tanh",
                    "exp",
                    "log",
                    "sqrt",
                    "neg",
                    "abs",
                } or any(
                    f in str(fn)
                    for f in [
                        "add",
                        "mul",
                        "sub",
                        "div",
                        "relu",
                        "gelu",
                        "silu",
                        "sigmoid",
                        "tanh",
                        "exp",
                    ]
                ):
                    captured.append(
                        Operation(
                            node.name, "elementwise", in_shapes or (out_shape,), out_shape, dtype_b
                        )
                    )
                elif fn_name in {"layer_norm", "batch_norm"} or fn in {
                    torch.nn.functional.layer_norm,
                    torch.nn.functional.batch_norm,
                }:
                    kind = "layernorm" if "layer" in fn_name else "batchnorm"
                    captured.append(
                        Operation(node.name, kind, in_shapes or (out_shape,), out_shape, dtype_b)
                    )
                elif fn_name == "softmax" or fn == torch.nn.functional.softmax:
                    captured.append(
                        Operation(
                            node.name, "softmax", in_shapes or (out_shape,), out_shape, dtype_b
                        )
                    )
                elif fn_name in _FUNCTIONAL_CONVS and len(in_shapes) >= 2:
                    transposed = fn_name.startswith("conv_transpose")
                    # conv{n}d(input, weight, bias, stride, padding, dilation, groups);
                    # conv_transpose{n}d(..., stride, padding, output_padding, groups, dilation).
                    group_index = 7 if transposed else 6
                    groups = int(
                        node.kwargs.get(
                            "groups",
                            node.args[group_index] if len(node.args) > group_index else 1,
                        )
                    )
                    kind, conv_inputs, conv_output, conv_attrs = _conv_shapes(
                        in_shapes[0], in_shapes[1], out_shape, groups, transposed
                    )
                    captured.append(
                        Operation(node.name, kind, conv_inputs, conv_output, dtype_b, conv_attrs)
                    )
                elif fn_name == "scaled_dot_product_attention" or fn == getattr(
                    torch.nn.functional, "scaled_dot_product_attention", None
                ):
                    all_args = list(node.args) + list(node.kwargs.values())
                    in_shapes = tuple(
                        s for s in (get_shape(arg) for arg in all_args) if s is not None
                    )
                    q_shape = in_shapes[0] if in_shapes else out_shape
                    k_shape = in_shapes[1] if len(in_shapes) > 1 else q_shape
                    is_causal = node.kwargs.get(
                        "is_causal", node.args[5] if len(node.args) > 5 else False
                    )
                    captured.append(
                        Operation(
                            node.name,
                            "attention",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            _sdpa_attrs(q_shape, k_shape, is_causal is True),
                        )
                    )
                elif fn_name in {"rms_norm"} or fn == getattr(
                    torch.nn.functional, "rms_norm", None
                ):
                    captured.append(
                        Operation(
                            node.name, "rmsnorm", in_shapes or (out_shape,), out_shape, dtype_b
                        )
                    )

        return captured if captured else super().operations(model, example_inputs)
