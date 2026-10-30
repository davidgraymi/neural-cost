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

    ``use_profiler=False`` makes :meth:`benchmark` time without ``torch.profiler``.
    The profiler adds per-operator overhead, so use it for validation and
    gap attribution where the wall-clock time itself is the measurement.
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

        captured: list[Operation] = []
        hooks = []

        def dtype_bytes(tensor: Any) -> int:
            return tensor.element_size()

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
                elif isinstance(module, torch.nn.Conv2d):
                    if not isinstance(output, torch.Tensor):
                        return
                    captured.append(
                        Operation(
                            name,
                            "conv2d",
                            (tuple(source.shape), tuple(module.weight.shape)),
                            tuple(output.shape),
                            dtype_bytes(source),
                            {
                                "parameter_bytes": sum(
                                    p.numel() * p.element_size()
                                    for p in module.parameters(recurse=False)
                                ),
                                "parameter_id": id(module.weight),
                                "groups": module.groups,
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
                            },
                        )
                    )
                elif (
                    isinstance(
                        module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
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
            torch.nn.Conv2d,
            torch.nn.Embedding,
            torch.nn.RNN,
            torch.nn.GRU,
            torch.nn.LSTM,
            torch.nn.MultiheadAttention,
            torch.nn.LayerNorm,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
        ]
        if hasattr(torch.nn, "RMSNorm"):
            _tracked_list.append(torch.nn.RMSNorm)
        _TRACKED = tuple(_tracked_list)
        # MultiheadAttention may dispatch to scaled_dot_product_attention internally;
        # its hook already accounts for the full attention cost, so suppress the
        # functional capture while inside one.
        mha_depth = [0]

        def enter_mha(*_args: Any) -> None:
            mha_depth[0] += 1

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
                    hooks.append(module.register_forward_pre_hook(enter_mha))
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
                    if len(q_shape) == 4:
                        _, h, l, d_h = q_shape
                    elif len(q_shape) == 3:
                        _, l, d = q_shape
                        h, d_h = 1, d
                    else:
                        h, l, d_h = 1, 1, 1
                    captured.append(
                        Operation(
                            f"scaled_dot_product_attention_{sdpa_counter[0]}",
                            "attention",
                            (q_shape, k_shape, v_shape),
                            tuple(out.shape),
                            dtype_bytes(query),
                            {
                                "num_heads": h,
                                "seq_len": l,
                                "head_dim": d_h,
                                "include_projections": False,
                            },
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
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Benchmark execution and capture CUDA allocator statistics when present."""
        if not getattr(self, "use_profiler", True):
            return self._time(function, *args, warmup=warmup, repeats=repeats, **kwargs)
        return self.trace(function, *args, warmup=warmup, repeats=repeats, **kwargs)

    def _time(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Synchronized wall-clock timing with no profiler attached."""
        import torch

        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")
        device = next((arg.device for arg in args if isinstance(arg, torch.Tensor)), None)
        kind = device.type if device is not None else "cpu"

        def sync() -> None:
            if kind == "cuda":
                torch.cuda.synchronize(device)
            elif kind == "mps":
                torch.mps.synchronize()

        if kind == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            function(*args, **kwargs)
        sync()
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            function(*args, **kwargs)
            sync()
            samples.append((perf_counter_ns() - start) / 1_000_000_000)
        peak = torch.cuda.max_memory_allocated(device) if kind == "cuda" else None
        return Measurement(median(samples), tuple(samples), peak, str(device or "cpu"))

    def trace(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Profile PyTorch events and CUDA memory for a representative workload.

        The trace is aggregated over ``repeats`` iterations.  CUDA timings are
        synchronized before every sample, and CUDA allocator counters are
        reported separately from the static tensor-memory estimate.
        """
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error
        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")
        device = next((arg.device for arg in args if isinstance(arg, torch.Tensor)), None)
        is_cuda = device is not None and device.type == "cuda"
        is_mps = device is not None and device.type == "mps"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        elif is_mps:
            torch.mps.synchronize()
        for _ in range(warmup):
            function(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize(device)
        elif is_mps:
            torch.mps.synchronize()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        samples: list[float] = []
        with torch.profiler.profile(
            activities=activities,
            profile_memory=is_cuda or (device is None or device.type == "cpu"),
        ) as trace:
            for _ in range(repeats):
                start = perf_counter_ns()
                function(*args, **kwargs)
                if is_cuda:
                    torch.cuda.synchronize(device)
                elif is_mps:
                    torch.mps.synchronize()
                samples.append((perf_counter_ns() - start) / 1_000_000_000)
        events = trace.events()
        if is_cuda:
            peak = torch.cuda.max_memory_allocated(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            device_time = sum(event.cuda_time_total for event in events) / 1_000_000
            event_count = sum(event.cuda_time_total > 0 for event in events)
        elif is_mps:
            peak = torch.mps.current_allocated_memory()
            allocated = peak
            reserved = torch.mps.driver_allocated_memory()
            device_time = sum(event.cpu_time_total for event in events) / 1_000_000
            event_count = len(events)
        else:
            cpu_allocs = [e.cpu_memory_usage for e in events if e.cpu_memory_usage > 0]
            param_bytes = 0
            if hasattr(function, "parameters") and callable(function.parameters):
                param_bytes = sum(p.numel() * p.element_size() for p in function.parameters())
            peak = (param_bytes + sum(cpu_allocs)) if cpu_allocs else None
            allocated = peak
            reserved = peak
            device_time = sum(event.cpu_time_total for event in events) / 1_000_000
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

        try:
            gm = torch.fx.symbolic_trace(model)
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
                elif isinstance(mod, torch.nn.Conv2d):
                    if in_shapes:
                        captured.append(
                            Operation(
                                node.name,
                                "conv2d",
                                (in_shapes[0], tuple(mod.weight.shape)),
                                out_shape,
                                dtype_b,
                                {
                                    "parameter_bytes": sum(
                                        p.numel() * p.element_size()
                                        for p in mod.parameters(recurse=False)
                                    ),
                                    "parameter_id": id(mod.weight),
                                    "groups": mod.groups,
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
                elif isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
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
                elif isinstance(
                    mod, (torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)
                ):
                    k = getattr(mod, "kernel_size", (1, 1))
                    k_tuple = k if isinstance(k, tuple) else (k, k)
                    captured.append(
                        Operation(
                            node.name,
                            "pooling",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            {"kernel_size": k_tuple},
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
                elif (
                    fn_name in {"conv2d"}
                    or fn in {torch.nn.functional.conv2d, getattr(torch, "conv2d", None)}
                ) and len(in_shapes) >= 2:
                    groups = int(
                        node.kwargs.get("groups", node.args[6] if len(node.args) > 6 else 1)
                    )
                    captured.append(
                        Operation(
                            node.name,
                            "conv2d",
                            in_shapes[:2],
                            out_shape,
                            dtype_b,
                            {"groups": groups},
                        )
                    )
                elif fn_name == "scaled_dot_product_attention" or fn == getattr(
                    torch.nn.functional, "scaled_dot_product_attention", None
                ):
                    all_args = list(node.args) + list(node.kwargs.values())
                    in_shapes = tuple(
                        s for s in (get_shape(arg) for arg in all_args) if s is not None
                    )
                    q_shape = in_shapes[0] if in_shapes else out_shape
                    if len(q_shape) == 4:
                        _, h, l, d_h = q_shape
                    elif len(q_shape) == 3:
                        _, l, d = q_shape
                        h, d_h = 1, d
                    else:
                        h, l, d_h = 1, 1, 1
                    captured.append(
                        Operation(
                            node.name,
                            "attention",
                            in_shapes or (out_shape,),
                            out_shape,
                            dtype_b,
                            {
                                "num_heads": h,
                                "seq_len": l,
                                "head_dim": d_h,
                                "include_projections": False,
                            },
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
