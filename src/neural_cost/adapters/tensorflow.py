"""TensorFlow/Keras adapter (available only when TensorFlow is installed)."""

from collections.abc import Sequence
from statistics import median
from time import perf_counter_ns
from typing import Any

from ..operations import Operation
from ..profiler import Measurement
from .base import FrameworkAdapter


def _parameter_count(layer: Any) -> int:
    """Parameter elements, excluding BatchNorm moving statistics.

    ``count_params`` includes BatchNorm's moving mean/variance, which are buffers
    rather than parameters (PyTorch and JAX never count them).  Frozen layers
    still count: being non-trainable does not make weights disappear.
    """
    if type(layer).__name__ == "BatchNormalization":
        weights = [
            w
            for w in (getattr(layer, "gamma", None), getattr(layer, "beta", None))
            if w is not None
        ]
    else:
        weights = list(getattr(layer, "weights", ()))
    total = 0
    for weight in weights:
        count = 1
        for dim in weight.shape:
            count *= int(dim)
        total += count
    return total


def _layer_types(tf: Any, *names: str) -> tuple[type, ...]:
    """Keras layer classes by name, skipping ones this Keras version lacks."""
    layers = tf.keras.layers
    return tuple(getattr(layers, name) for name in names if hasattr(layers, name))


def _channels_first(layer: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    if getattr(layer, "data_format", "channels_last") == "channels_first":
        return shape
    return (shape[0], shape[-1], *shape[1:-1])


def _weight_shape(weight: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in weight.shape)


def _keras_conv_operations(
    tf: Any, layer: Any, x: tuple[int, ...], y: tuple[int, ...], dtype_bytes: int
) -> list[Operation]:
    """Portable channels-first conv records for any Keras convolution layer.

    Keras kernels are ``(*k, C_in / groups, C_out)`` for convolutions,
    ``(*k, C_out, C_in)`` for transposed convolutions and ``(*k, C_in, mult)``
    for depthwise kernels.  A separable convolution becomes its depthwise and
    pointwise halves.  The transposed/depthwise/separable checks come first
    because older Keras versions subclass them from the plain ``ConvND``.
    """
    data, out = _channels_first(layer, x), _channels_first(layer, y)
    rank = len(data) - 2

    def record(
        name: str,
        inputs: tuple[int, ...],
        kernel: tuple[int, ...],
        output: tuple[int, ...],
        weight: Any,
        parameter_count: int,
        **attrs: Any,
    ) -> Operation:
        kind = "conv2d" if rank == 2 and not attrs.get("transposed") else "conv"
        return Operation(
            name,
            kind,
            (inputs, kernel),
            output,
            dtype_bytes,
            {
                "parameter_bytes": parameter_count * dtype_bytes,
                "parameter_id": id(weight),
                **attrs,
            },
        )

    transposed = _layer_types(tf, "Conv1DTranspose", "Conv2DTranspose", "Conv3DTranspose")
    depthwise = _layer_types(tf, "DepthwiseConv1D", "DepthwiseConv2D")
    separable = _layer_types(tf, "SeparableConv1D", "SeparableConv2D")
    if transposed and isinstance(layer, transposed):
        k = _weight_shape(layer.kernel)
        kernel = (k[-2], k[-1], *k[:-2])
        return [
            record(
                layer.name,
                data,
                kernel,
                out,
                layer.kernel,
                _parameter_count(layer),
                transposed=True,
            )
        ]
    if separable and isinstance(layer, separable):
        dw_weight, pw_weight = layer.depthwise_kernel, layer.pointwise_kernel
        dw, pw = _weight_shape(dw_weight), _weight_shape(pw_weight)
        channels, multiplier = dw[-2], dw[-1]
        middle = (out[0], channels * multiplier, *out[2:])
        dw_params = channels * multiplier
        for size in dw[:-2]:
            dw_params *= size
        return [
            record(
                f"{layer.name}.depthwise",
                data,
                (channels * multiplier, 1, *dw[:-2]),
                middle,
                dw_weight,
                dw_params,
                groups=channels,
            ),
            record(
                f"{layer.name}.pointwise",
                middle,
                (pw[-1], pw[-2], *pw[:-2]),
                out,
                pw_weight,
                _parameter_count(layer) - dw_params,
            ),
        ]
    if depthwise and isinstance(layer, depthwise):
        weight = getattr(layer, "depthwise_kernel", None)
        if weight is None:  # Keras 3 names it ``kernel``
            weight = layer.kernel
        dw = _weight_shape(weight)
        channels, multiplier = dw[-2], dw[-1]
        return [
            record(
                layer.name,
                data,
                (channels * multiplier, 1, *dw[:-2]),
                out,
                weight,
                _parameter_count(layer),
                groups=channels,
            )
        ]
    k = _weight_shape(layer.kernel)
    return [
        record(
            layer.name,
            data,
            (k[-1], k[-2], *k[:-2]),
            out,
            layer.kernel,
            _parameter_count(layer),
            groups=int(getattr(layer, "groups", 1)),
        )
    ]


class TensorFlowAdapter(FrameworkAdapter):
    """Capture Keras layer calls (dense, convolution, pooling, ...) through temporary hooks."""

    name = "tensorflow"

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import tensorflow as tf
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[tensorflow] to use TensorFlowAdapter") from error
        captured: list[Operation] = []
        original_calls: list[tuple[Any, Any]] = []

        def shape(tensor: Any) -> tuple[int, ...]:
            values = tuple(tensor.shape)
            if any(value is None for value in values):
                raise ValueError("example inputs must provide concrete TensorFlow shapes")
            return tuple(int(value) for value in values)

        # Keras 3 removed ``Model.submodules``.  ``layers`` is the public API;
        # walking it also supports nested Sequential/Model containers.
        pending = list(getattr(model, "layers", ()))
        seen: set[int] = set()
        layers: list[Any] = []
        while pending:
            layer = pending.pop(0)
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            layers.append(layer)
            pending.extend(getattr(layer, "layers", ()))

        conv_types = _layer_types(
            tf,
            "Conv1D",
            "Conv2D",
            "Conv3D",
            "Conv1DTranspose",
            "Conv2DTranspose",
            "Conv3DTranspose",
            "DepthwiseConv1D",
            "DepthwiseConv2D",
            "SeparableConv1D",
            "SeparableConv2D",
        )
        window_pool_types = _layer_types(
            tf,
            *(f"{kind}Pooling{dims}D" for kind in ("Max", "Average") for dims in (1, 2, 3)),
        )
        global_pool_types = _layer_types(
            tf,
            *(f"Global{kind}Pooling{dims}D" for kind in ("Max", "Average") for dims in (1, 2, 3)),
        )

        for layer in layers:
            if not isinstance(
                layer,
                (
                    tf.keras.layers.Dense,
                    *conv_types,
                    *window_pool_types,
                    *global_pool_types,
                    tf.keras.layers.Activation,
                    tf.keras.layers.ReLU,
                    tf.keras.layers.LeakyReLU,
                    tf.keras.layers.ELU,
                    tf.keras.layers.Softmax,
                    tf.keras.layers.BatchNormalization,
                    tf.keras.layers.LayerNormalization,
                    tf.keras.layers.Embedding,
                    tf.keras.layers.LSTM,
                    tf.keras.layers.GRU,
                    tf.keras.layers.SimpleRNN,
                    tf.keras.layers.MultiHeadAttention,
                ),
            ):
                continue
            previous_call = layer.call

            def wrapped(
                inputs: Any,
                *args: Any,
                _layer: Any = layer,
                _call: Any = previous_call,
                **kwargs: Any,
            ) -> Any:
                output = _call(inputs, *args, **kwargs)
                if isinstance(_layer, tf.keras.layers.Dense):
                    captured.append(
                        Operation(
                            _layer.name,
                            "linear",
                            (shape(inputs), shape(_layer.kernel)),
                            shape(output),
                            inputs.dtype.size,
                            {
                                "parameter_bytes": _parameter_count(_layer) * inputs.dtype.size,
                                "parameter_id": id(_layer.kernel),
                            },
                        )
                    )
                elif isinstance(_layer, conv_types):
                    captured.extend(
                        _keras_conv_operations(
                            tf, _layer, shape(inputs), shape(output), inputs.dtype.size
                        )
                    )
                elif isinstance(
                    _layer,
                    (
                        tf.keras.layers.Activation,
                        tf.keras.layers.ReLU,
                        tf.keras.layers.LeakyReLU,
                        tf.keras.layers.ELU,
                    ),
                ):
                    captured.append(
                        Operation(
                            _layer.name,
                            "elementwise",
                            (shape(inputs),),
                            shape(output),
                            inputs.dtype.size,
                        )
                    )
                elif isinstance(_layer, tf.keras.layers.Softmax):
                    captured.append(
                        Operation(
                            _layer.name,
                            "softmax",
                            (shape(inputs),),
                            shape(output),
                            inputs.dtype.size,
                        )
                    )
                elif isinstance(
                    _layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)
                ):
                    kind = (
                        "batchnorm"
                        if isinstance(_layer, tf.keras.layers.BatchNormalization)
                        else "layernorm"
                    )
                    attrs = {}
                    if hasattr(_layer, "gamma") and _layer.gamma is not None:
                        attrs["parameter_bytes"] = _parameter_count(_layer) * inputs.dtype.size
                        attrs["parameter_id"] = id(_layer.gamma)
                    x = shape(inputs)
                    y = shape(output)
                    if len(x) == 4:
                        in_shape = (x[0], x[3], x[1], x[2])
                        out_shape = (y[0], y[3], y[1], y[2])
                    else:
                        in_shape = x
                        out_shape = y
                    captured.append(
                        Operation(
                            _layer.name, kind, (in_shape,), out_shape, inputs.dtype.size, attrs
                        )
                    )
                elif isinstance(_layer, window_pool_types):
                    captured.append(
                        Operation(
                            _layer.name,
                            "pooling",
                            (_channels_first(_layer, shape(inputs)),),
                            _channels_first(_layer, shape(output)),
                            inputs.dtype.size,
                            {"kernel_size": tuple(int(size) for size in _layer.pool_size)},
                        )
                    )
                elif isinstance(_layer, global_pool_types):
                    x = _channels_first(_layer, shape(inputs))
                    captured.append(
                        Operation(
                            _layer.name,
                            "pooling",
                            (x,),
                            shape(output),
                            inputs.dtype.size,
                            {"kernel_size": x[2:]},
                        )
                    )
                elif isinstance(_layer, tf.keras.layers.Embedding):
                    y = shape(output)
                    captured.append(
                        Operation(
                            _layer.name,
                            "embedding",
                            (shape(inputs),),
                            y,
                            _layer.embeddings.dtype.itemsize,
                            {
                                "parameter_bytes": _parameter_count(_layer)
                                * _layer.embeddings.dtype.itemsize,
                                "parameter_id": id(_layer.embeddings),
                            },
                        )
                    )
                elif isinstance(
                    _layer,
                    (tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.SimpleRNN),
                ):
                    # Capture as a linear operation representing the combined gate projections.
                    # LSTM: 4 gates; GRU: 3 gates; SimpleRNN: 1.
                    if isinstance(_layer, tf.keras.layers.LSTM):
                        gates = 4
                    elif isinstance(_layer, tf.keras.layers.GRU):
                        gates = 3
                    else:
                        gates = 1
                    x = shape(inputs)
                    units = _layer.units
                    # inputs shape: (batch, timesteps, features) or (batch, features)
                    if len(x) == 3:
                        batch, timesteps, in_features = x
                    else:
                        batch, in_features = x[0], x[-1]
                        timesteps = 1
                    flat_batch = batch * timesteps
                    param_bytes = _parameter_count(_layer) * inputs.dtype.size
                    # Keras 3 stores weights on the inner cell; Keras 2 stores on the layer.
                    cell = getattr(_layer, "cell", _layer)
                    kernel = getattr(_layer, "kernel", None) or getattr(cell, "kernel", None)
                    rec_kernel = getattr(_layer, "recurrent_kernel", None) or getattr(
                        cell, "recurrent_kernel", None
                    )
                    kernel_id = id(kernel) if kernel is not None else id(_layer)
                    rec_kernel_id = id(rec_kernel) if rec_kernel is not None else id(cell)
                    # input-hidden
                    captured.append(
                        Operation(
                            f"{_layer.name}.ih",
                            "linear",
                            ((flat_batch, in_features), (in_features, gates * units)),
                            (flat_batch, gates * units),
                            inputs.dtype.size,
                            {"parameter_bytes": param_bytes, "parameter_id": kernel_id},
                        )
                    )
                    # hidden-hidden: T dependent (batch, units) GEMMs
                    hh_attrs: dict[str, Any] = {"parameter_bytes": 0, "parameter_id": rec_kernel_id}
                    if timesteps > 1:
                        hh_attrs["repeat"] = timesteps
                    captured.append(
                        Operation(
                            f"{_layer.name}.hh",
                            "linear",
                            ((batch, units), (units, gates * units)),
                            (batch, gates * units),
                            inputs.dtype.size,
                            hh_attrs,
                        )
                    )
                elif isinstance(_layer, tf.keras.layers.MultiHeadAttention):
                    # MultiHeadAttention call signature: call(query, value, key=None, ...)
                    # inputs here is the query tensor; capture as an attention operation.
                    query = inputs
                    try:
                        q = shape(query)
                    except (ValueError, AttributeError):
                        return output
                    num_heads = _layer.num_heads
                    key_dim = _layer.key_dim
                    embed_dim = q[-1] if len(q) >= 1 else key_dim * num_heads
                    seq_len = q[-2] if len(q) >= 2 else 1
                    batch = q[0] if len(q) >= 3 else 1
                    # call(query, value, key=None, query_mask, value_mask, key_mask,
                    #      attention_mask, return_attention_scores, training, use_causal_mask)
                    value = args[0] if args else kwargs.get("value")
                    key = args[1] if len(args) > 1 else kwargs.get("key")
                    source_kv = key if key is not None else value
                    try:
                        kv = shape(source_kv) if source_kv is not None else q
                    except (ValueError, AttributeError):
                        kv = q
                    kv_len = kv[-2] if len(kv) >= 2 else seq_len
                    use_causal = args[8] if len(args) > 8 else kwargs.get("use_causal_mask", False)
                    mha_attrs: dict[str, Any] = {}
                    if kv_len != seq_len:
                        mha_attrs["kv_len"] = kv_len
                    if use_causal is True:
                        # Keras builds a top-left aligned lower-triangular mask.
                        mha_attrs.update(causal=True, q_offset=0)
                    captured.append(
                        Operation(
                            _layer.name,
                            "attention",
                            (q,),
                            (batch, seq_len, embed_dim),
                            query.dtype.size,
                            {
                                "num_heads": num_heads,
                                "seq_len": seq_len,
                                "head_dim": key_dim,
                                "proj_dim": num_heads * key_dim,
                                "parameter_bytes": _parameter_count(_layer) * query.dtype.size,
                                "parameter_id": id(_layer),
                                **mha_attrs,
                            },
                        )
                    )
                return output

            original_calls.append((layer, previous_call))
            layer.call = wrapped
        try:
            model(*example_inputs, training=False)
        finally:
            for layer, previous_call in original_calls:
                layer.call = previous_call
        return captured

    def benchmark(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Benchmark eager TensorFlow and collect supported GPU allocator telemetry."""
        try:
            import tensorflow as tf
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[tensorflow] to use TensorFlowAdapter") from error
        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")

        gpus = tf.config.list_logical_devices("GPU")
        memory_device = "GPU:0" if gpus else None

        def wait() -> None:
            async_wait = getattr(tf.experimental, "async_wait", None)
            if async_wait is not None:
                async_wait()

        if memory_device is not None:
            try:
                tf.config.experimental.reset_memory_stats(memory_device)
            except (RuntimeError, ValueError):
                memory_device = None
        for _ in range(warmup):
            function(*args, **kwargs)
        wait()
        samples: list[float] = []
        for _ in range(repeats):
            start = perf_counter_ns()
            function(*args, **kwargs)
            wait()
            samples.append((perf_counter_ns() - start) / 1_000_000_000)
        info: dict[str, int] = {}
        if memory_device is not None:
            try:
                info = tf.config.experimental.get_memory_info(memory_device)
            except (RuntimeError, ValueError):
                pass
        return Measurement(
            median(samples),
            tuple(samples),
            info.get("peak"),
            gpus[0].name if gpus else "cpu",
            info.get("current"),
        )
