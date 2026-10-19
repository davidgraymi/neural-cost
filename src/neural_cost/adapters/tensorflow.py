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


class TensorFlowAdapter(FrameworkAdapter):
    """Capture Keras Dense and Conv2D layer calls through temporary hooks."""

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

        for layer in layers:
            if not isinstance(
                layer,
                (
                    tf.keras.layers.Dense,
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.Activation,
                    tf.keras.layers.ReLU,
                    tf.keras.layers.LeakyReLU,
                    tf.keras.layers.ELU,
                    tf.keras.layers.Softmax,
                    tf.keras.layers.BatchNormalization,
                    tf.keras.layers.LayerNormalization,
                    tf.keras.layers.MaxPooling2D,
                    tf.keras.layers.AveragePooling2D,
                    tf.keras.layers.GlobalAveragePooling2D,
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
                elif isinstance(_layer, tf.keras.layers.Conv2D):
                    # TF is NHWC while the portable Conv2D record is NCHW.
                    x = shape(inputs)
                    y = shape(output)
                    kernel = shape(_layer.kernel)  # HWIO -> OIHW
                    captured.append(
                        Operation(
                            _layer.name,
                            "conv2d",
                            (
                                (x[0], x[3], x[1], x[2]),
                                (kernel[3], kernel[2], kernel[0], kernel[1]),
                            ),
                            (y[0], y[3], y[1], y[2]),
                            inputs.dtype.size,
                            {
                                "parameter_bytes": _parameter_count(_layer) * inputs.dtype.size,
                                "parameter_id": id(_layer.kernel),
                            },
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
                elif isinstance(
                    _layer, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)
                ):
                    x = shape(inputs)
                    y = shape(output)
                    captured.append(
                        Operation(
                            _layer.name,
                            "pooling",
                            ((x[0], x[3], x[1], x[2]),),
                            (y[0], y[3], y[1], y[2]),
                            inputs.dtype.size,
                            {"kernel_size": tuple(_layer.pool_size)},
                        )
                    )
                elif isinstance(_layer, tf.keras.layers.GlobalAveragePooling2D):
                    x = shape(inputs)
                    y = shape(output)
                    captured.append(
                        Operation(
                            _layer.name,
                            "pooling",
                            ((x[0], x[3], x[1], x[2]),),
                            y,
                            inputs.dtype.size,
                            {"kernel_size": (x[1], x[2])},
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
