"""Benchmark model architectures, registry, and scale tier configurations."""

from __future__ import annotations

from typing import Any, Literal

ScaleTier = Literal["micro", "standard", "production"]

SCALE_TIERS: list[ScaleTier] = ["micro", "standard", "production"]

MODERN_ARCHITECTURES: list[str] = ["FF DNN", "ConvNeXt", "ViT", "Transformer"]
LEGACY_ARCHITECTURES: list[str] = ["CNN", "RNN", "LSTM"]
ALL_ARCHITECTURES: list[str] = MODERN_ARCHITECTURES + LEGACY_ARCHITECTURES

try:
    from .operators import RMSNorm, SDPASelfAttention, SwiGLU
except ImportError:
    pass

SCALE_BATCH_SIZES: dict[str, list[int]] = {
    "micro": [1, 8, 32, 128],
    "standard": [1, 4, 16, 64],
    "production": [1, 2, 4, 8, 16],
}
STANDARD_BATCH_SIZES: list[int] = [1, 4, 16, 64]

QUICK_BATCH_SIZES: dict[str, list[int]] = {
    "micro": [8, 64],
    "standard": [4, 16],
    "production": [1, 4],
}


def get_batch_sizes(scale: str = "standard", quick: bool = False) -> list[int]:
    """Return default batch sizes for the specified scale tier."""
    if scale not in SCALE_TIERS:
        raise ValueError(f"Invalid scale tier: {scale!r}; must be one of {SCALE_TIERS}")
    if quick:
        return QUICK_BATCH_SIZES.get(scale, [4, 16])
    return SCALE_BATCH_SIZES.get(scale, [1, 4, 16, 64])


def get_available_models(include_legacy: bool = False) -> list[str]:
    """Return active benchmark architecture names."""
    if include_legacy:
        return list(ALL_ARCHITECTURES)
    return list(MODERN_ARCHITECTURES)


def _normalize_name(name: str) -> str:
    norm = name.strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "convnext": "convnext",
        "vit": "vit",
        "vision transformer": "vit",
        "deep dnn": "deep_dnn",
        "ff dnn": "deep_dnn",
        "transformer": "transformer",
        "cnn": "cnn",
        "rnn": "rnn",
        "lstm": "lstm",
    }
    if norm in mapping:
        return mapping[norm]
    # Direct match attempt
    for key, val in mapping.items():
        if key in norm:
            return val
    return name.strip().lower()


def get_model(
    name: str,
    framework: str = "torch",
    scale: str = "standard",
    batch: int = 1,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Retrieve model instance and dummy input tuple for a given framework and scale tier."""
    if scale not in SCALE_TIERS:
        raise ValueError(f"Invalid scale tier: {scale!r}; must be one of {SCALE_TIERS}")

    fw = framework.lower().strip()
    arch_key = _normalize_name(name)

    if fw in {"torch", "pytorch"}:
        from .torch_models import TORCH_BUILDERS

        if arch_key not in TORCH_BUILDERS:
            raise KeyError(
                f"Unknown PyTorch architecture: {name!r} (normalized: {arch_key!r}). "
                f"Available: {list(TORCH_BUILDERS.keys())}"
            )
        return TORCH_BUILDERS[arch_key](scale=scale, batch=batch, **kwargs)

    elif fw == "jax":
        from .jax_models import JAX_BUILDERS

        if arch_key not in JAX_BUILDERS:
            raise KeyError(
                f"Unknown JAX architecture: {name!r} (normalized: {arch_key!r}). "
                f"Available: {list(JAX_BUILDERS.keys())}"
            )
        return JAX_BUILDERS[arch_key](scale=scale, batch=batch, **kwargs)

    elif fw in {"tf", "tensorflow"}:
        from .tf_models import TF_BUILDERS

        if arch_key not in TF_BUILDERS:
            raise KeyError(
                f"Unknown TensorFlow architecture: {name!r} (normalized: {arch_key!r}). "
                f"Available: {list(TF_BUILDERS.keys())}"
            )
        return TF_BUILDERS[arch_key](scale=scale, batch=batch, **kwargs)

    raise ValueError(f"Unsupported framework: {framework!r}")
