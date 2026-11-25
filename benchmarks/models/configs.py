"""Scale-tier dimensions shared by the PyTorch, JAX and TensorFlow benchmark models.

Every framework builds its models from these tables, so a given architecture and
scale is the same network, with the same FLOPs and parameters, in all three.
"""

from __future__ import annotations

from typing import Any

NUM_CLASSES = 10
IN_CHANNELS = 3

CONVNEXT: dict[str, dict[str, Any]] = {
    "micro": {"depths": [1, 1], "dims": [32, 64], "stem": 2, "img_size": 32},
    "standard": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768], "stem": 4, "img_size": 224},
    "production": {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024],
        "stem": 4,
        "img_size": 224,
    },
}

VIT: dict[str, dict[str, Any]] = {
    "micro": {"img_size": 32, "patch_size": 4, "embed_dim": 128, "num_heads": 4, "depth": 2},
    "standard": {"img_size": 224, "patch_size": 16, "embed_dim": 384, "num_heads": 6, "depth": 6},
    "production": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 768,
        "num_heads": 12,
        "depth": 12,
    },
}
VIT_MLP_RATIO = 4

DEEP_DNN: dict[str, dict[str, int]] = {
    "micro": {"in_features": 784, "hidden_features": 128, "out_features": 128},
    "standard": {"in_features": 1024, "hidden_features": 4096, "out_features": 1024},
    "production": {"in_features": 2048, "hidden_features": 8192, "out_features": 2048},
}

TRANSFORMER: dict[str, dict[str, int]] = {
    "micro": {"seq_len": 32, "embed_dim": 128, "num_heads": 4, "num_layers": 2, "ffn_dim": 512},
    "standard": {
        "seq_len": 512,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 6,
        "ffn_dim": 3072,
    },
    "production": {
        "seq_len": 1024,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 12,
        "ffn_dim": 4096,
    },
}


def scale_config(table: dict[str, Any], scale: str) -> Any:
    """Return ``table[scale]``, rejecting unknown scale tiers."""
    if scale not in table:
        raise ValueError(f"Unknown scale: {scale}")
    return table[scale]


def describe(arch: str, scale: str) -> str | None:
    """One-line description of a modern architecture at ``scale``, for reports."""
    if arch == "ConvNeXt":
        c = scale_config(CONVNEXT, scale)
        return (
            f"{len(c['dims'])}-stage ConvNeXt, depths {c['depths']}, dims {c['dims']} "
            f"({c['stem']}×{c['stem']} stem; 7×7 depthwise → LayerNorm → 1×1 (dim×4) → GELU → 1×1), "
            f"input {c['img_size']}×{c['img_size']}×3"
        )
    if arch == "ViT":
        c = scale_config(VIT, scale)
        return (
            f"{c['depth']}-block pre-norm ViT, {c['patch_size']}×{c['patch_size']} patches, "
            f"embed {c['embed_dim']}, {c['num_heads']} heads, MLP×{VIT_MLP_RATIO}, "
            f"input {c['img_size']}×{c['img_size']}×3"
        )
    if arch in {"FF DNN", "Deep DNN"}:
        c = scale_config(DEEP_DNN, scale)
        return (
            f"{c['in_features']} → {c['hidden_features']} → {c['hidden_features']} → "
            f"{c['out_features']} → {NUM_CLASSES}, GELU + LayerNorm"
        )
    if arch == "Transformer":
        c = scale_config(TRANSFORMER, scale)
        return (
            f"{c['num_layers']}-layer pre-norm encoder (SDPA, {c['num_heads']} heads, "
            f"FFN {c['ffn_dim']}, LayerNorm), embed {c['embed_dim']}, seq {c['seq_len']}"
        )
    return None
