"""PyTorch benchmark model architectures across scale tiers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import CONVNEXT, DEEP_DNN, TRANSFORMER, VIT, VIT_MLP_RATIO, scale_config

# ---------------------------------------------------------------------------
# Common Sub-modules
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """LayerNorm across channels for 2D feature maps (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class SDPASelfAttention(nn.Module):
    """Multi-Head Self-Attention using hardware-accelerated F.scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 1. ConvNeXt Architecture
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with 7x7 depthwise convolution and inverted bottleneck."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + x


class ConvNeXt(nn.Module):
    """ConvNeXt architecture supporting micro, standard, and production scale tiers."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        stem_kernel: int = 4,
        stem_stride: int = 4,
    ) -> None:
        super().__init__()
        depths = depths or [3, 3, 9, 3]
        dims = dims or [96, 192, 384, 768]

        # Stem downsampler
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=stem_kernel, stride=stem_stride),
            LayerNorm2d(dims[0]),
        )

        # Stages and downsamplers
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i in range(len(depths)):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

            if i < len(depths) - 1:
                downsampler = nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamplers.append(downsampler)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2. Vision Transformer (ViT) Architecture
# ---------------------------------------------------------------------------


class ViTBlock(nn.Module):
    """Vision Transformer Encoder Block with SDPA attention and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SDPASelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer architecture."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 384,
        num_heads: int = 6,
        depth: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, embed_dim, H/patch, W/patch) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        # Global average pool over sequence
        return self.head(x.mean(dim=1))


# ---------------------------------------------------------------------------
# 3. Deep DNN Architecture
# ---------------------------------------------------------------------------


class DeepDNN(nn.Module):
    """Deep Feed-Forward DNN architecture."""

    def __init__(
        self,
        in_features: int = 1024,
        hidden_features: int = 4096,
        out_features: int = 1024,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(hidden_features)

        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.act2 = nn.GELU()
        self.ln2 = nn.LayerNorm(hidden_features)

        self.fc3 = nn.Linear(hidden_features, out_features)
        self.act3 = nn.GELU()
        self.ln3 = nn.LayerNorm(out_features)

        self.head = nn.Linear(out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(self.act1(self.fc1(x)))
        x = self.ln2(self.act2(self.fc2(x)))
        x = self.ln3(self.act3(self.fc3(x)))
        return self.head(x)


# ---------------------------------------------------------------------------
# 4. Standard Transformer Encoder Architecture
# ---------------------------------------------------------------------------


from .operators import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with SDPA attention, RMSNorm, and SwiGLU support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_rmsnorm: bool = False,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(embed_dim) if use_rmsnorm else nn.LayerNorm(embed_dim)
        self.attn = SDPASelfAttention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim) if use_rmsnorm else nn.LayerNorm(embed_dim)
        if use_swiglu:
            self.ffn = SwiGLU(embed_dim, ffn_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Transformer Encoder model using hardware-accelerated fused SDPA (scaled_dot_product_attention)."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        ffn_dim: int = 3072,
        num_classes: int = 10,
        use_rmsnorm: bool = False,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embed_dim) if use_rmsnorm else nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass using scaled_dot_product_attention (SDPA)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


# ---------------------------------------------------------------------------
# 5. Legacy Models (CNN, RNN, LSTM)
# ---------------------------------------------------------------------------


class LegacyCNN(nn.Module):
    """Legacy 2-layer convolutional network."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))
        return self.head(x)


class LegacyRNN(nn.Module):
    """Legacy vanilla RNN network."""

    def __init__(self, embed_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.rnn = nn.RNN(embed_dim, embed_dim, batch_first=True)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out.mean(dim=1))


class LegacyLSTM(nn.Module):
    """Legacy LSTM network."""

    def __init__(self, embed_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


# ---------------------------------------------------------------------------
# Builders & Registry
# ---------------------------------------------------------------------------


def build_torch_convnext(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[nn.Module, tuple[torch.Tensor]]:
    cfg = scale_config(CONVNEXT, scale)
    model = ConvNeXt(
        depths=cfg["depths"],
        dims=cfg["dims"],
        stem_kernel=cfg["stem"],
        stem_stride=cfg["stem"],
    ).eval()
    x = torch.randn(batch, 3, cfg["img_size"], cfg["img_size"])
    return model, (x,)


def build_torch_vit(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[nn.Module, tuple[torch.Tensor]]:
    cfg = scale_config(VIT, scale)
    model = ViT(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        depth=cfg["depth"],
        mlp_ratio=float(VIT_MLP_RATIO),
    ).eval()
    x = torch.randn(batch, 3, cfg["img_size"], cfg["img_size"])
    return model, (x,)


def build_torch_deep_dnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[nn.Module, tuple[torch.Tensor]]:
    cfg = scale_config(DEEP_DNN, scale)
    model = DeepDNN(**cfg).eval()
    x = torch.randn(batch, cfg["in_features"])
    return model, (x,)


def build_torch_transformer(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[nn.Module, tuple[torch.Tensor]]:
    cfg = scale_config(TRANSFORMER, scale)
    model = Transformer(
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ffn_dim=cfg["ffn_dim"],
    ).eval()
    x = torch.randn(batch, cfg["seq_len"], cfg["embed_dim"])
    return model, (x,)


def build_torch_cnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[nn.Module, tuple[torch.Tensor]]:
    size = 32 if scale == "micro" else 224
    model = LegacyCNN().eval()
    x = torch.randn(batch, 3, size, size)
    return model, (x,)


def build_torch_rnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[nn.Module, tuple[torch.Tensor]]:
    seq = 32 if scale == "micro" else 128
    dim = 128 if scale == "micro" else 256
    model = LegacyRNN(embed_dim=dim).eval()
    x = torch.randn(batch, seq, dim)
    return model, (x,)


def build_torch_lstm(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[nn.Module, tuple[torch.Tensor]]:
    seq = 32 if scale == "micro" else 128
    dim = 128 if scale == "micro" else 256
    model = LegacyLSTM(embed_dim=dim).eval()
    x = torch.randn(batch, seq, dim)
    return model, (x,)


TORCH_BUILDERS: dict[str, Any] = {
    "convnext": build_torch_convnext,
    "vit": build_torch_vit,
    "deep_dnn": build_torch_deep_dnn,
    "transformer": build_torch_transformer,
    "cnn": build_torch_cnn,
    "rnn": build_torch_rnn,
    "lstm": build_torch_lstm,
}
