"""Modern deep learning operators: Fused SDPA Attention, RMSNorm, and SwiGLU."""

from __future__ import annotations

import importlib.util
from typing import Any

# ---------------------------------------------------------------------------
# PyTorch Operators
# ---------------------------------------------------------------------------
if importlib.util.find_spec("torch") is not None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight

    class SwiGLU(nn.Module):
        """Swish Gated Linear Unit feed-forward block (Shazeer, 2020)."""

        def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            bias: bool = False,
        ) -> None:
            super().__init__()
            hidden_features = hidden_features or int(in_features * 8 / 3)
            out_features = out_features or in_features
            self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)
            self.w_up = nn.Linear(in_features, hidden_features, bias=bias)
            self.w_down = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

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
# JAX Operators
# ---------------------------------------------------------------------------
if importlib.util.find_spec("jax") is not None:
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp

    def jax_rmsnorm(x: Any, scale: Any, eps: float = 1e-6) -> Any:
        """JAX RMSNorm."""
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
        return (x / rms) * scale

    def jax_swiglu(x: Any, w_gate: Any, w_up: Any, w_down: Any) -> Any:
        """JAX SwiGLU MLP block."""
        return (jnn.silu(x @ w_gate) * (x @ w_up)) @ w_down

    def jax_attention(
        x: Any, wq: Any, wk: Any, wv: Any, wo: Any, num_heads: int = 4
    ) -> Any:
        """JAX Multi-Head Attention using jax.nn.dot_product_attention."""
        b, l, d = x.shape
        head_dim = d // num_heads
        q = (x @ wq).reshape((b, l, num_heads, head_dim))
        k = (x @ wk).reshape((b, l, num_heads, head_dim))
        v = (x @ wv).reshape((b, l, num_heads, head_dim))
        out = jax.nn.dot_product_attention(q, k, v)
        return out.reshape((b, l, d)) @ wo
