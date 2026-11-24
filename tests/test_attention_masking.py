"""Causal, sliding-window and cross-attention FLOP accounting."""

from __future__ import annotations

import itertools

import pytest

from neural_cost import Operation, attention_visible_pairs, estimate_operation, flop_breakdown
from neural_cost.estimate import attention_flop_parts


def _brute_pairs(q_len: int, kv_len: int, offset: int, window: int | None) -> int:
    return sum(
        1
        for i in range(q_len)
        for j in range(kv_len)
        if j <= offset + i and (window is None or j > offset + i - window)
    )


def test_visible_pairs_matches_brute_force():
    for q_len, kv_len, offset, window in itertools.product(
        range(7), range(7), range(-3, 9), (None, 1, 2, 3, 9)
    ):
        got = attention_visible_pairs(q_len, kv_len, causal=True, window=window, q_offset=offset)
        assert got == _brute_pairs(q_len, kv_len, offset, window)


def test_visible_pairs_defaults():
    assert attention_visible_pairs(8, 16) == 8 * 16
    assert attention_visible_pairs(4, 4, causal=True) == 4 * 5 // 2
    # Default alignment is bottom-right: a decode step sees the whole cache.
    assert attention_visible_pairs(1, 1024, causal=True) == 1024
    # PyTorch's top-left alignment: the single query sees only key 0.
    assert attention_visible_pairs(1, 1024, causal=True, q_offset=0) == 1
    # Sliding window caps each query's keys.
    assert attention_visible_pairs(1024, 1024, causal=True, window=128) == (
        128 * 1024 - 128 * 127 // 2
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"window": 4}, {"window": 0, "causal": True}, {"window": -1, "causal": True}],
)
def test_visible_pairs_rejects_bad_window(kwargs):
    with pytest.raises(ValueError):
        attention_visible_pairs(4, 4, **kwargs)


def _sdpa_op(q_len: int, kv_len: int, **attrs) -> Operation:
    b, h, d = 2, 4, 16
    return Operation(
        "sdpa",
        "attention",
        ((b, h, q_len, d), (b, h, kv_len, d), (b, h, kv_len, d)),
        (b, h, q_len, d),
        2,
        {
            "num_heads": h,
            "seq_len": q_len,
            "kv_len": kv_len,
            "head_dim": d,
            "include_projections": False,
            **attrs,
        },
    )


def test_causal_attention_counts_lower_triangle():
    t = 512
    dense = estimate_operation(_sdpa_op(t, t)).flops
    causal = estimate_operation(_sdpa_op(t, t, causal=True)).flops
    assert causal == dense * (t + 1) // (2 * t)
    _, scores, softmax = attention_flop_parts(_sdpa_op(t, t, causal=True))
    pairs = 2 * 4 * t * (t + 1) // 2
    assert scores == 4 * pairs * 16
    assert softmax == 5 * pairs


def test_decode_step_scales_with_kv_length():
    """One query against a 1024-token cache scores 1024 keys, not one."""
    _, scores, _ = attention_flop_parts(_sdpa_op(1, 1024))
    assert scores == 4 * 2 * 4 * 1024 * 16
    # Causal decode with default bottom-right alignment sees the full cache too.
    assert attention_flop_parts(_sdpa_op(1, 1024, causal=True))[1] == scores


def test_cross_attention_projections_use_kv_tokens():
    b, q_len, kv_len, d = 2, 8, 32, 64
    op = Operation(
        "xattn",
        "attention",
        ((b, q_len, d),),
        (b, q_len, d),
        4,
        {"num_heads": 4, "seq_len": q_len, "kv_len": kv_len},
    )
    proj, scores, _ = attention_flop_parts(op)
    assert proj == 2 * 2 * (b * q_len + b * kv_len) * d * d
    assert scores == 4 * b * q_len * kv_len * d


def test_self_attention_unchanged_without_new_attrs():
    """Existing operations without kv_len/causal keep the dense formula."""
    b, t, d, h = 2, 16, 64, 4
    op = Operation("mha", "attention", ((b, t, d),), (b, t, d), 4, {"num_heads": h})
    proj, scores, softmax = attention_flop_parts(op)
    assert proj == 4 * 2 * b * t * d * d
    assert scores == 4 * b * t * h * t * (d // h)
    assert softmax == 5 * b * t * h * t


def test_torch_sdpa_captures_causal_and_kv_len():
    torch = pytest.importorskip("torch")
    from neural_cost.adapters import TorchAdapter

    class Block(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    q = torch.randn(1, 2, 8, 16)
    (op,) = TorchAdapter(use_profiler=False).operations(Block(), (q, q, q))
    assert op.attrs["causal"] is True and op.attrs["q_offset"] == 0
    assert flop_breakdown([op]).dense_flops == 4 * 2 * (8 * 9 // 2) * 16

    class Decode(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    kv = torch.randn(1, 2, 64, 16)
    (dec,) = TorchAdapter(use_profiler=False).operations(Decode(), (q[:, :, :1], kv, kv))
    assert dec.attrs["kv_len"] == 64
    assert flop_breakdown([dec]).dense_flops == 4 * 2 * 64 * 16


def test_torch_fx_sdpa_captures_causal():
    torch = pytest.importorskip("torch")
    from neural_cost.adapters import TorchFxAdapter

    class Block(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    q = torch.randn(1, 2, 8, 16)
    ops = [op for op in TorchFxAdapter().operations(Block(), (q, q, q)) if op.kind == "attention"]
    assert len(ops) == 1 and ops[0].attrs.get("causal") is True


def test_torch_mha_causal_and_cross_attention():
    torch = pytest.importorskip("torch")
    from neural_cost.adapters import TorchAdapter

    t, d = 8, 32
    mha = torch.nn.MultiheadAttention(d, 4, batch_first=True)

    class Causal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = mha

        def forward(self, x):
            mask = torch.nn.Transformer.generate_square_subsequent_mask(t)
            return self.mha(x, x, x, attn_mask=mask, is_causal=True)[0]

    x = torch.randn(2, t, d)
    (op,) = TorchAdapter(use_profiler=False).operations(Causal(), (x,))
    assert op.attrs["causal"] is True
    _, scores, _ = attention_flop_parts(op)
    assert scores == 4 * 2 * (t * (t + 1) // 2) * d

    class Cross(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = mha

        def forward(self, x, memory):
            return self.mha(x, memory, memory)[0]

    memory = torch.randn(2, 20, d)
    (cross,) = TorchAdapter(use_profiler=False).operations(Cross(), (x, memory))
    assert cross.attrs["kv_len"] == 20
    assert attention_flop_parts(cross)[1] == 4 * 2 * t * 20 * d


def test_torch_replay_runs_masked_attention():
    pytest.importorskip("torch")
    from neural_cost.adapters._torch_replay import replay_operations

    ops = [
        _sdpa_op(16, 16, causal=True),
        _sdpa_op(4, 16, causal=True),  # bottom-right alignment needs an explicit mask
        _sdpa_op(16, 16, causal=True, window=4),
        Operation(
            "xattn",
            "attention",
            ((2, 8, 32),),
            (2, 8, 32),
            4,
            {"num_heads": 4, "seq_len": 8, "kv_len": 20, "causal": True},
        ),
    ]
    times = replay_operations(ops, target_seconds=1e-4)
    assert all(seconds is not None and seconds > 0 for seconds in times)


def test_jax_fused_attention_reads_causal_param():
    jax = pytest.importorskip("jax")
    from jax.extend.core import Primitive

    from neural_cost.adapters import JaxAdapter

    prim = Primitive("mps.sdpa")
    prim.def_abstract_eval(lambda q, k, v, **_: q)
    prim.def_impl(lambda q, k, v, **_: q)
    q = jax.numpy.ones((1, 2, 8, 16))
    (op,) = JaxAdapter().operations(lambda q, k, v: prim.bind(q, k, v, is_causal=True), (q, q, q))
    assert op.kind == "attention" and op.attrs["causal"] is True
    assert flop_breakdown([op]).dense_flops == 4 * 2 * (8 * 9 // 2) * 16


def test_keras_mha_use_causal_mask():
    tf = pytest.importorskip("tensorflow")
    from neural_cost.adapters import TensorFlowAdapter

    t, d = 8, 32
    x_in = tf.keras.Input((t, d))
    out = tf.keras.layers.MultiHeadAttention(4, d // 4)(x_in, x_in, use_causal_mask=True)
    model = tf.keras.Model(x_in, out)
    ops = [
        op
        for op in TensorFlowAdapter().operations(model, (tf.ones((2, t, d)),))
        if op.kind == "attention"
    ]
    assert len(ops) == 1 and ops[0].attrs.get("causal") is True
