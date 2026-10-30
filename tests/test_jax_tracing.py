"""JAX tracer: higher-order primitives, general contractions, parameter tracking."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax import lax

from neural_cost import flop_breakdown
from neural_cost.adapters import JaxAdapter


def test_scan_body_scaled_by_trip_count():
    def model(xs, w):
        def step(h, x):
            h = jnp.tanh(x @ w + h)
            return h, h

        return lax.scan(step, jnp.zeros((2, 4)), xs)[1]

    ops = JaxAdapter().operations(model, (jnp.ones((7, 2, 3)), jnp.ones((3, 4))))
    dots = [op for op in ops if op.kind == "matmul"]
    assert len(dots) == 1 and dots[0].attrs["repeat"] == 7
    assert flop_breakdown(ops).dense_flops == 7 * 2 * 2 * 3 * 4
    assert flop_breakdown(ops).parameter_count == 12


def test_nested_jit_and_closed_over_constants_are_counted():
    w = jnp.ones((8, 8))

    @jax.jit
    def inner(x):
        return x @ w  # closed-over constant

    ops = JaxAdapter().operations(lambda x: inner(x) @ w.T, (jnp.ones((2, 8)),))
    breakdown = flop_breakdown(ops)
    assert breakdown.dense_flops == 2 * (2 * 2 * 8 * 8)
    assert breakdown.parameter_count == 64  # one tied weight, counted once


def test_batched_dot_general_flops():
    q = jnp.ones((2, 3, 5, 4))  # (B, H, T, D)
    ops = JaxAdapter().operations(lambda a, b: jnp.einsum("bhtd,bhsd->bhts", a, b), (q, q))
    assert flop_breakdown(ops).dense_flops == 2 * 2 * 3 * 5 * 5 * 4


def test_cond_takes_most_expensive_branch_and_while_warns():
    def model(x, w):
        y = lax.cond(x.sum() > 0, lambda v: v @ w @ w, lambda v: v, x)
        # Data-dependent trip count: the body is counted once, with a warning.
        return lax.while_loop(lambda c: c[0].sum() < 1e6, lambda c: (c[0] @ w,), (y,))[0]

    adapter = JaxAdapter()
    ops = adapter.operations(model, (jnp.ones((2, 4)), jnp.ones((4, 4))))
    assert flop_breakdown(ops).dense_flops == 3 * (2 * 2 * 4 * 4)  # 2 in cond + 1 while body
    assert any("while_loop #0" in warning for warning in adapter.warnings)
    assert adapter.while_trip_counts == [(1, "assumed")]


def test_parameter_argnums_selects_weights():
    def model(x, w):
        return x @ w

    args = (jnp.ones((2, 3)), jnp.ones((3, 4)))
    assert flop_breakdown(JaxAdapter().operations(model, args)).parameter_count == 12
    none = JaxAdapter(parameter_argnums=()).operations(model, args)
    assert flop_breakdown(none).parameter_count == 0
