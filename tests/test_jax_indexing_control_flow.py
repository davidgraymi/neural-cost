"""JAX gather/scatter/dynamic-slice traffic and while/scan trip counts (issue #52).

Byte counts are computed by hand from the shapes: f32 = 4 bytes, i32 = 4 bytes.
"""

import pytest

from neural_cost import Operation, estimate_operation, estimate_operations

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax import lax

from neural_cost.adapters import JaxAdapter

TABLE = jnp.ones((1000, 8))
IDX = jnp.array([1, 2, 3], dtype=jnp.int32)


def only(ops, kind):
    found = [op for op in ops if op.kind == kind]
    assert len(found) == 1, found
    return found[0]


def test_gather_reads_only_touched_rows():
    op = only(JaxAdapter().operations(lambda t, i: t[i], (TABLE, IDX)), "gather")
    cost = estimate_operation(op)
    # 3 rows x 8 f32 read + 3 i32 indices; 3 x 8 f32 written; no FLOPs.
    assert cost.read_bytes == 3 * 8 * 4 + 3 * 4
    assert cost.write_bytes == 3 * 8 * 4
    assert cost.flops == 0


def test_scatter_add_is_in_place_and_combines():
    op = only(JaxAdapter().operations(lambda t, i: t.at[i].add(1.0), (TABLE, IDX)), "scatter")
    cost = estimate_operation(op)
    updates = 3 * 8 * 4
    assert cost.read_bytes == updates + 3 * 4 + updates  # updates, indices, old values
    assert cost.write_bytes == updates
    assert cost.flops == 3 * 8


def test_scatter_set_has_no_flops():
    def fn(t, i):
        return t.at[i].set(jnp.zeros((3, 8)))

    cost = estimate_operation(only(JaxAdapter().operations(fn, (TABLE, IDX)), "scatter"))
    assert (cost.flops, cost.read_bytes, cost.write_bytes) == (0, 3 * 8 * 4 + 3 * 4, 3 * 8 * 4)


def test_dynamic_slice_and_update_slice():
    start = jnp.int32(5)
    sliced = JaxAdapter().operations(
        lambda t, s: lax.dynamic_slice(t, (s, 0), (4, 8)), (TABLE, start)
    )
    cost = estimate_operation(only(sliced, "gather"))
    # 4 x 8 f32 read and written, plus two i32 start indices.
    assert (cost.read_bytes, cost.write_bytes) == (4 * 8 * 4 + 2 * 4, 4 * 8 * 4)

    def update(t, s):
        return lax.dynamic_update_slice(t, jnp.zeros((4, 8)), (s, 0))

    cost = estimate_operation(only(JaxAdapter().operations(update, (TABLE, start)), "scatter"))
    assert (cost.flops, cost.read_bytes, cost.write_bytes) == (0, 4 * 8 * 4 + 2 * 4, 4 * 8 * 4)


def test_counted_while_loop_trip_count_is_inferred():
    def fn(x, w):
        return lax.while_loop(lambda c: c[1] < 7, lambda c: (c[0] @ w, c[1] + 2), (x, 1))[0]

    adapter = JaxAdapter()
    ops = adapter.operations(fn, (jnp.ones((2, 4)), jnp.ones((4, 4))))
    assert adapter.while_trip_counts == [(3, "inferred")]  # i = 1, 3, 5
    assert only(ops, "matmul").attrs["repeat"] == 3
    assert not adapter.warnings


def data_dependent(x, w):
    def body(c):
        y = c[0] @ w
        return (c[1].at[0].set(y.sum()), y)[::-1]

    first = lax.while_loop(lambda c: c[0].sum() < 1e6, body, (x, jnp.zeros(4)))[0]
    return lax.while_loop(lambda c: c.sum() < 1e9, lambda c: c @ w, first)


def test_user_and_default_trip_counts():
    args = (jnp.ones((2, 4)), jnp.ones((4, 4)))
    adapter = JaxAdapter(trip_counts={1: 5}, default_trip_count=16)
    ops = adapter.operations(data_dependent, args)
    assert adapter.while_trip_counts == [(16, "default"), (5, "user")]
    repeats = sorted(op.attrs.get("repeat", 1) for op in ops if op.kind == "matmul")
    assert repeats == [5, 16]
    assert not adapter.warnings


def test_zero_trip_count_skips_body_and_validation():
    adapter = JaxAdapter(trip_counts={0: 0, 1: 0})
    ops = adapter.operations(data_dependent, (jnp.ones((2, 4)), jnp.ones((4, 4))))
    assert not [op for op in ops if op.kind == "matmul"]
    with pytest.raises(ValueError):
        JaxAdapter(trip_counts={0: -1})
    with pytest.raises(ValueError):
        JaxAdapter(default_trip_count=-1)


def test_scan_body_indexing_scaled_by_length():
    def fn(table, idx):
        def step(acc, i):
            return acc + lax.dynamic_slice(table, (i, 0), (1, 8)).sum(), None

        return lax.scan(step, 0.0, idx)[0]

    ops = JaxAdapter().operations(fn, (TABLE, jnp.arange(10, dtype=jnp.int32)))
    gathers = [op for op in ops if op.kind == "gather"]
    assert gathers and all(op.attrs["repeat"] == 10 for op in gathers)
    total = estimate_operations(gathers)
    assert total.write_bytes == 10 * 8 * 4


def test_portable_gather_scatter_validation():
    with pytest.raises(ValueError):
        estimate_operation(Operation("s", "scatter", ((4, 4), (2,)), (4, 4)))
    default = estimate_operation(Operation("g", "gather", ((100, 8), (5,)), (5, 8)))
    assert default.read_bytes == 5 * 8 * 4 + 5 * 4  # default 4-byte indices


def _fake_sdpa():
    """A stand-in for jax-mps's fused ``mps.sdpa`` primitive (q, k, v in B, H, T, D)."""
    from jax.extend.core import Primitive

    prim = Primitive("mps.sdpa")
    prim.def_abstract_eval(lambda q, k, v: q)
    prim.def_impl(lambda q, k, v: q)
    return prim


def test_backend_fused_attention_primitive_is_modeled():
    from neural_cost import flop_breakdown

    sdpa = _fake_sdpa()
    q = jnp.ones((2, 4, 8, 16))  # B, H, T, D
    adapter = JaxAdapter()
    ops = adapter.operations(lambda q, k, v: sdpa.bind(q, k, v), (q, q, q))
    assert adapter.coverage.unmodeled == 0
    (op,) = ops
    assert op.kind == "attention"
    # QKᵀ + AV: 2 × (2 × B·T × T × H·D), the same count torch's SDPA capture uses.
    assert flop_breakdown(ops).dense_flops == 4 * 2 * 8 * 8 * 4 * 16

    # Cross attention (different key length) gets explicit FLOPs.
    kv = jnp.ones((2, 4, 32, 16))
    (cross,) = JaxAdapter().operations(lambda q, k, v: sdpa.bind(q, k, v), (q, kv, kv))
    scores = 2 * 4 * 8 * 32
    assert estimate_operation(cross).flops == 4 * scores * 16 + 5 * scores
