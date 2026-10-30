"""Coverage reporting: modeled vs unmodeled parts of a traced function (issue #53)."""

import pytest

from neural_cost import (
    CostEstimate,
    Coverage,
    HardwareSpec,
    Measurement,
    analyze_gap,
)

HW = HardwareSpec("test", 1e12, 1e11)
EST = CostEstimate(10**9, 10**7, 10**7, 3)
RUN = Measurement(0.01, (0.01,))


def test_coverage_counts_and_summary():
    cov = Coverage.from_counts(6, 2, ["sort", "while", "sort"], unmodeled_output_bytes=64)
    assert (cov.total, cov.unmodeled) == (11, 3)
    assert cov.fraction == pytest.approx(8 / 11)
    assert cov.unmodeled_primitives == (("sort", 2), ("while", 1))
    assert cov.summary() == (
        "coverage: 72.7% of 11 equations modeled (unmodeled: sort x2, while x1)"
    )
    assert cov.summary(top=1).endswith("(unmodeled: sort x2, +1 more)")
    assert cov.warning(0.9).startswith("Low coverage")
    assert cov.warning(0.5) is None
    assert Coverage(0, 0, 0).fraction == 1.0
    with pytest.raises(ValueError):
        Coverage(-1, 0, 0)


def test_render_prints_coverage_and_warning_once():
    low = Coverage.from_counts(1, 0, ["sort"] * 9)
    gap = analyze_gap(EST, RUN, HW, coverage=low)
    text = gap.render()
    assert "coverage: 10.0% of 10 equations modeled" in text
    assert text.count("Low coverage") == 1 and "WARNING: Low coverage" in text
    assert any(f.startswith("Low coverage") for f in gap.findings)

    high = analyze_gap(EST, RUN, HW, coverage=Coverage(9, 1, 0))
    assert "coverage: 100.0%" in high.render() and "WARNING" not in high.render()
    assert "coverage" not in analyze_gap(EST, RUN, HW).render()
    relaxed = analyze_gap(EST, RUN, HW, coverage=low, coverage_threshold=0.05)
    assert "WARNING" not in relaxed.render()


def test_jax_adapter_reports_unmodeled_primitives():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from neural_cost.adapters import JaxAdapter

    def fn(x, w):
        y = jnp.tanh(x @ w).reshape(-1)  # dot + tanh modeled, reshape free
        return jnp.sort(y)  # sort is not modeled

    adapter = JaxAdapter()
    adapter.operations(fn, (jnp.ones((4, 8)), jnp.ones((8, 8))))
    cov = adapter.coverage
    assert cov.modeled == 2 and cov.free >= 1
    assert dict(cov.unmodeled_primitives) == {"sort": 1}
    assert cov.unmodeled_output_bytes == 32 * 4


def test_jax_profile_model_carries_coverage():
    jax = pytest.importorskip("jax")
    from neural_cost import analyze_model_gap, profile_model
    from neural_cost.adapters import JaxAdapter

    jnp = jax.numpy
    profile = profile_model(lambda x: jnp.sort(x) * 2, (jnp.ones(16),), JaxAdapter())
    assert profile.coverage is not None and profile.coverage.unmodeled == 1
    gap = analyze_model_gap(profile, RUN, HW).performance
    assert gap.coverage is profile.coverage
    assert "WARNING: Low coverage" in gap.render()


def test_torch_adapter_counts_leaf_modules():
    torch = pytest.importorskip("torch")
    from neural_cost.adapters import TorchAdapter

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.GELU(),
        torch.nn.Dropout(),
        torch.nn.LSTM(8, 8),  # tracked: its internal submodules are not units
        torch.nn.Linear(8, 2),  # defined but never called below
    )

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = model

        def forward(self, x):
            h = self.seq[2](self.seq[1](self.seq[0](x)))
            return self.seq[3](h)[0]

    adapter = TorchAdapter()
    adapter.operations(Wrapper(), (torch.randn(3, 2, 8),))
    cov = adapter.coverage
    assert (cov.modeled, cov.free, cov.unit) == (2, 1, "modules")
    assert cov.unmodeled_primitives == (("GELU", 1),)
