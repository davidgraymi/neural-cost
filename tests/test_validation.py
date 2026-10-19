import math

import pytest

from neural_cost import (
    HardwareSpec,
    Operation,
    ValidationCase,
    fit_cost_model,
    validate_lower_bound,
    validate_predictor,
)
from neural_cost.validation import nnls, spearman

HW = HardwareSpec("test", peak_flops=1e12, memory_bandwidth=1e11)


def _synthetic(c0=20e-6, alpha=3e-6, pf=2e11, bw=4e10, noise=0.0):
    cases = []
    for i, (n, k) in enumerate(
        [
            (64, 2),
            (96, 9),
            (128, 4),
            (192, 1),
            (256, 3),
            (384, 7),
            (512, 8),
            (768, 2),
            (1024, 5),
            (2048, 12),
        ]
    ):
        flops, nbytes = 2 * n**3, 12 * n * n
        t = c0 + alpha * k + flops / pf + nbytes / bw
        t *= 1 + noise * (1 if i % 2 else -1)
        bound = max(flops / HW.peak_flops, nbytes / HW.memory_bandwidth)
        cases.append(ValidationCase(f"mm{n}", flops, nbytes, k, t, bound))
    return cases


def test_nnls_recovers_nonnegative_solution():
    rows = [[1, 0], [0, 1], [1, 1]]
    assert nnls(rows, [1, 2, 3]) == pytest.approx([1, 2], abs=1e-9)
    # Unconstrained solution has a negative coefficient; NNLS clamps it to 0.
    x = nnls([[1, 1], [1, 2], [1, 3]], [3, 2, 1])
    assert min(x) >= 0


def test_fit_recovers_generating_coefficients():
    model = fit_cost_model(_synthetic())
    assert model.call_overhead_seconds == pytest.approx(20e-6, rel=1e-4)
    assert model.dispatch_seconds == pytest.approx(3e-6, rel=1e-4)
    assert model.effective_flops == pytest.approx(2e11, rel=1e-4)
    assert model.effective_bandwidth == pytest.approx(4e10, rel=1e-3)
    cal = model.to_calibration(HW)
    assert cal.dispatch_seconds == pytest.approx(3e-6, rel=1e-4)


def test_loo_error_is_small_on_model_generated_data():
    report = validate_predictor(_synthetic(noise=0.02), hardware="test")
    assert report.bound_violations == 0
    assert report.loo_mape < 0.1
    assert report.loo_rank_correlation > 0.95
    assert report.roofline_only_mape > report.loo_mape
    assert "LOO MAPE" in report.to_markdown()
    assert '"loo_mape"' in report.to_json()


def test_lower_bound_violation_detected():
    cases = [ValidationCase("fast", 10, 10, 1, 1e-6, 2e-6), ValidationCase("ok", 1, 1, 1, 1, 0.5)]
    result = validate_lower_bound(cases)
    assert result["violations"] == 1
    assert result["tightness_max"] == pytest.approx(2.0)


def test_case_from_operations_serialized_vs_fused():
    ops = [
        Operation("fc", "linear", ((64, 64), (64, 64)), (64, 64)),
        Operation("act", "elementwise", ((64, 64),), (64, 64)),
    ]
    serial = ValidationCase.from_operations("a", ops, 1e-3, HW)
    fused = ValidationCase.from_operations("a", ops, 1e-3, HW, fused=True)
    assert fused.bytes < serial.bytes
    assert fused.lower_bound_seconds <= serial.lower_bound_seconds
    assert serial.kernels == 2
    with pytest.raises(ValueError):
        ValidationCase.from_operations("a", ops, 0, HW)


def test_spearman():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1], [1]))


def test_zero_shot_and_replay_predictions_are_scored():
    from neural_cost import HardwareCalibration
    from neural_cost.validation import replay_prediction

    cal = HardwareCalibration(1e11, 1e10, dispatch_seconds=1e-6, call_overhead_seconds=1e-5)
    cases = []
    for n in (32, 64, 128, 256):
        ops = [Operation("fc", "linear", ((n, n), (n, n)), (n, n))]
        measured = 2 * n**3 / 1e11 * 1.2 + 2e-5
        cases.append(
            ValidationCase.from_operations(
                f"fc{n}", ops, measured, HW, calibration=cal, isolated_seconds=[measured - 1e-5]
            )
        )
    report = validate_predictor(cases)
    assert report.zero_shot_mape is not None and report.replay_mape is not None
    assert report.replay_mape < 0.01  # replay plus call overhead reproduces the measurement
    assert "Op-replay" in report.to_markdown()

    costs = [Operation("a", "linear", ((4, 4), (4, 4)), (4, 4))]
    from neural_cost import estimate_operation

    est = [estimate_operation(op) for op in costs]
    # A missing replay falls back to the calibrated model; extra kernels add traffic.
    assert replay_prediction(est, [None], cal) > cal.call_overhead_seconds
    with pytest.raises(ValueError):
        replay_prediction(est, [], cal)


def test_cache_aware_bound_uses_resident_level():
    from neural_cost import CacheSpec

    hw = HardwareSpec("c", 1e12, 1e11, caches=(CacheSpec("L2", 1e12, 10**8),))
    ops = [Operation("act", "elementwise", ((256, 256),), (256, 256))]  # memory-bound
    warm = ValidationCase.from_operations("w", ops, 1e-3, hw)
    cold = ValidationCase.from_operations("w", ops, 1e-3, hw, cache_aware=False)
    assert warm.metadata["bound_memory_level"] == "L2"
    assert cold.metadata["bound_memory_level"] == "DRAM"
    assert warm.lower_bound_seconds < cold.lower_bound_seconds
