import pytest

from neural_cost import (
    HardwareCalibration,
    HardwareSpec,
    Measurement,
    Operation,
    attribute_gap,
)

HW = HardwareSpec("test", peak_flops=1e12, memory_bandwidth=1e11)


def M(seconds, samples=None):
    return Measurement(seconds, tuple(samples) if samples else (seconds,))


def _ops():
    # A compute-heavy matmul followed by a fusible memory-bound activation.
    return [
        Operation("fc", "linear", ((256, 1024), (1024, 1024)), (256, 1024)),
        Operation("act", "elementwise", ((256, 1024),), (256, 1024)),
    ]


def test_components_sum_to_measured_and_are_ordered():
    ops = _ops()
    cal = HardwareCalibration(5e11, 5e10, dispatch_seconds=5e-6, source="test")
    measured = Measurement(2e-3, (1.9e-3, 2e-3, 2.1e-3))
    attr = attribute_gap(ops, measured, HW, cal)
    names = [c.name for c in attr.components]
    assert names == [
        "roofline_floor",
        "unfused_traffic",
        "serialization",
        "peak_derate",
        "dispatch",
        "timing_jitter",
        "residual",
    ]
    assert sum(c.seconds for c in attr.components) == pytest.approx(2e-3)
    assert sum(c.fraction for c in attr.components) == pytest.approx(1.0)
    assert all(c.seconds >= 0 for c in attr.components)
    assert attr.component("dispatch").seconds == pytest.approx(2 * 5e-6)
    assert attr.component("timing_jitter").seconds == pytest.approx(1e-4)
    # The floor is the aggregate fused roofline.
    flops = 2 * 256 * 1024 * 1024 + 256 * 1024
    assert attr.component("roofline_floor").seconds >= flops / HW.peak_flops
    # Compute-bound: eliminating traffic does not move the aggregate floor.
    assert attr.component("unfused_traffic").seconds == 0
    assert not attr.overexplained and not attr.bound_violated
    assert "dispatch" in attr.render()


def test_unfused_traffic_charged_when_memory_bound():
    ops = [
        Operation("fc", "linear", ((4096, 64), (64, 64)), (4096, 64)),
        Operation("act", "elementwise", ((4096, 64),), (4096, 64)),
    ]
    attr = attribute_gap(ops, M(1e-3), HW)
    assert attr.component("unfused_traffic").seconds > 0
    eager = attribute_gap(ops, M(1e-3), HW, assume_fusion=False)
    assert eager.component("unfused_traffic").seconds == 0


def test_peak_derate_matches_calibration_ratio():
    op = [Operation("fc", "linear", ((512, 4096), (4096, 4096)), (512, 4096))]
    cal = HardwareCalibration(5e11, 1e11)  # half of peak compute
    attr = attribute_gap(op, M(1.0), HW, cal)
    floor = attr.component("roofline_floor").seconds
    assert attr.component("peak_derate").seconds == pytest.approx(floor, rel=1e-6)


def test_dispatch_dominated_workload_gets_dispatch_recommendation():
    ops = [Operation(f"e{i}", "elementwise", ((8,),), (8,)) for i in range(50)]
    cal = HardwareCalibration(1e12, 1e11, dispatch_seconds=10e-6)
    attr = attribute_gap(ops, M(6e-4), HW, cal)
    assert attr.dominant.name == "dispatch"
    assert attr.recommendations()[0].startswith("dispatch")
    assert attr.component("dispatch").speedup_if_removed > 5


def test_bound_violation_and_overexplained_are_flagged():
    op = [Operation("fc", "linear", ((1024, 4096), (4096, 4096)), (1024, 4096))]
    attr = attribute_gap(op, M(1e-6), HW)
    assert attr.bound_violated
    assert attr.component("residual").seconds < 0
    assert "below the roofline floor" in attr.render()


def test_without_calibration_warns_and_folds_into_residual():
    attr = attribute_gap(_ops(), M(1e-2), HW)
    assert attr.component("dispatch").seconds == 0
    assert attr.component("peak_derate").seconds == 0
    assert attr.warnings


def test_repeat_scales_kernel_count():
    op = [Operation("step", "matmul", ((4, 8), (8, 8)), (4, 8), attrs={"repeat": 16})]
    attr = attribute_gap(op, M(1e-3), HW, HardwareCalibration(1e12, 1e11, 1e-6))
    assert attr.kernel_count == 16
    assert attribute_gap(op, M(1e-3), HW, kernel_count=1).kernel_count == 1


def test_calibration_validation():
    with pytest.raises(ValueError):
        HardwareCalibration(0, 1)
    with pytest.raises(ValueError):
        HardwareCalibration(1, 1, dispatch_seconds=-1)


def test_calibration_curves_interpolate_log_log():
    cal = HardwareCalibration(
        1e12,
        1e11,
        flops_curve=((1e6, 1e10), (1e9, 1e12)),
        bandwidth_curve=((1e3, 1e12), (1e9, 1e11)),
    )
    assert cal.flops_rate(1e6) == pytest.approx(1e10)
    assert cal.flops_rate(1e3) == pytest.approx(1e10)  # clamped below range
    assert cal.flops_rate(1e12) == pytest.approx(1e12)  # clamped above range
    assert cal.flops_rate(10**7.5) == pytest.approx(1e11)  # geometric midpoint
    assert cal.kernel_seconds(1e9, 0) == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        HardwareCalibration(1, 1, flops_curve=((0, 1),))


def test_async_dispatch_only_exposes_launch_gap():
    from neural_cost import CostEstimate

    tiny = CostEstimate(10, 10, 0, 1)  # ~0 execution
    big = CostEstimate(10**12, 10, 0, 1)  # 1 s at 1e12 FLOP/s
    sync = HardwareCalibration(1e12, 1e11, dispatch_seconds=1e-5)
    asyn = HardwareCalibration(1e12, 1e11, dispatch_seconds=1e-5, async_dispatch=True)
    assert sync.predict([tiny, big])[1] == pytest.approx(2e-5)
    # Async: only the tiny kernel exposes (almost) its whole launch latency.
    assert asyn.predict([tiny, big])[1] == pytest.approx(1e-5, rel=1e-3)


def test_extra_kernels_modeled_by_traffic_and_dispatch_level():
    cal = HardwareCalibration(1e12, 1e9, dispatch_seconds=1e-6)
    execution, exposed = cal.predict([], extra_kernels=[(10**6, True), (10**6, False)])
    assert execution == pytest.approx(2e-3)  # 2 × 1 MB at 1 GB/s
    assert exposed == pytest.approx(1e-6)  # only the Python-issued kernel pays dispatch


def test_isolated_seconds_add_kernel_inefficiency_component():
    ops = _ops()
    cal = HardwareCalibration(1e12, 1e11, dispatch_seconds=1e-6)
    base = attribute_gap(ops, M(5e-3), HW, cal)
    with_replay = attribute_gap(ops, M(5e-3), HW, cal, isolated_seconds=[3e-3, 5e-4])
    names = [c.name for c in with_replay.components]
    assert "kernel_inefficiency" in names and "kernel_inefficiency" not in [
        c.name for c in base.components
    ]
    assert with_replay.component("kernel_inefficiency").seconds > 0
    assert with_replay.component("residual").seconds < base.component("residual").seconds
    assert "Composition effects" in with_replay.component("residual").explanation
    assert sum(c.seconds for c in with_replay.components) == pytest.approx(5e-3)
    with pytest.raises(ValueError):
        attribute_gap(ops, M(5e-3), HW, isolated_seconds=[1e-3, 1e-3])  # needs calibration
