"""Unit tests for standalone vector SVG roofline visualizer."""

from __future__ import annotations

import pytest

from neural_cost.hardware import HardwareSpec
from neural_cost.plot import (
    RooflinePoint,
    generate_roofline_svg,
    save_roofline_svg,
)


@pytest.fixture
def a100_spec() -> HardwareSpec:
    return HardwareSpec(
        name="NVIDIA A100-SXM4-80GB",
        peak_flops=312e12,
        memory_bandwidth=2039e9,
    )


class TestRooflineSVG:
    def test_generate_svg_basic_dark(self, a100_spec):
        svg = generate_roofline_svg(a100_spec)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "NVIDIA A100-SXM4-80GB" in svg
        assert "Arithmetic Intensity" in svg
        assert "Performance (FLOP/s)" in svg
        assert "Ridge: 153.0 FLOP/B" in svg
        assert "Peak: 312.0 TFLOP/s" in svg
        assert "Bandwidth: 2039.0 GB/s" in svg
        # Dark theme background
        assert 'fill="#0b0f19"' in svg

    def test_generate_svg_light_theme(self, a100_spec):
        svg = generate_roofline_svg(a100_spec, theme="light", title="Custom Roofline")
        assert "Custom Roofline" in svg
        # Light theme background
        assert 'fill="#ffffff"' in svg

    def test_generate_svg_with_points(self, a100_spec):
        pts = [
            RooflinePoint(
                arithmetic_intensity=12.5,
                flops=12.5 * a100_spec.memory_bandwidth,
                label="LLaMA-3-8B Decode",
                bottleneck="memory",
                latency_ms=1.2,
            ),
            RooflinePoint(
                arithmetic_intensity=250.0,
                flops=a100_spec.peak_flops,
                label="LLaMA-3-8B Prefill",
                bottleneck="compute",
                latency_ms=15.4,
            ),
        ]
        svg = generate_roofline_svg(a100_spec, points=pts)
        assert "LLaMA-3-8B Decode" in svg
        assert "LLaMA-3-8B Prefill" in svg
        assert "MEMORY" in svg
        assert "COMPUTE" in svg
        assert "1.2ms" in svg
        assert "15.4ms" in svg

    def test_generate_svg_with_single_point(self, a100_spec):
        pt = RooflinePoint(
            arithmetic_intensity=50.0,
            flops=50.0 * a100_spec.memory_bandwidth,
            label="Single Point Workload",
        )
        svg = generate_roofline_svg(a100_spec, points=pt)
        assert "Single Point Workload" in svg

    def test_save_roofline_svg(self, tmp_path, a100_spec):
        out_file = tmp_path / "diagrams" / "roofline.svg"
        saved = save_roofline_svg(out_file, a100_spec)
        assert saved == out_file
        assert out_file.is_file()
        content = out_file.read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert "NVIDIA A100-SXM4-80GB" in content
