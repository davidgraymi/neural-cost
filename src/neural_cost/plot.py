"""Publication-quality standalone vector SVG roofline visualizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from neural_cost.hardware import HardwareSpec


@dataclass(frozen=True, slots=True)
class RooflinePoint:
    """An operational workload point plotted on the roofline chart."""

    arithmetic_intensity: float  # FLOP/byte
    flops: float  # Achieved or bound FLOP/s
    label: str
    bottleneck: str = "memory"  # 'memory' or 'compute'
    latency_ms: float | None = None


def generate_roofline_svg(
    hardware: HardwareSpec,
    points: list[RooflinePoint] | RooflinePoint | None = None,
    *,
    title: str | None = None,
    width: int = 860,
    height: int = 540,
    theme: str = "dark",
) -> str:
    """Generate a clean, standalone publication-ready vector SVG roofline diagram.

    Parameters:
        hardware: HardwareSpec providing compute and memory bandwidth ceilings.
        points: Single RooflinePoint or list of operational points to overlay.
        title: Chart title (defaults to hardware device name).
        width: SVG image width in pixels.
        height: SVG image height in pixels.
        theme: 'dark' (slate) or 'light' (clean white).
    """
    pts: list[RooflinePoint] = []
    if isinstance(points, RooflinePoint):
        pts = [points]
    elif points:
        pts = list(points)

    chart_title = title or f"Roofline Model — {hardware.device_name}"

    # Colors
    if theme == "light":
        bg_color = "#ffffff"
        text_primary = "#0f172a"
        text_secondary = "#64748b"
        grid_color = "#e2e8f0"
        axis_color = "#94a3b8"
        peak_line = "#2563eb"  # blue
        bw_line = "#dc2626"  # red
        ridge_color = "#d97706"  # amber
        card_bg = "#f8fafc"
        card_border = "#cbd5e1"
    else:  # dark theme (default)
        bg_color = "#0b0f19"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        grid_color = "#1e293b"
        axis_color = "#334155"
        peak_line = "#38bdf8"  # cyan
        bw_line = "#f43f5e"  # rose
        ridge_color = "#fbbf24"  # amber
        card_bg = "#111827"
        card_border = "#1f2937"

    # Margins and plot area
    margin_left = 90
    margin_right = 50
    margin_top = 70
    margin_bottom = 70

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Log-log coordinate ranges
    # X range: Arithmetic intensity (FLOP/byte)
    x_min_val = 0.05
    x_max_val = max(1000.0, hardware.ridge_point * 10.0)
    for p in pts:
        if p.arithmetic_intensity > 0:
            x_min_val = min(x_min_val, p.arithmetic_intensity * 0.4)
            x_max_val = max(x_max_val, p.arithmetic_intensity * 2.5)

    log_x_min = math.floor(math.log10(x_min_val))
    log_x_max = math.ceil(math.log10(x_max_val))

    # Y range: Performance (FLOP/s)
    y_peak = hardware.peak_flops
    y_min_val = y_peak * 0.0005
    y_max_val = y_peak * 2.0
    for p in pts:
        if p.flops > 0:
            y_min_val = min(y_min_val, p.flops * 0.4)
            y_max_val = max(y_max_val, p.flops * 1.5)

    log_y_min = math.floor(math.log10(y_min_val))
    log_y_max = math.ceil(math.log10(y_max_val))

    def x_to_px(val: float) -> float:
        if val <= 0:
            return float(margin_left)
        lx = math.log10(val)
        ratio = (lx - log_x_min) / (log_x_max - log_x_min)
        return margin_left + ratio * plot_w

    def y_to_px(val: float) -> float:
        if val <= 0:
            return float(margin_top + plot_h)
        ly = math.log10(val)
        ratio = (ly - log_y_min) / (log_y_max - log_y_min)
        return margin_top + plot_h - (ratio * plot_h)

    # SVG Elements accumulator
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{bg_color}"/>',
    ]

    # Grid lines (Logarithmic decades)
    # X decades:
    for exp in range(int(log_x_min), int(log_x_max) + 1):
        v = 10.0**exp
        px = x_to_px(v)
        elements.append(
            f'<line x1="{px:.1f}" y1="{margin_top}" x2="{px:.1f}" y2="{margin_top + plot_h}" stroke="{grid_color}" stroke-width="1"/>'
        )
        lbl = f"10{exp}" if exp not in (0, 1, 2) else str(int(v))
        elements.append(
            f'<text x="{px:.1f}" y="{margin_top + plot_h + 20}" fill="{text_secondary}" font-size="11" font-family="system-ui, sans-serif" text-anchor="middle">{lbl}</text>'
        )

    # Y decades:
    for exp in range(int(log_y_min), int(log_y_max) + 1):
        v = 10.0**exp
        py = y_to_px(v)
        elements.append(
            f'<line x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_w}" y2="{py:.1f}" stroke="{grid_color}" stroke-width="1"/>'
        )
        if exp >= 12:
            lbl = f"{int(v / 1e12)} T"
        elif exp >= 9:
            lbl = f"{int(v / 1e9)} G"
        elif exp >= 6:
            lbl = f"{int(v / 1e6)} M"
        else:
            lbl = f"10{exp}"
        elements.append(
            f'<text x="{margin_left - 12}" y="{py + 4:.1f}" fill="{text_secondary}" font-size="11" font-family="system-ui, sans-serif" text-anchor="end">{lbl}</text>'
        )

    # Plot axes borders
    elements.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="{axis_color}" stroke-width="1.5"/>'
    )

    # Title & Axis Labels
    elements.append(
        f'<text x="{margin_left}" y="38" fill="{text_primary}" font-size="18" font-weight="700" font-family="system-ui, sans-serif">{chart_title}</text>'
    )
    elements.append(
        f'<text x="{margin_left + plot_w / 2}" y="{height - 18}" fill="{text_primary}" font-size="12" font-weight="600" font-family="system-ui, sans-serif" text-anchor="middle">Arithmetic Intensity (FLOP/byte)</text>'
    )
    elements.append(
        f'<text x="24" y="{margin_top + plot_h / 2}" fill="{text_primary}" font-size="12" font-weight="600" font-family="system-ui, sans-serif" text-anchor="middle" transform="rotate(-90, 24, {margin_top + plot_h / 2})">Performance (FLOP/s)</text>'
    )

    # --- Draw Hardware Roofline Boundary ---
    ridge_x = hardware.ridge_point
    peak_y = hardware.peak_flops
    bw = hardware.memory_bandwidth

    x_left = 10.0**log_x_min
    x_right = 10.0**log_x_max

    px_left = x_to_px(x_left)
    py_left = y_to_px(x_left * bw)

    px_ridge = x_to_px(ridge_x)
    py_ridge = y_to_px(peak_y)

    px_right = x_to_px(x_right)

    # Roofline path: Slanted bandwidth line -> Ridge point -> Flat compute peak
    roofline_d = f"M {px_left:.1f} {py_left:.1f} L {px_ridge:.1f} {py_ridge:.1f} L {px_right:.1f} {py_ridge:.1f}"
    elements.append(
        f'<path d="{roofline_d}" fill="none" stroke="{peak_line}" stroke-width="3" stroke-linecap="round"/>'
    )

    # Ridge Point Indicator line (vertical dashed)
    elements.append(
        f'<line x1="{px_ridge:.1f}" y1="{py_ridge:.1f}" x2="{px_ridge:.1f}" y2="{margin_top + plot_h}" stroke="{ridge_color}" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    # Ridge badge
    elements.append(f'<circle cx="{px_ridge:.1f}" cy="{py_ridge:.1f}" r="4" fill="{ridge_color}"/>')
    elements.append(
        f'<text x="{px_ridge:.1f}" y="{margin_top + plot_h - 10}" fill="{ridge_color}" font-size="10" font-weight="600" font-family="system-ui, sans-serif" text-anchor="middle">Ridge: {ridge_x:.1f} FLOP/B</text>'
    )

    # Ceilings Labels
    # Peak Compute label
    elements.append(
        f'<text x="{px_right - 10}" y="{py_ridge - 8:.1f}" fill="{peak_line}" font-size="11" font-weight="600" font-family="system-ui, sans-serif" text-anchor="end">Peak: {peak_y / 1e12:.1f} TFLOP/s</text>'
    )
    # Memory Bandwidth label
    angle_deg = -35  # approx slope on square decade aspect
    bw_label_x = px_left + (px_ridge - px_left) * 0.4
    bw_label_y = py_left + (py_ridge - py_left) * 0.4 - 10
    elements.append(
        f'<text x="{bw_label_x:.1f}" y="{bw_label_y:.1f}" fill="{bw_line}" font-size="11" font-weight="600" font-family="system-ui, sans-serif" transform="rotate({angle_deg}, {bw_label_x:.1f}, {bw_label_y:.1f})">Bandwidth: {bw / 1e9:.1f} GB/s</text>'
    )

    # --- Draw Operational Points ---
    for pt in pts:
        pt_x = x_to_px(pt.arithmetic_intensity)
        pt_y = y_to_px(pt.flops)
        is_mem = pt.bottleneck.lower() == "memory"
        dot_color = "#f43f5e" if is_mem else "#38bdf8"

        # Halo
        elements.append(
            f'<circle cx="{pt_x:.1f}" cy="{pt_y:.1f}" r="9" fill="{dot_color}" fill-opacity="0.25"/>'
        )
        # Center dot
        elements.append(
            f'<circle cx="{pt_x:.1f}" cy="{pt_y:.1f}" r="4.5" fill="{dot_color}" stroke="#ffffff" stroke-width="1.5"/>'
        )

        # Drop lines to axes
        elements.append(
            f'<line x1="{pt_x:.1f}" y1="{pt_y:.1f}" x2="{pt_x:.1f}" y2="{margin_top + plot_h}" stroke="{dot_color}" stroke-width="1" stroke-dasharray="2 3" opacity="0.6"/>'
        )
        elements.append(
            f'<line x1="{margin_left}" y1="{pt_y:.1f}" x2="{pt_x:.1f}" y2="{pt_y:.1f}" stroke="{dot_color}" stroke-width="1" stroke-dasharray="2 3" opacity="0.6"/>'
        )

        # Label card
        box_w = 170
        box_h = 44
        box_x = pt_x + 12
        box_y = pt_y - box_h / 2
        # Boundary bounds check
        if box_x + box_w > margin_left + plot_w:
            box_x = pt_x - box_w - 12
        if box_y < margin_top:
            box_y = margin_top + 4

        elements.append(
            f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_w}" height="{box_h}" rx="6" fill="{card_bg}" stroke="{card_border}" stroke-width="1" filter="drop-shadow(0 2px 4px rgba(0,0,0,0.4))"/>'
        )
        elements.append(
            f'<text x="{box_x + 8:.1f}" y="{box_y + 16:.1f}" fill="{text_primary}" font-size="11" font-weight="600" font-family="system-ui, sans-serif">{pt.label}</text>'
        )
        sub_text = f"{pt.arithmetic_intensity:.1f} FLOP/B • {pt.bottleneck.upper()}"
        if pt.latency_ms is not None:
            sub_text += f" • {pt.latency_ms:.1f}ms"
        elements.append(
            f'<text x="{box_x + 8:.1f}" y="{box_y + 32:.1f}" fill="{dot_color}" font-size="10" font-family="system-ui, sans-serif">{sub_text}</text>'
        )

    elements.append("</svg>")
    return "\n".join(elements)


def save_roofline_svg(
    filepath: str | Path,
    hardware: HardwareSpec,
    points: list[RooflinePoint] | RooflinePoint | None = None,
    *,
    title: str | None = None,
    width: int = 860,
    height: int = 540,
    theme: str = "dark",
) -> Path:
    """Generate and save an SVG roofline chart to the specified file path."""
    svg_content = generate_roofline_svg(
        hardware=hardware,
        points=points,
        title=title,
        width=width,
        height=height,
        theme=theme,
    )
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(svg_content)
    return p
