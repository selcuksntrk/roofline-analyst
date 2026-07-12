"""Roofline-geometry calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math

from roofline_analyst.hardware import HardwareLimits
from roofline_analyst.metrics import compute_ridge_point

import plotly.graph_objects as go

from roofline_analyst.layer_points import LayerRooflinePoint


@dataclass(frozen=True)
class RooflineSegment:
    """One two-point line segment in roofline coordinates."""

    arithmetic_intensities: tuple[float, float]
    performances_gflops: tuple[float, float]


@dataclass(frozen=True)
class RooflineGeometry:
    """Two-segment roofline derived from measured hardware limits."""

    ridge_point: float
    memory_segment: RooflineSegment
    compute_segment: RooflineSegment


def build_roofline_geometry(
        hardware_limits: HardwareLimits,
        x_min: float = 0.001,
        x_max: float = 1000.0,
) -> RooflineGeometry:
    """Construct memory-bound and compute-bound roofline segments.

    Args:
        hardware_limits: Measured CPU bandwidth and NEON FP32 throughput.
        x_min: Minimum positive arithmetic intensity to display.
        x_max: Maximum arithmetic intensity to display.

    Returns:
        Two roofline segments and their intersection ridge point.

    Raises:
        ValueError: If plot bounds are invalid or exclude the ridge point.
    """
    if not math.isfinite(x_min) or x_min <= 0.0:
        raise ValueError("x_min must be positive and finite")

    if not math.isfinite(x_max) or x_max <= x_min:
        raise ValueError("x_max must be finite and greater than x_min")

    ridge_point = compute_ridge_point(
        hardware_limits.cpu_neon_fp32_gflops,
        hardware_limits.memory_bandwidth_gbps,
    )

    if not x_min < ridge_point < x_max:
        raise ValueError("plot bounds must strictly contain the ridge point")

    memory_segment = RooflineSegment(
        arithmetic_intensities=(x_min, ridge_point),
        performances_gflops=(
            hardware_limits.memory_bandwidth_gbps * x_min,
            hardware_limits.cpu_neon_fp32_gflops,
        ),
    )

    compute_segment = RooflineSegment(
        arithmetic_intensities=(ridge_point, x_max),
        performances_gflops=(
            hardware_limits.cpu_neon_fp32_gflops,
            hardware_limits.cpu_neon_fp32_gflops,
        ),
    )

    return RooflineGeometry(
        ridge_point=ridge_point,
        memory_segment=memory_segment,
        compute_segment=compute_segment,
    )


def create_roofline_figure(
        geometry: RooflineGeometry,
        layer_points: tuple[LayerRooflinePoint, ...],
) -> go.Figure:
    """Create an interactive log-log roofline figure."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=geometry.memory_segment.arithmetic_intensities,
            y=geometry.memory_segment.performances_gflops,
            mode="lines",
            name="Memory roof",
            line={"color": "#1f77b4", "width": 3},
        )
    )

    figure.add_trace(
        go.Scatter(
            x=geometry.compute_segment.arithmetic_intensities,
            y=geometry.compute_segment.performances_gflops,
            mode="lines",
            name="Compute roof",
            line={"color": "#d62728", "width": 3},
        )
    )

    regime_colors = {
        "memory_bound": "#1f77b4",
        "at_ridge": "#ff7f0e",
        "compute_bound": "#d62728",
    }

    figure.add_trace(
        go.Scatter(
            x=[point.arithmetic_intensity for point in layer_points],
            y=[point.achieved_gflops for point in layer_points],
            mode="markers",
            name="Supported layer invocations",
            marker={
                "color": [
                    regime_colors[point.regime.value]
                    for point in layer_points
                ],
                "size": 10,
                "line": {"color": "#111111", "width": 1},
            },
            customdata=[
                [
                    point.module_name,
                    point.module_type,
                    point.flops,
                    point.logical_bytes_moved,
                    point.regime.value,
                ]
                for point in layer_points
            ],
            hovertemplate=(
                "module=%{customdata[0]}"
                "<br>type=%{customdata[1]}"
                "<br>AI=%{x:.4f} FLOPs/byte"
                "<br>achieved=%{y:.4f} GFLOPS"
                "<br>FLOPs=%{customdata[2]}"
                "<br>logical bytes=%{customdata[3]}"
                "<br>regime=%{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="CPU Roofline Analysis",
        xaxis={
            "title": "Arithmetic Intensity (FLOPs/byte)",
            "type": "log",
        },
        yaxis={
            "title": "Achieved Performance (GFLOPS)",
            "type": "log",
        },
        template="plotly_white",
        legend={"x": 0.02, "y": 0.98},
    )

    return figure