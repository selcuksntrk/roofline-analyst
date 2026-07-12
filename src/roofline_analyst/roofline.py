"""Roofline-geometry calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math

from roofline_analyst.hardware import HardwareLimits
from roofline_analyst.metrics import compute_ridge_point


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