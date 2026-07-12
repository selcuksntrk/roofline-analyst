from enum import StrEnum
import math


def compute_arithmetic_intensity(
        flops: int,
        logical_bytes_moved: int,
) -> float:
    """Return arithmetic intensity in FLOPs per logical byte moved."""
    if flops < 0:
        raise ValueError("flops must not be negative")

    if logical_bytes_moved <= 0:
        raise ValueError("logical_bytes_moved must be positive")

    return float(flops) / float(logical_bytes_moved)

class RooflineRegime(StrEnum):
    """Roofline region implied by arithmetic intensity."""

    MEMORY_BOUND = "memory_bound"
    AT_RIDGE = "at_ridge"
    COMPUTE_BOUND = "compute_bound"


def compute_ridge_point(
        peak_gflops: float,
        memory_bandwidth_gbps: float,
) -> float:
    """Return the roofline ridge point in FLOPs per byte."""
    if not math.isfinite(peak_gflops) or peak_gflops <= 0.0:
        raise ValueError("peak_gflops must be positive and finite")

    if (
            not math.isfinite(memory_bandwidth_gbps)
            or memory_bandwidth_gbps <= 0.0
    ):
        raise ValueError("memory_bandwidth_gbps must be positive and finite")

    return peak_gflops / memory_bandwidth_gbps


def classify_roofline_regime(
        arithmetic_intensity: float,
        ridge_point: float,
) -> RooflineRegime:
    """Classify arithmetic intensity relative to a ridge point."""
    if not math.isfinite(arithmetic_intensity) or arithmetic_intensity < 0.0:
        raise ValueError("arithmetic_intensity must be finite and non-negative")

    if not math.isfinite(ridge_point) or ridge_point <= 0.0:
        raise ValueError("ridge_point must be positive and finite")

    if math.isclose(arithmetic_intensity, ridge_point, rel_tol=1e-9):
        return RooflineRegime.AT_RIDGE

    if arithmetic_intensity < ridge_point:
        return RooflineRegime.MEMORY_BOUND

    return RooflineRegime.COMPUTE_BOUND