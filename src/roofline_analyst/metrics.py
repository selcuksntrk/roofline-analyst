"""Roofline-analysis metric calculations."""

from __future__ import annotations


def compute_arithmetic_intensity(
        flops: int,
        logical_bytes_moved: int,
) -> float:
    """Return arithmetic intensity in FLOPs per logical byte moved.

    Args:
        flops: Known theoretical operation count for one invocation.
        logical_bytes_moved: Estimated logical read/write traffic.

    Returns:
        Arithmetic intensity in FLOPs per byte.

    Raises:
        ValueError: If FLOPs are negative or byte traffic is not positive.
    """
    if flops < 0:
        raise ValueError("flops must not be negative")

    if logical_bytes_moved <= 0:
        raise ValueError("logical_bytes_moved must be positive")

    return float(flops) / float(logical_bytes_moved)