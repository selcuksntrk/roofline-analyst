"""Bridge to the standalone C++ hardware benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class HardwareLimits:
    """Measured single-machine hardware limits for roofline analysis."""

    memory_bandwidth_gbps: float
    cpu_neon_fp32_gflops: float


def _require_finite_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"hardware benchmark JSON field {key!r} must be numeric")

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"hardware benchmark JSON field {key!r} must be finite"
        )

    return numeric_value


def measure_hardware_limits(binary_path: Path) -> HardwareLimits:
    """Run the C++ benchmark and return its validated hardware limits.

    Args:
        binary_path: Path to the pre-built `roofline_hw` executable.

    Returns:
        Validated memory-bandwidth and CPU-NEON-throughput measurements.

    Raises:
        subprocess.CalledProcessError: If the binary exits unsuccessfully.
        json.JSONDecodeError: If stdout is not valid JSON.
        ValueError: If required JSON fields are missing or invalid.
    """
    completed_process = subprocess.run(
        [str(binary_path)],
        capture_output=True,
        check=True,
        text=True,
    )

    payload = json.loads(completed_process.stdout)

    if not isinstance(payload, dict):
        raise ValueError("hardware benchmark JSON root must be an object")

    return HardwareLimits(
        memory_bandwidth_gbps=_require_finite_number(
            payload,
            "memory_bandwidth_gbps",
        ),
        cpu_neon_fp32_gflops=_require_finite_number(
            payload,
            "cpu_neon_fp32_gflops",
        ),
    )