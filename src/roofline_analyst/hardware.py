"""Bridge to the standalone C++ hardware benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess


class HardwareBenchmarkError(RuntimeError):
    """Raised when the C++ hardware benchmark cannot produce valid limits."""


@dataclass(frozen=True)
class HardwareLimits:
    """Measured single-machine hardware limits for roofline analysis."""

    memory_bandwidth_gbps: float
    cpu_neon_fp32_gflops: float


def _require_finite_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"JSON field {key!r} must be numeric")

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(f"JSON field {key!r} must be finite")

    return numeric_value


def measure_hardware_limits(
        binary_path: Path,
        timeout_seconds: float = 30.0,
) -> HardwareLimits:
    """Run the C++ benchmark and return validated hardware limits.

    Args:
        binary_path: Path to the pre-built `roofline_hw` executable.
        timeout_seconds: Maximum permitted benchmark duration.

    Returns:
        Validated memory-bandwidth and CPU-NEON-throughput measurements.

    Raises:
        HardwareBenchmarkError: If process execution or JSON validation fails.
        ValueError: If `timeout_seconds` is not positive and finite.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive and finite")

    if not binary_path.is_file():
        raise HardwareBenchmarkError(
            f"hardware binary not found: {binary_path}. "
            "Build it with `cmake --build build-homebrew-release`."
        )

    if not os.access(binary_path, os.X_OK):
        raise HardwareBenchmarkError(
            f"hardware binary is not executable: {binary_path}"
        )

    try:
        completed_process = subprocess.run(
            [str(binary_path)],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as error:
        diagnostic = error.stderr.strip() or "no diagnostic on stderr"
        raise HardwareBenchmarkError(
            f"hardware benchmark exited with status {error.returncode}: "
            f"{diagnostic}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise HardwareBenchmarkError(
            f"hardware benchmark timed out after {timeout_seconds} seconds"
        ) from error
    except OSError as error:
        raise HardwareBenchmarkError(
            f"could not start hardware benchmark: {error}"
        ) from error

    try:
        payload = json.loads(completed_process.stdout)
    except json.JSONDecodeError as error:
        raise HardwareBenchmarkError(
            f"hardware benchmark emitted invalid JSON: {error.msg}"
        ) from error

    if not isinstance(payload, dict):
        raise HardwareBenchmarkError(
            "hardware benchmark JSON root must be an object"
        )

    try:
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
    except ValueError as error:
        raise HardwareBenchmarkError(
            f"hardware benchmark JSON schema is invalid: {error}"
        ) from error