"""Per-invocation roofline-point construction for supported modules."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from roofline_analyst.hardware import HardwareLimits
from roofline_analyst.hooks import ModuleExecution
from roofline_analyst.metrics import (
    RooflineRegime,
    classify_roofline_regime,
    compute_arithmetic_intensity,
    compute_ridge_point,
)
from roofline_analyst.traffic import estimate_module_traffic


@dataclass(frozen=True)
class LayerRooflinePoint:
    """One supported module invocation placed in roofline coordinates."""

    invocation_index: int
    module_name: str
    module_type: str
    flops: int
    logical_bytes_moved: int
    arithmetic_intensity: float
    achieved_gflops: float
    regime: RooflineRegime


def _linear_flops(
        module: nn.Linear,
        execution: ModuleExecution,
) -> int | None:
    if len(execution.inputs) != 1:
        return None

    input_tensor = execution.inputs[0]

    if not input_tensor.shape:
        return None

    if input_tensor.shape[-1] != module.in_features:
        return None

    positions = input_tensor.numel // module.in_features

    return 2 * positions * module.in_features * module.out_features


def build_layer_roofline_points(
        model: nn.Module,
        executions: tuple[ModuleExecution, ...],
        hardware_limits: HardwareLimits,
) -> tuple[LayerRooflinePoint, ...]:
    """Build roofline points for module invocations with known FLOP formulas."""
    modules_by_name = {
        name or "<root>": module
        for name, module in model.named_modules()
    }
    traffic_estimates = estimate_module_traffic(model, executions)
    ridge_point = compute_ridge_point(
        hardware_limits.cpu_neon_fp32_gflops,
        hardware_limits.memory_bandwidth_gbps,
    )

    points: list[LayerRooflinePoint] = []

    for execution, traffic in zip(
            executions,
            traffic_estimates,
            strict=True,
    ):
        module = modules_by_name[execution.module_name]

        if not isinstance(module, nn.Linear):
            continue

        flops = _linear_flops(module, execution)

        if flops is None:
            continue

        if execution.elapsed_ns <= 0:
            raise ValueError(
                f"module {execution.module_name!r} has non-positive elapsed time"
            )

        arithmetic_intensity = compute_arithmetic_intensity(
            flops,
            traffic.logical_bytes_moved,
        )
        achieved_gflops = float(flops) / float(execution.elapsed_ns)

        points.append(
            LayerRooflinePoint(
                invocation_index=traffic.invocation_index,
                module_name=execution.module_name,
                module_type=execution.module_type,
                flops=flops,
                logical_bytes_moved=traffic.logical_bytes_moved,
                arithmetic_intensity=arithmetic_intensity,
                achieved_gflops=achieved_gflops,
                regime=classify_roofline_regime(
                    arithmetic_intensity,
                    ridge_point,
                ),
            )
        )

    return tuple(points)