"""PyTorch operator-level FLOP profiling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile


@dataclass(frozen=True)
class OperatorFlopProfile:
    """FLOP estimate and CPU timing aggregated for one PyTorch operator."""

    operator_name: str
    flops: int
    call_count: int
    cpu_time_total_us: float


def profile_operator_flops(
        model: nn.Module,
        example_inputs: tuple[Tensor, ...],
) -> tuple[OperatorFlopProfile, ...]:
    """Profile supported CPU operator FLOP estimates for one inference forward.

    Args:
        model: PyTorch model to execute in evaluation mode.
        example_inputs: Positional tensor inputs for `model(*example_inputs)`.

    Returns:
        Operator statistics with strictly positive profiler-reported FLOPs.

    Raises:
        ValueError: If no inputs are supplied or an input is not on CPU.
    """
    if not example_inputs:
        raise ValueError("example_inputs must contain at least one tensor")

    if any(tensor.device.type != "cpu" for tensor in example_inputs):
        raise ValueError("this initial profiler supports CPU tensors only")

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            with profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    with_flops=True,
            ) as profiler:
                model(*example_inputs)
    finally:
        model.train(was_training)

    profiles: list[OperatorFlopProfile] = []

    for event in profiler.key_averages():
        flops = int(event.flops or 0)

        if flops <= 0:
            continue

        profiles.append(
            OperatorFlopProfile(
                operator_name=str(event.key),
                flops=flops,
                call_count=int(event.count),
                cpu_time_total_us=float(event.cpu_time_total),
            )
        )

    return tuple(profiles)