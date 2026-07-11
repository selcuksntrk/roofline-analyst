"""Forward-hook utilities for capturing executed module shapes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class ModuleExecution:
    """Observed tensor shapes for one leaf-module invocation."""

    module_name: str
    module_type: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]


def _collect_tensor_shapes(value: object) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, Tensor):
        return (tuple(value.shape),)

    if isinstance(value, tuple | list):
        return tuple(
            shape
            for item in value
            for shape in _collect_tensor_shapes(item)
        )

    if isinstance(value, dict):
        return tuple(
            shape
            for item in value.values()
            for shape in _collect_tensor_shapes(item)
        )

    return ()


def _make_forward_hook(
        module_name: str,
        module_type: str,
        executions: list[ModuleExecution],
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    def hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
    ) -> None:
        executions.append(
            ModuleExecution(
                module_name=module_name,
                module_type=module_type,
                input_shapes=_collect_tensor_shapes(inputs),
                output_shapes=_collect_tensor_shapes(output),
            )
        )

    return hook


def capture_module_executions(
        model: nn.Module,
        example_inputs: tuple[Tensor, ...],
) -> tuple[ModuleExecution, ...]:
    """Capture leaf-module tensor shapes from one inference forward pass.

    Args:
        model: PyTorch model to execute.
        example_inputs: Positional tensor inputs for `model(*example_inputs)`.

    Returns:
        Leaf-module invocation records in forward-execution order.

    Raises:
        ValueError: If no example inputs are supplied.
    """
    if not example_inputs:
        raise ValueError("example_inputs must contain at least one tensor")

    executions: list[ModuleExecution] = []
    handles: list[RemovableHandle] = []

    for name, module in model.named_modules():
        if any(module.children()):
            continue

        module_name = name or "<root>"
        hook = _make_forward_hook(
            module_name,
            type(module).__name__,
            executions,
        )
        handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            model(*example_inputs)
    finally:
        model.train(was_training)

        for handle in handles:
            handle.remove()

    return tuple(executions)