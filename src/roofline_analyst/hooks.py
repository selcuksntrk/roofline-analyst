"""Forward-hook utilities for capturing executed module tensor metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class TensorMetadata:
    """Shape and element-size metadata for one observed tensor."""

    shape: tuple[int, ...]
    dtype: str
    element_size_bytes: int

    @property
    def numel(self) -> int:
        """Return the tensor element count."""
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Return logical tensor storage bytes."""
        return self.numel * self.element_size_bytes


@dataclass(frozen=True)
class ModuleExecution:
    """Observed metadata and elapsed time for one leaf-module invocation."""

    module_name: str
    module_type: str
    inputs: tuple[TensorMetadata, ...]
    outputs: tuple[TensorMetadata, ...]
    elapsed_ns: int


def _collect_tensor_metadata(value: object) -> tuple[TensorMetadata, ...]:
    if isinstance(value, Tensor):
        return (
            TensorMetadata(
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                element_size_bytes=value.element_size(),
            ),
        )

    if isinstance(value, tuple | list):
        return tuple(
            metadata
            for item in value
            for metadata in _collect_tensor_metadata(item)
        )

    if isinstance(value, dict):
        return tuple(
            metadata
            for item in value.values()
            for metadata in _collect_tensor_metadata(item)
        )

    return ()


def _make_timing_hooks(
        module_name: str,
        module_type: str,
        executions: list[ModuleExecution],
) -> tuple[
    Callable[[nn.Module, tuple[object, ...]], None],
    Callable[[nn.Module, tuple[object, ...], object], None],
]:
    start_times_ns: list[int] = []

    def pre_hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
    ) -> None:
        start_times_ns.append(time.perf_counter_ns())

    def post_hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
    ) -> None:
        end_time_ns = time.perf_counter_ns()

        if not start_times_ns:
            raise RuntimeError(
                f"missing pre-hook timestamp for module {module_name!r}"
            )

        start_time_ns = start_times_ns.pop()

        executions.append(
            ModuleExecution(
                module_name=module_name,
                module_type=module_type,
                inputs=_collect_tensor_metadata(inputs),
                outputs=_collect_tensor_metadata(output),
                elapsed_ns=end_time_ns - start_time_ns,
            )
        )

    return pre_hook, post_hook


def capture_module_executions(
        model: nn.Module,
        example_inputs: tuple[Tensor, ...],
) -> tuple[ModuleExecution, ...]:
    """Capture leaf-module metadata and timing from one inference forward."""
    if not example_inputs:
        raise ValueError("example_inputs must contain at least one tensor")

    executions: list[ModuleExecution] = []
    handles: list[RemovableHandle] = []

    for name, module in model.named_modules():
        if any(module.children()):
            continue

        module_name = name or "<root>"
        pre_hook, post_hook = _make_timing_hooks(
            module_name,
            type(module).__name__,
            executions,
        )
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

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