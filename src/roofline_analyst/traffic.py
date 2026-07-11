"""Logical per-module memory-traffic estimates."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from roofline_analyst.hooks import ModuleExecution


@dataclass(frozen=True)
class LayerTrafficEstimate:
    """Logical bytes-moved estimate for one module invocation."""

    invocation_index: int
    module_name: str
    module_type: str
    logical_bytes_moved: int
    estimation_method: str


def _tensor_nbytes(tensor: Tensor | None) -> int:
    if tensor is None:
        return 0

    return tensor.numel() * tensor.element_size()


def estimate_module_traffic(
        model: nn.Module,
        executions: tuple[ModuleExecution, ...],
) -> tuple[LayerTrafficEstimate, ...]:
    """Estimate logical read/write bytes for captured module invocations."""
    modules_by_name = {
        name or "<root>": module
        for name, module in model.named_modules()
    }

    estimates: list[LayerTrafficEstimate] = []

    for invocation_index, execution in enumerate(executions):
        module = modules_by_name[execution.module_name]

        input_bytes = sum(tensor.nbytes for tensor in execution.inputs)
        output_bytes = sum(tensor.nbytes for tensor in execution.outputs)

        if isinstance(module, nn.Linear):
            logical_bytes_moved = (
                    input_bytes
                    + _tensor_nbytes(module.weight)
                    + _tensor_nbytes(module.bias)
                    + output_bytes
            )
            estimation_method = "linear_input_weight_bias_output"
        elif isinstance(module, nn.LayerNorm):
            logical_bytes_moved = (
                    input_bytes
                    + _tensor_nbytes(module.weight)
                    + _tensor_nbytes(module.bias)
                    + output_bytes
            )
            estimation_method = "layernorm_input_affine_output"
        else:
            logical_bytes_moved = input_bytes + output_bytes
            estimation_method = "generic_input_output"

        estimates.append(
            LayerTrafficEstimate(
                invocation_index=invocation_index,
                module_name=execution.module_name,
                module_type=execution.module_type,
                logical_bytes_moved=logical_bytes_moved,
                estimation_method=estimation_method,
            )
        )

    return tuple(estimates)