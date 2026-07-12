"""Built-in reproducible models for roofline analysis."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenMLP(nn.Module):
    """Token-wise MLP with input shape [batch, sequence, hidden]."""

    def __init__(
            self,
            hidden_size: int = 128,
            intermediate_size: int = 512,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(intermediate_size, hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply the token-wise MLP."""
        hidden_states = self.input_projection(tokens)
        activated_states = self.activation(hidden_states)
        return self.output_projection(activated_states)


def build_model(
        model_name: str,
        batch_size: int,
        sequence_length: int,
) -> tuple[nn.Module, tuple[Tensor, ...]]:
    """Build a named CPU model and its example inputs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if model_name != "token-mlp":
        raise ValueError(f"unsupported built-in model: {model_name}")

    hidden_size = 128
    model = TokenMLP(hidden_size=hidden_size)
    example_inputs = (
        torch.randn(batch_size, sequence_length, hidden_size),
    )

    return model, example_inputs