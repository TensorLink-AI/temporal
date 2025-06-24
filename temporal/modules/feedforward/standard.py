# Modified temporal/modules/feedforward/standard.py
import torch
from torch import nn
from temporal.registry.core import register_module
from typing import Optional, Tuple

# Dictionary mapping activation names (lowercase) to nn Modules
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,          # alias for SiLU
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu":  nn.ELU,
}

def _resolve_activation(name: str) -> nn.Module:
    """Resolves an activation function name to an instantiated nn.Module."""
    name_lower = name.lower()
    if name_lower in _ACTIVATIONS:
        return _ACTIVATIONS[name_lower]() # Instantiate the module
    else:
        raise ValueError(f"Unsupported activation: {name}. Supported: {list(_ACTIVATIONS.keys())}")

@register_module("feedforward", "standard")
class StandardFeedForward(nn.Module):
    """A standard two-layer feed-forward network (FFN).

    This module implements the position-wise feed-forward network found in
    standard Transformer architectures. It consists of two linear layers
    with a non-linear activation function in between.

    Attributes:
        fc1 (nn.Linear): The first linear layer (expansion).
        fc2 (nn.Linear): The second linear layer (contraction).
        activation_fn (nn.Module): The non-linear activation function.
        dropout (nn.Dropout): The dropout layer.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        **kwargs
    ):
        """Initializes the StandardFeedForward module.

        Args:
            hidden_size (int): The input and output dimension of the network (d_model).
            intermediate_size (int): The dimension of the hidden layer.
            activation (str): The name of the activation function to use.
            dropout (float): The dropout probability.
            bias (bool): Whether to include a bias term in the linear layers.
            **kwargs: Catches any other unused arguments.
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.activation_fn = _resolve_activation(activation)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Performs the forward pass of the FFN.

        Args:
            hidden_states (torch.Tensor): The input tensor of shape
                `[..., seq_len, hidden_size]`.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - The output tensor with the same shape as the input.
                - An auxiliary loss, which is `None` for this standard FFN.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        # Note: Dropout after the second linear layer is common in many implementations,
        # but is sometimes placed differently. We apply it before the final residual
        # connection in the main Transformer block.
        return hidden_states, None
