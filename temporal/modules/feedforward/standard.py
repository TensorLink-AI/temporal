# Modified temporal/modules/feedforward/standard.py
from torch import nn
from temporal.registry.core import register_module

# Dictionary mapping activation names (lowercase) to nn Modules
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,          # alias for SiLU
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu":  nn.ELU,
    # Add other activations as needed
}

@register_module("feedforward", "standard")
class StandardFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation="gelu", dropout=0.1, **kwargs):
        super().__init__()
        self.activation_fn = self._resolve_activation(activation) # Store the instantiated activation module
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) # Use separate dropout if different points need different behavior during eval, otherwise can reuse

    def _resolve_activation(self, name: str) -> nn.Module:
        """Resolves activation function name to an instantiated nn.Module."""
        name_lower = name.lower()
        if name_lower in _ACTIVATIONS:
            return _ACTIVATIONS[name_lower]() # Instantiate the module
        else:
            raise ValueError(f"Unsupported activation: {name}. Supported: {list(_ACTIVATIONS.keys())}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x) # Apply dropout after activation
        x = self.fc2(x)
        x = self.dropout2(x) # Apply dropout after second linear layer
        return x
