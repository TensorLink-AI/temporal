from torch import nn
from temporal.registry.core import register_module


@register_module("feedforward", "standard")
class StandardFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation="gelu", dropout=0.1, **kwargs):
        super().__init__()
        self.activation = self._resolve_activation(activation)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _resolve_activation(self, name):
        try:
            return getattr(nn, name.capitalize())()  # e.g., ReLU(), GELU(), SiLU(), etc.
        except AttributeError:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
