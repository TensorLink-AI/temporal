from typing import Dict, Optional
import torch.nn as nn
import torch

class BaseOutputHead(nn.Module):
    """
    Abstract base class for all output heads.
    An output head takes the final hidden state from the model's backbone
    and produces the final output, which can be predictions, distribution parameters, etc.
    It can also be responsible for calculating the loss if the loss is tightly coupled
    with the output generation logic.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Processes the backbone's output.
        Args:
            hidden_states (torch.Tensor): The final hidden states from the model's backbone.
            **kwargs: Additional arguments, which may include `labels`, `loss_mask`, etc.
        Returns:
            A dictionary containing at least 'preds' (the model's predictions) and
            optionally 'loss' if the head calculates it directly.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def sample(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generates samples or predictions from the model's output.
        This is used during inference/generation.
        """
        # A simple default for heads that don't have a special sampling method.
        # More complex heads (like TimeFlow) will override this.
        output = self.forward(hidden_states)
        if "preds" not in output or output["preds"] is None:
            raise NotImplementedError("The 'sample' method must be implemented for heads that don't return 'preds' in forward.")
        return output["preds"]
