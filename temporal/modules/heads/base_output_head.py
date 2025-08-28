import torch.nn as nn
import torch
from typing import Callable, Optional

class BaseOutputHead(nn.Module):
    """An abstract base class for all model output heads.

    This class defines the common interface that all output head modules must
    adhere to. An output head is responsible for taking the final hidden state
    from the model's backbone and transforming it into the desired output format
    (e.g., a point forecast, a probability distribution).

    It also defines a method for retrieving the appropriate loss function
    to be used with the head's output.
    """

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Processes the model's final hidden state to produce the output.

        This method must be implemented by all subclasses.

        Args:
            hidden_state (torch.Tensor): The final hidden state from the model's
                backbone, typically of shape `[batch_size, seq_len, d_model]`.

        Returns:
            torch.Tensor: The model's final output, with its shape and meaning
                determined by the specific head implementation.
        """
        raise NotImplementedError("Each head must implement the forward method.")

    def get_loss_fn(self) -> Optional[Callable]:
        """
        Returns the default loss function associated with this head.

        This method should be implemented by subclasses to provide a suitable
        loss function for the type of output they produce. For example, a
        point forecast head might return Mean Squared Error, while a
        probabilistic head might return Negative Log-Likelihood.

        Returns:
            Optional[Callable]: A callable loss function, or None if the head
            does not have a default loss.
        """
        raise NotImplementedError("Each head must provide its corresponding loss function.")
