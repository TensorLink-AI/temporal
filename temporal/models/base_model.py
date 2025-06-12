import os
import json
import torch
import torch.nn as nn

class BaseTemporalModel(nn.Module):
    """Base class for all full time series model architectures.

    This class defines the interface for temporal models, enforcing the
    implementation of core `forward` and `generate` methods. It also holds
    the model's configuration and main components.

    Saving and loading of this model, especially for Hugging Face compatibility,
    should be handled by the `save_hf` and `load_hf` functions in
    `temporal.utils.hf_accessors`, often used in conjunction with the
    `temporal.utils.hf_adapter.TimeSeriesTransformerModel` wrapper.

    Attributes:
        config: The configuration object containing model hyperparameters.
        encoder (nn.Module or None): The encoder module of the model.
        decoder (nn.Module or None): The decoder module of the model.
        output_heads (nn.Module or None): The module(s) producing the final outputs.
        head_aggregator (callable or None): A function or module to aggregate
            outputs from multiple heads if they exist.
        loss_fn (callable or None): The loss function to be used during training.
    """

    def __init__(
        self,
        config,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        output_heads: nn.Module = None,
        head_aggregator: callable = None,
        loss_fn: callable = None,
    ):
        """Initializes the BaseTemporalModel.

        Args:
            config: The configuration instance for the model.
            encoder (nn.Module, optional): The encoder network. Defaults to None.
            decoder (nn.Module, optional): The decoder network. Defaults to None.
            output_heads (nn.Module, optional): One or more output head modules.
                Defaults to None.
            head_aggregator (callable, optional): A function or module to combine
                outputs from multiple heads. Defaults to None.
            loss_fn (callable, optional): The loss function to use during training.
                Defaults to None.
        """
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.output_heads = output_heads
        self.head_aggregator = head_aggregator
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        """Performs a forward pass through the model.

        This method defines the core computation of the model, from inputs to
        final outputs (before loss calculation). Subclasses must override this
        method.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the forward() method.")

    def generate(self, *args, **kwargs):
        """Generates predictions or forecasts from the model in inference mode.

        This method defines the generation or sampling behavior of the model,
        which might involve autoregressive decoding or other sampling strategies.
        Subclasses must override this method.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the generate() method.")
