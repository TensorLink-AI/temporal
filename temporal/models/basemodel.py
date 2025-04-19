import os
import json
import torch
import torch.nn as nn


class BaseTemporalModel(nn.Module):
    """Base class for all full time series model architectures.

    This class defines the interface for temporal models, handling common save/load
    functionality and enforcing implementation of core methods.

    Attributes:
        config: Configuration object containing model hyperparameters.
        encoder (nn.Module or None): Encoder module, if any.
        decoder (nn.Module or None): Decoder module, if any.
        output_heads (nn.Module or None): Module(s) producing final outputs.
        head_aggregator (callable or None): Aggregator for multiple output heads.
        loss_fn (callable or None): Loss function used during training.
    """

    def __init__(
        self,
        config,
        encoder=None,
        decoder=None,
        output_heads=None,
        head_aggregator=None,
        loss_fn=None,
    ):
        """Initialize the BaseTemporalModel.

        Args:
            config: Configuration instance for the model.
            encoder (nn.Module, optional): Encoder network.
            decoder (nn.Module, optional): Decoder network.
            output_heads (nn.Module, optional): One or more output head modules.
            head_aggregator (callable, optional): Function/module to combine output heads.
            loss_fn (callable, optional): Loss function to use during training.
        """
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.output_heads = output_heads
        self.head_aggregator = head_aggregator
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        """Perform a forward pass through the model.

        Subclasses must override this method to define the computation logic.

        Raises:
            NotImplementedError: Indicates that subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def generate(self, *args, **kwargs):
        """Generate predictions from the model in inference mode.

        Subclasses must override this method to define generation behavior.

        Raises:
            NotImplementedError: Indicates that subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def save_pretrained(self, save_path: str):
        """Save model weights and configuration to a directory.

        Creates the directory if it does not exist, saves the model's state dict
        as 'pytorch_model.bin', and writes the config JSON to 'config.json'.

        Args:
            save_path (str): Path to the directory where files will be saved.
        """
        os.makedirs(save_path, exist_ok=True)
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        # Save configuration
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, config_cls=None):
        """Load a pretrained model from a directory.

        Reads 'config.json' to reconstruct the configuration (using `config_cls`
        if provided), instantiates the model via `cls(config)`, then loads
        the weights from 'pytorch_model.bin'.

        Args:
            path (str): Path to the directory containing saved artifacts.
            config_cls (type, optional): Class with a `.from_dict()` method to
                convert the config dict to an object. If None, the raw dict is used.

        Returns:
            BaseTemporalModel: An instance of the model with loaded weights.
        """
        # Load configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)
        config = (
            config_cls.from_dict(config_dict)
            if (config_cls and hasattr(config_cls, "from_dict"))
            else config_dict
        )

        # Initialize and load weights
        model = cls(config)
        state_dict = torch.load(os.path.join(path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        return model
