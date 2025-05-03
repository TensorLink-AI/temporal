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

    def save_pretrained(self, save_path: str, safe: bool = False):
            """
            Save model weights and config. If `safe=True` and safetensors installed,
            writes `model.safetensors`, otherwise writes `pytorch_model.bin`.
            """
            os.makedirs(save_path, exist_ok=True)

            # 1) Save config.json
            config_path = os.path.join(save_path, "config.json")
            with open(config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)

            # 2) Save weights
            if safe:
                if not _has_safetensors:
                    raise RuntimeError("`safe=True` requires the `safetensors` library.")
                path = os.path.join(save_path, "model.safetensors")
                # state_dict must be all CPU tensors
                sd = {k: v.cpu() for k, v in self.state_dict().items()}
                _safetensors_save(sd, path)
            else:
                path = os.path.join(save_path, "pytorch_model.bin")
                torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: str, config_cls=None, safe: bool = False, **kwargs):
        """
        Load config & weights. If `safe=True`, tries to read `model.safetensors`,
        otherwise `pytorch_model.bin`.
        """
        # 1) Load config
        cfg_path = os.path.join(path, "config.json")
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
        config = (
            config_cls.from_dict(cfg_dict)
            if (config_cls and hasattr(config_cls, "from_dict"))
            else cfg_dict
        )

        # 2) Initialize model
        model = cls(config, **kwargs)

        # 3) Load weights
        if safe:
            if not _has_safetensors:
                raise RuntimeError("`safe=True` requires the `safetensors` library.")
            weights_path = os.path.join(path, "model.safetensors")
            sd = _safetensors_load(weights_path)
            # safetensors returns cpu tensors
            model.load_state_dict(sd)
        else:
            weights_path = os.path.join(path, "pytorch_model.bin")
            sd = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(sd)

        return model
