import os
import json
import torch
import torch.nn as nn


class BaseTemporalModel(nn.Module):
    """
    Base class for all full time series architectures.
    """

    def __init__(self, config, encoder=None, decoder=None, output_heads=None, head_aggregator=None, loss_fn=None):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.output_heads = output_heads
        self.head_aggregator = head_aggregator
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement generate()")

    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # Save weights
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        # Save config
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, config_cls=None):
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)
        config = config_cls.from_dict(config_dict) if config_cls else config_dict

        model = cls(config)  # ✅ This is the key fix: use cls, not a builder
        model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        return model

