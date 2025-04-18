import os
import torch
from transformers import PreTrainedModel

from temporal.configs.transformer_config import TransformerTimeSeriesConfig
from temporal.models.builder import build_time_series_transformer


class TimeSeriesTransformerModel(PreTrainedModel):
    """
    Hugging Face-compatible wrapper for loading a Temporal transformer model
    from a local checkpoint (not from hub yet).
    """
    config_class = TransformerTimeSeriesConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = build_time_series_transformer(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str, config=None, device=None, **kwargs):
        """
        Load a model from a local checkpoint folder (containing config.json and pytorch_model.bin).

        Args:
            model_path: path to folder
            config: optional pre-loaded config object
            device: optional device to move model to
        """
        if not os.path.isdir(model_path):
            raise ValueError(f"{model_path} is not a valid directory.")

        # === Load config ===
        config_path = os.path.join(model_path, "config.json")
        if config is None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found at {config_path}")
            config = cls.config_class.from_json(config_path)

        # === Initialize model ===
        model = cls(config)

        # === Load weights ===
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device or "cpu")
        model.load_state_dict(state_dict, strict=False)

        # === Finalize ===
        if device is not None:
            model = model.to(device)
        model.eval()
        return model
