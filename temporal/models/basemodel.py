import torch
import os
from transformers import PreTrainedModel
from ..configs.basetimeseriesconfig import BaseTimeseriesConfig

class BaseTimeSeriesModel(PreTrainedModel):
    """
    Base class for all time series transformer models with a standard from_pretrained method.
    """

    @classmethod
    def from_pretrained(cls, model_path: str, config=None, device=None, **kwargs):
        """
        Load a pretrained model from a checkpoint.

        Args:
            model_path (str): Path to the model checkpoint or directory.
            config (Optional[TimeSeriesConfig]): Configuration for the model.
            device (Optional[str]): Device to load the model onto ('cpu' or 'cuda').
            **kwargs: Additional arguments for model overrides.

        Returns:
            An instance of the loaded model.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")

        # Load configuration if provided, otherwise infer from checkpoint
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found in {config_path}. Provide a config manually.")
            config = BaseTimeseriesConfig.from_json_file(config_path)

        # Instantiate the correct model class
        model = cls(config)

        # Load weights
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found in {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=torch.device(device or "cpu"))
        model.load_state_dict(state_dict, strict=False)

        # Move model to the specified device
        model.to(device or "cpu")
        model.eval()

        return model
