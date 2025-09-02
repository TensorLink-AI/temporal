from transformers import PretrainedConfig


class HFCompatibleTimeSeriesConfig(PretrainedConfig):
    model_type = "transformer_time_series"

    @classmethod
    def from_custom(cls, config):
        # Initialize using the custom config's dict
        instance = cls(**config.to_flat_dict())
        # Explicitly set the correct model_type, overriding any value from the input dict
        instance.model_type = cls.model_type
        return instance

    def to_custom(self):
        from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
        # Convert HF config back to a dictionary
        custom_dict = self.to_dict()
        # Remove HF-specific keys potentially not expected by the custom config's from_dict
        custom_dict.pop("model_type", None)
        custom_dict.pop("_name_or_path", None)
        # Use the custom config's standard from_dict method
        return TransformerTimeSeriesConfig.from_dict(custom_dict)
