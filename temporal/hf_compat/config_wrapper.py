from transformers import PretrainedConfig


class HFCompatibleTimeSeriesConfig(PretrainedConfig):
    model_type = "transformer_time_series"

    @classmethod
    def from_custom(cls, config):
        return cls(**config.to_flat_dict())

    def to_custom(self):
        from temporal.configs.transformer_config import TransformerTimeSeriesConfig
        return TransformerTimeSeriesConfig.from_flat_dict(self.to_dict())
