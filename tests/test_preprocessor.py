import logging
import torch
import pytest
from temporal.models.preprocessor import InputPreprocessor
from temporal.configs.embedding_config import TimeSeriesValueEmbeddingConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig

from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.models.module_builder_helper import ModuleBuilder

class TestPreprocessor:
    def setup_method(self):
        self.d_model = 16
        self.feature_size = 4
        self.value_embedding_config = TimeSeriesValueEmbeddingConfig(
            type="value", feature_size=self.feature_size
        )
        self.layernorm_embedding_config = NormalizationConfig(type="layer")
        self.dropout = 0.1

        # New: build via (config, builder)
        self.config = TransformerTimeSeriesConfig(
            d_model=self.d_model,
            feature_size=self.feature_size,
            value_embedding_config=self.value_embedding_config,
            layer_norm_config=self.layernorm_embedding_config,
            hidden_dropout_prob=self.dropout,
        )
        self.builder = ModuleBuilder(self.config)
        self.preprocessor = InputPreprocessor(self.config, self.builder)

    def test_verbose_output(self, caplog):
        with caplog.at_level(logging.INFO):
            input_values = torch.randn(2, 10, 4)
            out = self.preprocessor.process(input_values, verbose=True)
        assert out["hidden_states"].shape[0] == 2


    def test_preprocessor_initialization(self):
        assert self.preprocessor.value_embedding is not None
        assert self.preprocessor.layernorm_embedding is not None
        assert self.preprocessor.dropout is not None

    def test_process_method(self):
        input_values = torch.randn(2, 10, self.config.feature_size)
        processed_output = self.preprocessor.process(input_values)
        assert "hidden_states" in processed_output
        assert processed_output["hidden_states"].shape == (2, 10, self.config.d_model)

    def test_denormalize(self):
        input_tensor = torch.randn(2, 10, self.config.feature_size)
        denormalized_tensor = self.preprocessor.denormalize(input_tensor)
        assert torch.equal(input_tensor, denormalized_tensor)


