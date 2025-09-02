import unittest
import torch
import logging
import pytest
from temporal.models.preprocessor import InputPreprocessor
from temporal.configs.embedding_config import TimeSeriesValueEmbeddingConfig
from temporal.configs.normalization_config import NormalizationConfig

class TestPreprocessor:
    def setup_method(self):
        self.d_model = 16
        self.feature_size = 4
        self.value_embedding_config = TimeSeriesValueEmbeddingConfig(
            type="value", feature_size=self.feature_size
        )
        self.layernorm_embedding_config = NormalizationConfig(type="layer")
        self.dropout = 0.1
        self.preprocessor = InputPreprocessor(
            d_model=self.d_model,
            value_embedding_config=self.value_embedding_config,
            layernorm_embedding_config=self.layernorm_embedding_config,
            dropout=self.dropout,
        )

    def test_preprocessor_initialization(self):
        assert self.preprocessor.value_embedding is not None
        assert self.preprocessor.layernorm_embedding is not None
        assert self.preprocessor.dropout is not None

    def test_process_method(self):
        input_values = torch.randn(2, 10, self.feature_size)
        processed_output = self.preprocessor.process(input_values)
        assert "hidden_states" in processed_output
        assert processed_output["hidden_states"].shape == (2, 10, self.d_model)

    def test_denormalize(self):
        input_tensor = torch.randn(2, 10, self.feature_size)
        denormalized_tensor = self.preprocessor.denormalize(input_tensor)
        assert torch.equal(input_tensor, denormalized_tensor)

    def test_verbose_output(self, caplog):
        with caplog.at_level(logging.INFO):
            input_values = torch.randn(2, 10, 4)
            self.preprocessor.process(input_values, verbose=True)
        assert "Preprocessor: Processing input" in caplog.text