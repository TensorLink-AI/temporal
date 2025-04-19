# tests/test_configs.py

import pytest
from temporal.configs.transformer_config import TransformerConfig


def test_transformer_config_initialization():
    config = TransformerConfig()
    assert config is not None


def test_transformer_config_validation_no_attention_blocks():
    config = TransformerConfig()
    config.attention_blocks = None  # Simulate missing attention blocks
    # The validate method shouldn't raise an AttributeError when attention_blocks is None
    try:
        config.validate()
    except AttributeError:
        pytest.fail("AttributeError raised when attention_blocks is None")


def test_transformer_config_validation_quantile_output_head():
    config = TransformerConfig(output_head_config=dict(type="quantile"))
    try:
        config.validate()
    except Exception as e:
        pytest.fail(f"Validation failed with quantile output head: {e}")


def test_transformer_config_validation_no_encoder_decoder_blocks():
    config = TransformerConfig()
    config.encoder_blocks = None
    config.decoder_blocks = None
    try:
        config.validate()
    except Exception as e:
        pytest.fail(f"Validation failed when encoder or decoder blocks are None: {e}")


def test_transformer_config_validation_invalid_quantile_values():
    with pytest.raises(AssertionError):
        config = TransformerConfig(output_head_config=dict(type="quantile"), quantiles=[0.2, 1.1, 0.5])
        config.validate()


def test_transformer_config_validation_non_list_quantiles():
    with pytest.raises(AssertionError):
        config = TransformerConfig(output_head_config=dict(type="quantile"), quantiles=0.5)
        config.validate()


def test_transformer_config_validation_missing_quantiles():
    with pytest.raises(AssertionError):
        config = TransformerConfig(output_head_config=dict(type="quantile"), quantiles=None)
        config.validate()

class MockInvalidBlockConfig:
    def validate(self):
        raise ValueError("Invalid block configuration")


def test_transformer_config_validation_invalid_block_config():
    with pytest.raises(ValueError, match="Invalid block configuration"):
        config = TransformerConfig(encoder_blocks=[MockInvalidBlockConfig()])
        config.validate()
