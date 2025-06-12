# tests/test_configs.py

import pytest
from pydantic import ValidationError
from temporal.configs.transformer_config import (
    TransformerTimeSeriesConfig,
    AttentionConfig,
    FeedForwardConfig,
    ArchitectureConfig,
    TransformerBlockConfig
)

def test_attention_config_defaults():
    """Tests that AttentionConfig can be initialized with minimal arguments."""
    cfg = AttentionConfig(attention_type="full")
    assert cfg.attention_type == "full"
    assert cfg.num_heads is None # Defaults to None, to be filled by main config

def test_ffn_config_defaults():
    """Tests that FeedForwardConfig can be initialized with minimal arguments."""
    cfg = FeedForwardConfig(type="standard", intermediate_size=128)
    assert cfg.type == "standard"
    assert cfg.intermediate_size == 128
    assert cfg.activation == "gelu"

def test_transformer_block_config_initialization():
    """Tests initialization of a single TransformerBlockConfig."""
    attn_cfg = AttentionConfig(attention_type="full", num_heads=4)
    ffn_cfg = FeedForwardConfig(type="standard", intermediate_size=128)
    block_cfg = TransformerBlockConfig(
        block_type="default_encoder",
        attention_config=attn_cfg,
        ffn_config=ffn_cfg
    )
    assert block_cfg.block_type == "default_encoder"
    assert block_cfg.attention_config.num_heads == 4

def test_main_config_initialization():
    """Tests the initialization of the main TransformerTimeSeriesConfig."""
    config = TransformerTimeSeriesConfig(
        d_model=32,
        num_heads=4,
        feature_size=5,
        prediction_length=10,
        context_length=50,
        encoder_blocks=[
            TransformerBlockConfig(block_type="default_encoder")
        ],
        decoder_blocks=[
            TransformerBlockConfig(block_type="default_decoder")
        ]
    )
    assert config.d_model == 32
    assert config.num_heads == 4
    assert config.architecture.layout == "encoder-decoder" # Inferred

def test_main_config_validation_error():
    """Tests that a validation error is raised for inconsistent d_model/num_heads."""
    with pytest.raises(ValidationError):
        # d_model (30) is not divisible by num_heads (4)
        TransformerTimeSeriesConfig(
            d_model=30,
            num_heads=4,
            feature_size=5,
            prediction_length=10,
            context_length=50,
        )

def test_main_config_to_dict_serialization():
    """Tests that the config can be successfully serialized to a dictionary."""
    config = TransformerTimeSeriesConfig(
        d_model=32,
        num_heads=4,
        feature_size=5,
        prediction_length=10,
        context_length=50,
    )
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict['d_model'] == 32
    assert config_dict['architecture']['layout'] == "decoder-only" # Default
