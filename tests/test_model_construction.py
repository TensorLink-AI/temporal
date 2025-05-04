# tests/test_module_builder.py

import pytest
import torch.nn as nn

# --- Core Components ---
# Assuming ModuleBuilderHelper is where ModuleBuilder resides now
from temporal.models.module_builder_helper import ModuleBuilder
# Assuming config structure is defined here
from temporal.configs.transformer_config import (
    TransformerTimeSeriesConfig,
    AttentionConfig,
    EmbeddingConfig,
    # Import other necessary config components if needed
)

# --- Example Module Classes (Import the actual classes you want to test) ---
# Attention
from temporal.modules.attentions.common_attentions import StandardAttention
from temporal.modules.attentions.time_attention import TimeAttention
from temporal.modules.attentions.hybrid_attention import HybridAttention
# Embedders (replace with your actual embedder classes and paths)
from temporal.modules.embedders.embedding import FeatureProjectionEmbedding, TimeFeatureEmbedding

# --- Test Fixtures ---

@pytest.fixture
def base_config_dict():
    """Provides a base dictionary for TransformerTimeSeriesConfig."""
    # Adjust with minimal valid settings your config requires
    return {
        "architecture": {"layout": "decoder"}, # Example
        "embedding": { # Minimal embedding config for builder init if needed
            "feature_embed_dim": 64,
            "time_embed_dim": 16,
            "cat_embed_dims": {},
            "embedding_type": "feature_projection", # Default
            "time_encoding_strategy": "positional",
        },
        "decoder_blocks": [], # Needs to be present but can be empty for these tests
        "output_head": {"head_type": "point_forecast", "loss_fn": "mse", "target_idx": [0]},
        "context_length": 50,
        "prediction_length": 10,
        "input_chunk_length": 50,
        "output_chunk_length": 10,
        "input_feature_dims": {"num_feat_dynamic_real": 1},
        "target_lags": {},
        # Derived value: hidden_size often depends on embedding dims
        "hidden_size": 64 + 16, # Example: feature_embed_dim + time_embed_dim
    }

@pytest.fixture
def builder(base_config_dict):
    """Provides an instance of ModuleBuilder."""
    # Create the main config object required by ModuleBuilder
    # Use ** to unpack the dictionary into keyword arguments
    main_config = TransformerTimeSeriesConfig(**base_config_dict)
    return ModuleBuilder(main_config)

# --- Attention Module Tests ---

def test_build_standard_attention(builder):
    """Test building the standard attention module."""
    # Config for the specific attention block
    attention_cfg = AttentionConfig(
        attention_type="standard", # Key used in @register_module
        num_heads=4,
        dropout=0.1
    )
    # Pass the specific AttentionConfig to the builder method
    attention_module = builder.build_attention(attention_cfg)

    assert isinstance(attention_module, StandardAttention)
    assert attention_module.num_heads == 4
    # Verify common parameters passed from main config via builder logic
    assert attention_module.embed_dim == builder.config.hidden_size

def test_build_time_attention(builder):
    """Test building the time-aware attention module with specific kwargs."""
    # Config for the specific attention block
    attention_cfg = AttentionConfig(
        attention_type="time", # Key used in @register_module
        num_heads=8,
        dropout=0.15,
        kwargs={ # Specific arguments for TimeAttention passed via kwargs
            "max_position_embeddings": 1024,
            "rope_base": 5000,
            "num_rel_pos_buckets": 16,
            "max_rel_pos_distance": 64
        }
    )
    attention_module = builder.build_attention(attention_cfg)

    assert isinstance(attention_module, TimeAttention)
    assert attention_module.num_heads == 8
    assert attention_module.dropout == 0.15 # Check if specific dropout is used
    assert attention_module.embed_dim == builder.config.hidden_size
    # Check specific parameters were passed correctly (assuming they are stored)
    assert attention_module.max_position_embeddings == 1024
    assert attention_module.rope_base == 5000
    assert attention_module.num_rel_pos_buckets == 16
    assert attention_module.max_rel_pos_distance == 64


def test_build_hybrid_attention(builder):
    """Test building the hybrid attention module."""
     # Config for the specific attention block
    attention_cfg = AttentionConfig(
        attention_type="hybrid", # Key used in @register_module
        num_heads=8, # Total heads
        dropout=0.1,
        kwargs={ # Specific arguments for HybridAttention
             "head_splits": [4, 4], # Example split
             "head_types": ["standard", "time"] # Example types
             # Add other required kwargs for HybridAttention if any
        }
    )
    attention_module = builder.build_attention(attention_cfg)
    assert isinstance(attention_module, HybridAttention)
    assert attention_module.num_heads == 8
    assert attention_module.embed_dim == builder.config.hidden_size
    # Check specific HybridAttention parameters
    assert attention_module.head_splits == [4, 4]
    assert attention_module.head_types == ["standard", "time"]

def test_build_attention_invalid_type(builder):
    """Test building with an unregistered attention type."""
    attention_cfg = AttentionConfig(attention_type="non_existent_attention", num_heads=4)
    with pytest.raises(KeyError): # Assuming registry resolve raises KeyError
        builder.build_attention(attention_cfg)

def test_build_attention_head_dim_mismatch(builder):
    """Test ValueError when embed_dim is not divisible by num_heads."""
    # Temporarily modify the builder's config for this test
    original_hidden_size = builder.config.hidden_size
    builder.config.hidden_size = 81 # Not divisible by 4 or 8

    attention_cfg = AttentionConfig(attention_type="standard", num_heads=4)
    # BaseMultiHeadAttention raises this error during init
    with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
        builder.build_attention(attention_cfg)

    # Reset hidden_size for other tests if builder fixture scope requires it
    builder.config.hidden_size = original_hidden_size

# --- Embedding Module Tests ---

# Note: You might need a different fixture or config adjustment if
# the embedder builder method expects different inputs/config structure.
# Assuming builder.build_embedding(embedding_config_obj) exists.

@pytest.fixture
def embedding_config(base_config_dict):
     """ Creates an EmbeddingConfig object from the base dict's embedding part."""
     # Directly create the EmbeddingConfig instance
     return EmbeddingConfig(**base_config_dict['embedding'])


# Assuming build_embedding takes the main config's embedding section/object
def test_build_feature_projection_embedding(builder, embedding_config):
     """Test building the FeatureProjectionEmbedding."""
     # Adjust the type if needed for the test
     embedding_config.embedding_type = "feature_projection"
     # Assume build_embedding takes the embedding config part
     # Or adjust if it takes the full config: builder.build_embedding()
     embedding_module = builder.build_embedding(embedding_config)

     assert isinstance(embedding_module, FeatureProjectionEmbedding)
     assert embedding_module.embed_dim == embedding_config.feature_embed_dim


def test_build_time_feature_embedding(builder, embedding_config):
     """Test building the TimeFeatureEmbedding."""
     embedding_config.embedding_type = "time_feature" # Key for TimeFeatureEmbedding
     embedding_config.time_encoding_strategy = "learned" # Example specific arg
     embedding_config.kwargs = {"max_seq_len": 512} # Example specific kwarg

     # Assume build_embedding uses the config passed to it
     embedding_module = builder.build_embedding(embedding_config)

     assert isinstance(embedding_module, TimeFeatureEmbedding)
     # Check specific parameters for TimeFeatureEmbedding (adjust based on its __init__)
     assert embedding_module.time_embed_dim == embedding_config.time_embed_dim
     assert embedding_module.strategy == "learned"
     assert embedding_module.max_seq_len == 512


def test_build_embedding_invalid_type(builder, embedding_config):
     """Test building with an unregistered embedding type."""
     embedding_config.embedding_type = "non_existent_embedding"
     with pytest.raises(KeyError): # Assuming registry resolve raises KeyError
         builder.build_embedding(embedding_config)


# --- Integration Test (Optional but recommended) ---
from temporal.models.builder import build_time_series_transformer

def test_full_model_build_with_specific_attention(base_config_dict):
    """Test that the full model builder uses the correct attention type."""
    # Modify the base config dict to specify a particular attention type
    base_config_dict["hidden_size"] = 80 # Ensure divisibility for heads (e.g., 8)
    base_config_dict["embedding"]["feature_embed_dim"] = 64 # Adjust dims to sum to hidden_size
    base_config_dict["embedding"]["time_embed_dim"] = 16

    # Define a decoder block using 'time' attention
    base_config_dict["decoder_blocks"] = [
        {
            "block_type": "transformer",
            "num_layers": 1, # Keep it simple for testing
            "attention": {
                "attention_type": "time",
                "num_heads": 8, # 80 hidden / 8 heads = 10 dim/head
                "dropout": 0.1,
                "kwargs": { # TimeAttention specific args
                    "max_position_embeddings": 512
                }
            },
            "feedforward": { # Need a valid feedforward config too
                "ff_type": "standard",
                "dim": 80,
                "hidden_dim_multiplier": 2,
                 "dropout": 0.1,
            },
             "normalization": {"norm_type": "layer_norm"}
        }
    ]

    # Create the full config object
    full_config = TransformerTimeSeriesConfig(**base_config_dict)

    # Build the full model
    model = build_time_series_transformer(full_config)

    assert model is not None
    assert isinstance(model, nn.Module) # Basic check

    # Inspect the built model's submodules (path depends on your model structure)
    # Example: Assuming decoder-only layout defined in base_config_dict
    assert hasattr(model, 'decoder'), "Model should have a decoder"
    assert hasattr(model.decoder, 'blocks'), "Decoder should have blocks"
    assert len(model.decoder.blocks) > 0, "Decoder should contain blocks"
    # Access the attention module within the first block (adjust path if needed)
    block_attention = model.decoder.blocks[0].attention
    assert isinstance(block_attention, TimeAttention), \
        f"Expected TimeAttention, but got {type(block_attention)}"
    assert block_attention.num_heads == 8
    assert block_attention.max_position_embeddings == 512
