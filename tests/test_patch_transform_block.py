import pytest
import torch
from unittest.mock import MagicMock, patch

from temporal.modules.blocks.block_patch_transformer import PatchTransformBlock
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.mixin.adaptive_patching import PatchSplitting, PatchMerging
from temporal.configs.transformer_model_config import (
    TransformerTimeSeriesConfig as TransformerConfig,
)
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    transformer_block_config_from_dict,
    AdaptivePatchTransformerBlockConfig
)
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import (
    FeedForwardConfig,
    StandardFeedForwardConfig
)
# FIX: Import the registry and manually add the missing config to it.
from temporal.configs.base_config import CONFIG_REGISTRY

# This ensures the 'default_encoder' key is available before any tests run.
if "default_encoder" not in CONFIG_REGISTRY:
    CONFIG_REGISTRY["default_encoder"] = EncoderBlockConfig


@pytest.fixture
def mock_builder():
    """Creates a mock ModuleBuilder with a basic config."""
    config = TransformerConfig(
        d_model=32,
        feature_size=1,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )
    builder = ModuleBuilder(config)
    return builder


@pytest.mark.parametrize("order", ["split_first", "merge_first"])
def test_patch_transform_block_init(mock_builder, order):
    """Tests the initialization of PatchTransformBlock for both orders."""
    expansion_factor = 2
    d_model = mock_builder.config.d_model

    default_attention_config = AttentionConfig(type="vanilla", num_heads=4)
    default_ffn_config = StandardFeedForwardConfig(intermediate_size=64)

    patch_block_config = AdaptivePatchTransformerBlockConfig(
        expansion_factor=expansion_factor,
        wrapped_block_type="default_encoder",
        order=order,
        attention_config=default_attention_config,
        ffn_config=default_ffn_config,
    )

    block = PatchTransformBlock(
        config=patch_block_config,
        builder=mock_builder,
    )

    assert block.order == order
    assert block.expansion_factor == expansion_factor
    assert hasattr(block, "transformer_layer")

    if order == "split_first":
        assert block.transformer_layer.config.d_model == d_model // expansion_factor
    else:  # merge_first
        assert block.transformer_layer.config.d_model == d_model * expansion_factor


def test_patch_transform_block_forward_split_first(mock_builder):
    """Tests the forward pass of PatchTransformBlock with order='split_first'."""
    expansion_factor = 2
    d_model = mock_builder.config.d_model
    seq_len = 20

    default_attention_config = AttentionConfig(type="vanilla", num_heads=4)
    default_ffn_config = StandardFeedForwardConfig(intermediate_size=64)

    patch_block_config = AdaptivePatchTransformerBlockConfig(
        expansion_factor=expansion_factor,
        wrapped_block_type="default_encoder",
        order="split_first",
        attention_config=default_attention_config,
        ffn_config=default_ffn_config,
    )

    block = PatchTransformBlock(
        config=patch_block_config,
        builder=mock_builder,
    )

    input_tensor = torch.randn(2, seq_len, d_model)
    output, _ = block(input_tensor)

    assert output.shape == input_tensor.shape


def test_patch_transform_block_forward_merge_first(mock_builder):
    """Tests the forward pass of PatchTransformBlock with order='merge_first'."""
    expansion_factor = 2
    d_model = mock_builder.config.d_model
    seq_len = 20

    default_attention_config = AttentionConfig(type="vanilla", num_heads=4)
    default_ffn_config = StandardFeedForwardConfig(intermediate_size=64)

    patch_block_config = AdaptivePatchTransformerBlockConfig(
        expansion_factor=expansion_factor,
        wrapped_block_type="default_encoder",
        order="merge_first",
        attention_config=default_attention_config,
        ffn_config=default_ffn_config,
    )

    block = PatchTransformBlock(
        config=patch_block_config,
        builder=mock_builder,
    )

    input_tensor = torch.randn(2, seq_len, d_model)
    output, _ = block(input_tensor)

    assert output.shape == input_tensor.shape


def test_patch_transform_block_invalid_order(mock_builder):
    """Tests that PatchTransformBlock raises an error for an invalid order."""
    expansion_factor = 2
    default_attention_config = AttentionConfig(type="vanilla", num_heads=4)
    default_ffn_config = StandardFeedForwardConfig(intermediate_size=64)

    with pytest.raises(
        ValueError, match="order must be one of 'split_first' or 'merge_first'"
    ):
        AdaptivePatchTransformerBlockConfig(
            expansion_factor=expansion_factor,
            wrapped_block_type="default_encoder",
            order="invalid_order",
            attention_config=default_attention_config,
            ffn_config=default_ffn_config,
        )


def test_patch_transform_block_merge_first_invalid_expansion(mock_builder):
    """Tests that 'merge_first' order raises an error with expansion_factor != 2."""
    expansion_factor = 3  # Invalid for merge_first

    default_attention_config = AttentionConfig(type="vanilla", num_heads=4)
    default_ffn_config = StandardFeedForwardConfig(intermediate_size=64)

    with pytest.raises(
        ValueError,
        match="For 'merge_first' order with an MLP, expansion_factor must be 2.",
    ):
        patch_block_config = AdaptivePatchTransformerBlockConfig(
            expansion_factor=expansion_factor,
            wrapped_block_type="default_encoder",
            order="merge_first",
            attention_config=default_attention_config,
            ffn_config=default_ffn_config,
        )
        PatchTransformBlock(
            config=patch_block_config,
            builder=mock_builder
        )


def test_patch_transform_block_non_divisible_d_model_split_first(mock_builder):
    """Tests that 'split_first' order raises an error if d_model is not divisible by expansion_factor."""
    config_dict = mock_builder.config.to_dict()
    config_dict["d_model"] = 33  # Not divisible by 2
    bad_config = TransformerConfig.from_dict(config_dict)
    bad_builder = ModuleBuilder(bad_config)

    expansion_factor = 2

    default_attention_config = AttentionConfig(type="vanilla", num_heads=4)
    default_ffn_config = StandardFeedForwardConfig(intermediate_size=64)

    with pytest.raises(
        ValueError, match=r"d_model \(33\) must be divisible by expansion_factor \(2\)"
    ):
        patch_block_config = AdaptivePatchTransformerBlockConfig(
            expansion_factor=expansion_factor,
            wrapped_block_type="default_encoder",
            order="split_first",
            attention_config=default_attention_config,
            ffn_config=default_ffn_config,
        )
        PatchTransformBlock(
            config=patch_block_config,
            builder=bad_builder,
        )