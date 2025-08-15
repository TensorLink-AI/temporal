
import pytest
import torch
from unittest.mock import MagicMock, patch
from temporal.modules.blocks.block_patch_transformer import PatchTransformBlock
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import (
    TransformerTimeSeriesConfig as TransformerConfig,
)


@pytest.fixture
def mock_builder():
    """Creates a mock ModuleBuilder with a basic config."""
    config = TransformerConfig(d_model=32, num_heads=4, feature_size=1)
    builder = ModuleBuilder(config)
    return builder


@pytest.mark.parametrize("order", ["split_first", "merge_first"])
def test_patch_transform_block_init(mock_builder, order):
    """Tests the initialization of PatchTransformBlock for both orders."""
    expansion_factor = 2
    d_model = mock_builder.config.d_model

    block = PatchTransformBlock(
        builder=mock_builder,
        wrapped_block_type="default_encoder",
        expansion_factor=expansion_factor,
        order=order,
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

    block = PatchTransformBlock(
        builder=mock_builder,
        wrapped_block_type="default_encoder",
        expansion_factor=expansion_factor,
        order="split_first",
    )

    input_tensor = torch.randn(2, seq_len, d_model)
    output, _ = block(input_tensor)

    assert output.shape == input_tensor.shape


def test_patch_transform_block_forward_merge_first(mock_builder):
    """Tests the forward pass of PatchTransformBlock with order='merge_first'."""
    expansion_factor = 2
    d_model = mock_builder.config.d_model
    seq_len = 20

    block = PatchTransformBlock(
        builder=mock_builder,
        wrapped_block_type="default_encoder",
        expansion_factor=expansion_factor,
        order="merge_first",
    )

    input_tensor = torch.randn(2, seq_len, d_model)
    output, _ = block(input_tensor)

    assert output.shape == input_tensor.shape


def test_patch_transform_block_invalid_order(mock_builder):
    """Tests that PatchTransformBlock raises an error for an invalid order."""
    with pytest.raises(
        ValueError, match="order must be one of 'split_first' or 'merge_first'"
    ):
        PatchTransformBlock(
            builder=mock_builder,
            wrapped_block_type="default_encoder",
            expansion_factor=2,
            order="invalid_order",
        )


def test_patch_transform_block_merge_first_invalid_expansion(mock_builder):
    """Tests that 'merge_first' order raises an error with expansion_factor != 2."""
    with pytest.raises(
        ValueError,
        match="For 'merge_first' order with an MLP, expansion_factor must be 2.",
    ):
        PatchTransformBlock(
            builder=mock_builder,
            wrapped_block_type="default_encoder",
            expansion_factor=3,
            order="merge_first",
        )


def test_patch_transform_block_non_divisible_d_model_split_first(mock_builder):
    """Tests that 'split_first' order raises an error if d_model is not divisible by expansion_factor."""
    mock_builder.config.d_model = 33  # Not divisible by 2
    with pytest.raises(
        ValueError, match="d_model \(33\) must be divisible by expansion_factor \(2\)"
    ):
        PatchTransformBlock(
            builder=mock_builder,
            wrapped_block_type="default_encoder",
            expansion_factor=2,
            order="split_first",
        )
