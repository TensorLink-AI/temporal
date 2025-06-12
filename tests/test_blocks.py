# tests/test_blocks.py

import pytest
import torch
from unittest.mock import MagicMock
from temporal.modules.blocks.block_multiscale import MultiScaleBlock
from temporal.modules.blocks.effitime_block import EffiTimeBlockHybridConvFirst

# --- Fixtures ---

@pytest.fixture
def mock_attention_module():
    """Creates a mock attention module that returns predictable shapes."""
    def mock_forward(hidden_states, **kwargs):
        # Returns a tuple: (output, attention_weights, kv_cache)
        return hidden_states, torch.randn(*hidden_states.shape[:2], hidden_states.shape[1]), None
    
    mock_attn = MagicMock()
    mock_attn.side_effect = mock_forward
    return mock_attn

# --- MultiScaleBlock Tests ---

@pytest.mark.parametrize("fusion_method", ["add", "concat", "gate"])
def test_multiscale_block_init(mock_attention_module, fusion_method):
    """Tests the initialization of MultiScaleBlock with different fusion methods."""
    block = MultiScaleBlock(
        fine_attn=mock_attention_module,
        coarse_attn=mock_attention_module,
        fusion_method=fusion_method,
        hidden_size=32
    )
    assert block.fusion_method == fusion_method
    if fusion_method == "concat":
        assert hasattr(block, "fuse_proj")
    elif fusion_method == "gate":
        assert hasattr(block, "gate")

def test_multiscale_block_forward_pass(mock_attention_module):
    """Tests the forward pass of MultiScaleBlock, ensuring correct output shape."""
    hidden_size = 32
    block = MultiScaleBlock(
        fine_attn=mock_attention_module,
        coarse_attn=mock_attention_module,
        hidden_size=hidden_size
    )
    
    input_tensor = torch.randn(2, 20, hidden_size) # seq_len=20
    output, _ = block(input_tensor)
    
    assert output.shape == input_tensor.shape
    assert mock_attention_module.call_count == 2 # Called for fine and coarse scales

def test_multiscale_block_padding(mock_attention_module):
    """Tests that MultiScaleBlock correctly pads input for the coarse scale."""
    hidden_size = 32
    downsample_factor = 4
    block = MultiScaleBlock(
        fine_attn=mock_attention_module,
        coarse_attn=mock_attention_module,
        downsample_factor=downsample_factor,
        hidden_size=hidden_size
    )
    
    # Sequence length (19) is not divisible by downsample_factor (4)
    input_tensor = torch.randn(2, 19, hidden_size)
    output, _ = block(input_tensor)
    
    assert output.shape == input_tensor.shape # Output shape must match original input

# --- EffiTimeBlock Tests ---

def test_effitime_block_init(mock_attention_module):
    """Tests the initialization of the EffiTimeBlock."""
    block = EffiTimeBlockHybridConvFirst(
        attention=mock_attention_module,
        embed_dim=32
    )
    assert hasattr(block, "dw_conv")
    assert hasattr(block, "pw_conv")
    assert hasattr(block, "temporal_fc1")
    assert hasattr(block, "channel_fc1")

def test_effitime_block_forward_pass(mock_attention_module):
    """Tests the forward pass of EffiTimeBlock, ensuring correct output shape."""
    embed_dim = 32
    block = EffiTimeBlockHybridConvFirst(
        attention=mock_attention_module,
        embed_dim=embed_dim
    )
    
    input_tensor = torch.randn(2, 20, embed_dim)
    output, _, _ = block(input_tensor)
    
    assert output.shape == input_tensor.shape
    mock_attention_module.assert_called_once()
