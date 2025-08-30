import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.attention_config import FullAttentionConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.output_head_config import OutputHeadConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.loss_config import LossConfig

# --- Fixtures ---

@pytest.fixture
def basic_config():
    """Provides a valid, detailed base configuration for component tests."""
    return TransformerTimeSeriesConfig(
        feature_size=1,
        d_model=16,
        context_length=10,
        prediction_length=5,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
        encoder_blocks=[
            EncoderBlockConfig(
                type="default_encoder",
                attention_config=FullAttentionConfig(type="full", num_heads=2),
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                attention_config=FullAttentionConfig(type="full", num_heads=2),
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
    )

# --- Test Cases ---

def test_encoder_only_forward_pass(basic_config):
    """Tests forward pass with an encoder-only architecture."""
    config_dict = basic_config.to_dict()
    config_dict["architecture"]["layout"] = "encoder"
    config_dict.pop("decoder_blocks", None)
    encoder_only_config = TransformerTimeSeriesConfig.from_dict(config_dict)
    model = build_time_series_transformer(encoder_only_config)

    inputs = torch.randn(
        2, encoder_only_config.context_length, encoder_only_config.feature_size
    )
    output = model(encoder_inputs=inputs)
    
    assert output["logits"].shape == (
        2,
        encoder_only_config.context_length,
        encoder_only_config.feature_size,
    )


def test_decoder_only_forward_pass(basic_config):
    """Tests forward pass with a decoder-only architecture."""
    config_dict = basic_config.to_dict()
    config_dict["architecture"]["layout"] = "decoder"
    config_dict.pop("encoder_blocks", None)
    decoder_only_config = TransformerTimeSeriesConfig.from_dict(config_dict)
    model = build_time_series_transformer(decoder_only_config)

    inputs = torch.randn(
        2, decoder_only_config.context_length, decoder_only_config.feature_size
    )
    output = model(decoder_inputs=inputs)

    assert output["logits"].shape == (
        2,
        decoder_only_config.context_length,
        decoder_only_config.feature_size,
    )


def test_encoder_decoder_forward_pass(basic_config):
    """Tests a standard encoder-decoder forward pass."""
    model = build_time_series_transformer(basic_config)
    encoder_inputs = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    decoder_inputs = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    
    output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)

    assert output["logits"].shape == (2, basic_config.prediction_length, basic_config.feature_size)

def test_multi_feature_forward_pass(basic_config):
    """Tests forward pass with more than one feature."""
    config_dict = basic_config.to_dict()
    config_dict["feature_size"] = 3
    config_dict["output_head_config"]["output_size"] = 3
    
    config_dict["value_embedding_config"]["kwargs"]["input_dim"] = 3
    
    multi_feature_config = TransformerTimeSeriesConfig.from_dict(config_dict)
    model = build_time_series_transformer(multi_feature_config)

    encoder_inputs = torch.randn(2, multi_feature_config.context_length, multi_feature_config.feature_size)
    decoder_inputs = torch.randn(2, multi_feature_config.prediction_length, multi_feature_config.feature_size)
    
    output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)

    assert output["logits"].shape == (2, multi_feature_config.prediction_length, multi_feature_config.feature_size)

def test_different_context_prediction_lengths(basic_config):
    """Tests that the model handles different context and prediction lengths."""
    config_dict = basic_config.to_dict()
    config_dict["context_length"] = 20
    config_dict["prediction_length"] = 10
    new_config = TransformerTimeSeriesConfig.from_dict(config_dict)
    model = build_time_series_transformer(new_config)

    encoder_inputs = torch.randn(2, new_config.context_length, new_config.feature_size)
    decoder_inputs = torch.randn(2, new_config.prediction_length, new_config.feature_size)

    output = model(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
    
    assert output["logits"].shape == (2, new_config.prediction_length, new_config.feature_size)


def test_backward_pass(basic_config):
    """Tests that a backward pass can be performed without errors."""
    model = build_time_series_transformer(basic_config)
    encoder_inputs = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    decoder_inputs = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    targets = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)

    output = model(
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        targets=targets,
    )
    
    output["loss"].backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_reproducibility(basic_config):
    """Tests that model initialization and forward pass are reproducible."""
    encoder_inputs = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    decoder_inputs = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)

    torch.manual_seed(0)
    model1 = build_time_series_transformer(basic_config)
    output1 = model1(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)

    torch.manual_seed(0)
    model2 = build_time_series_transformer(basic_config)
    output2 = model2(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)

    assert torch.allclose(output1["logits"], output2["logits"])


def test_inference_mode(basic_config):
    """Tests that model.eval() disables dropout."""
    model = build_time_series_transformer(basic_config)
    model.eval()

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert not module.training


def test_batch_invariance(basic_config):
    """Tests that the model handles different batch sizes."""
    model = build_time_series_transformer(basic_config)
    
    encoder_b1 = torch.randn(1, basic_config.context_length, basic_config.feature_size)
    decoder_b1 = torch.randn(1, basic_config.prediction_length, basic_config.feature_size)
    
    encoder_b4 = torch.randn(4, basic_config.context_length, basic_config.feature_size)
    decoder_b4 = torch.randn(4, basic_config.prediction_length, basic_config.feature_size)

    try:
        model(encoder_inputs=encoder_b1, decoder_inputs=decoder_b1)
        model(encoder_inputs=encoder_b4, decoder_inputs=decoder_b4)
    except Exception as e:
        pytest.fail(f"Model failed with different batch sizes: {e}")


def test_data_types(basic_config):
    """Tests that the model handles different data dtypes."""
    model = build_time_series_transformer(basic_config)
    model.to(torch.float16)
    
    encoder_inputs = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    decoder_inputs = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)

    try:
        model(
            encoder_inputs=encoder_inputs.to(torch.float16),
            decoder_inputs=decoder_inputs.to(torch.float16),
        )
        model.to(torch.float32)
        model(
            encoder_inputs=encoder_inputs.to(torch.float32),
            decoder_inputs=decoder_inputs.to(torch.float32),
        )
    except Exception as e:
        pytest.fail(f"Model failed with different data types: {e}")