import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.models.transformer_model import TransformerOutput
from temporal.configs.output_head_config import OutputHeadConfig
# FIX: Add missing imports for config objects
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.loss_config import LossConfig
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.feedforward_config import StandardFeedForwardConfig

# --- Fixtures ---


@pytest.fixture(scope="module")
def model_config():
    """Provides a simple, valid config for model testing."""
    return TransformerTimeSeriesConfig(
        d_model=16,
        feature_size=3,
        prediction_length=5,
        context_length=10,
        # FIX: Instantiate the ArchitectureConfig object directly with a 'type'
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        # FIX: Instantiate the LossConfig object with a valid registered type
        loss_config=LossConfig(type="timeseries_generic"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
        encoder_blocks=[
            EncoderBlockConfig(
                type="default_encoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=32
                ),
            )
        ],
    )


@pytest.fixture(scope="module")
def model(model_config):
    """Provides a fully built model instance."""
    return build_time_series_transformer(model_config)


# --- Test Cases ---


def test_model_initialization(model, model_config):
    """
    Tests that the model and its components are initialized correctly.
    """
    assert model.config == model_config
    assert hasattr(model, "encoder") and model.encoder is not None
    assert hasattr(model, "decoder") and model.decoder is not None
    assert hasattr(model, "output_heads") and model.output_heads is not None
    assert hasattr(model, "loss_fn") and model.loss_fn is not None


def test_model_device_placement(model):
    """
    Tests that the model and its parameters can be moved to a different device.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for device placement test")

    device = torch.device("cuda")
    model.to(device)

    # Check that all parameters have been moved to the target device
    for param in model.parameters():
        assert param.device == device


def test_enable_dropout_method(model):
    """
    Tests the `enable_dropout` method to ensure it sets dropout layers to train mode.
    """
    # First, set model to eval mode, which deactivates dropout
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert not module.training

    # Enable dropout using the mixin method
    model.enable_dropout()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.training

    # Set back to eval mode for other tests
    model.eval()


def test_model_output_dataclass():
    """
    Tests the functionality of the TransformerOutput dataclass.
    """
    output = TransformerOutput(logits=torch.randn(2, 5, 3), loss=torch.tensor(0.5))

    # Test attribute access
    assert output.loss == 0.5

    # Test item access
    assert output["loss"] == 0.5
    assert torch.equal(output["logits"], output.logits)

    # Test keys method
    assert "logits" in output.keys()
    assert "loss" in output.keys()
    assert "past_key_values" in output.keys()