import pytest
from pydantic import ValidationError
from temporal.configs.transformer_model_config import (
    TransformerTimeSeriesConfig,
)
from temporal.configs.attention_config import FullAttentionConfig
from temporal.configs.feedforward_config import StandardFeedForwardConfig
from temporal.configs.architecture_config import (
    TransformerArchitectureConfig as ArchitectureConfig,
)
from temporal.configs.transformer_block_config import (
    EncoderBlockConfig,
    DecoderBlockConfig,
)
from temporal.configs.embedding_config import EmbeddingConfig
from temporal.models.builder import build_time_series_transformer
from temporal.configs.loss_config import LossConfig
from temporal.configs.output_head_config import OutputHeadConfig


def test_attention_config_defaults():
    """Tests that AttentionConfig can be initialized with minimal arguments."""
    cfg = FullAttentionConfig(type="full", num_heads=4)
    assert cfg.type == "full"
    assert cfg.num_heads == 4


def test_ffn_config_defaults():
    """Tests that FeedForwardConfig can be initialized with minimal arguments."""
    cfg = StandardFeedForwardConfig(type="standard", intermediate_size=128)
    assert cfg.type == "standard"
    assert cfg.intermediate_size == 128
    assert cfg.activation == "gelu"


def test_transformer_block_config_initialization():
    """Tests initialization of a single TransformerBlockConfig."""
    attn_cfg = FullAttentionConfig(type="full", num_heads=4)
    ffn_cfg = StandardFeedForwardConfig(type="standard", intermediate_size=128)
    block_cfg = EncoderBlockConfig(
        type="default_encoder", attention_config=attn_cfg, ffn_config=ffn_cfg
    )
    assert block_cfg.type == "default_encoder"
    assert block_cfg.attention_config.num_heads == 4


def test_main_config_initialization():
    """Tests the initialization of the main TransformerTimeSeriesConfig."""
    config = TransformerTimeSeriesConfig(
        d_model=32,
        feature_size=5,
        prediction_length=10,
        context_length=50,
        encoder_blocks=[
            EncoderBlockConfig(
                type="default_encoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=64
                ),
            )
        ],
        decoder_blocks=[
            DecoderBlockConfig(
                type="default_decoder",
                ffn_config=StandardFeedForwardConfig(
                    type="standard", intermediate_size=64
                ),
            )
        ],
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        # FIX: Removed `loss_type` and set `type` directly to "mse"
        loss_config=LossConfig(type="mse"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )
    assert config.d_model == 32
    assert config.architecture.layout == "encoder-decoder"


def test_main_config_validation_error():
    """Tests that a validation error is raised for inconsistent d_model/num_heads."""
    with pytest.raises(ValueError):
        # d_model (30) is not divisible by num_heads (4)
        TransformerTimeSeriesConfig(
            d_model=30,
            feature_size=5,
            prediction_length=10,
            context_length=50,
            encoder_blocks=[
                EncoderBlockConfig(
                    type="default_encoder",
                    attention_config=FullAttentionConfig(type="full", num_heads=4),
                    ffn_config=StandardFeedForwardConfig(
                        type="standard", intermediate_size=64
                    ),
                )
            ],
            architecture=ArchitectureConfig(
                type="transformer_architecture", layout="encoder-decoder"
            ),
            # FIX: Removed `loss_type` and set `type` directly to "mse"
            loss_config=LossConfig(type="mse"),
            output_head_config=OutputHeadConfig(type="linear", output_size=1),
        )


def test_main_config_to_dict_serialization():
    """Tests that the config can be successfully serialized to a dictionary."""
    config = TransformerTimeSeriesConfig(
        d_model=32,
        feature_size=5,
        prediction_length=10,
        context_length=50,
        architecture=ArchitectureConfig(
            type="transformer_architecture", layout="encoder-decoder"
        ),
        # FIX: Removed `loss_type` and set `type` directly to "mse"
        loss_config=LossConfig(type="mse"),
        output_head_config=OutputHeadConfig(type="linear", output_size=1),
    )
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["d_model"] == 32
    assert config_dict["architecture"]["layout"] == "encoder-decoder"


def test_model_build_with_inconsistent_d_model():
    """
    Tests that building a model with inconsistent d_model values between
    the main config and a component (e.g., embedding) raises a RuntimeError.
    """
    with pytest.raises(RuntimeError, match="Shape mismatch"):
        build_time_series_transformer(
            TransformerTimeSeriesConfig(
                d_model=32,  # Main model dimension
                feature_size=5,
                prediction_length=10,
                context_length=50,
                architecture=ArchitectureConfig(
                    type="transformer_architecture", layout="encoder-decoder"
                ),
                # Override embedding config with a different d_model
                value_embedding_config=EmbeddingConfig(
                    type="linear", kwargs={"d_model": 64}
                ),
                encoder_blocks=[
                    EncoderBlockConfig(
                        type="default_encoder",
                        ffn_config=StandardFeedForwardConfig(
                            type="standard", intermediate_size=64
                        ),
                    )
                ],
                decoder_blocks=[
                    DecoderBlockConfig(
                        type="default_decoder",
                        ffn_config=StandardFeedForwardConfig(
                            type="standard", intermediate_size=64
                        ),
                    )
                ],
                # FIX: Removed `loss_type` and set `type` directly to "mse"
                loss_config=LossConfig(type="mse"),
                output_head_config=OutputHeadConfig(type="linear", output_size=1),
            )
        )