import torch
from temporal.configs.transformer_config import (
    TransformerTimeSeriesConfig,
    TransformerArchitectureConfig,
    # TransformerAttentionBlockConfig, # Not used if blocks are defined below
    TransformerBlockConfig,
    AttentionConfig,
    FeedForwardConfig,
    OutputHeadConfig,
)

from temporal.models.builder import build_time_series_transformer

# Define constants
HIDDEN_SIZE = 32
NUM_HEADS = 2

# 1) Define encoder and decoder block configurations
encoder_blocks = [
    TransformerBlockConfig(
        block_type="default_encoder",
        # Reverted: AttentionConfig does not take embed_dim
        attention_config=AttentionConfig(
            attention_type="full",
            num_heads=NUM_HEADS,
            dropout=0.1
        ),
        ffn_config=FeedForwardConfig(type="standard", intermediate_size=64, activation="gelu", dropout=0.1),
    )
    for _ in range(2)
]
decoder_blocks = [
    TransformerBlockConfig(
        block_type="default_decoder",
        # Reverted: AttentionConfig does not take embed_dim
        attention_config=AttentionConfig(
            attention_type="full",
            num_heads=NUM_HEADS,
            dropout=0.1
        ),
        ffn_config=FeedForwardConfig(type="standard", intermediate_size=64, activation="gelu", dropout=0.1),
    )
    for _ in range(2)
]

# 2) Assemble the top-level config, passing blocks directly
config = TransformerTimeSeriesConfig(
    feature_size=1,
    context_length=16,
    prediction_length=4,
    quantiles=[0.1, 0.5, 0.9],
    loss_type="quantile",
    hidden_size=HIDDEN_SIZE, # Should provide embed_dim internally
    output_token_lengths=1,
    architecture=TransformerArchitectureConfig(
        layout="encoder-decoder",
        num_encoder_layers=len(encoder_blocks),
        num_decoder_layers=len(decoder_blocks),
    ),
    output_head_config=OutputHeadConfig(type="linear", output_size=1),
    # Pass the block configurations directly here
    encoder_blocks=encoder_blocks,
    decoder_blocks=decoder_blocks,
)a

# 3) Build and test
model = build_time_series_transformer(config)

# dummy batch: (batch, time, features)
x = torch.randn(4, config.context_length, config.feature_size)

# generate forecasts
y_hat = model.generate(x)

print("✅ model built and ran! Output shape:", y_hat.shape)
