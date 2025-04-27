import torch.nn as nn

from temporal.registry.core import resolve
from temporal.configs.transformer_config import (
    AttentionConfig,
    FeedForwardConfig,
    EmbeddingConfig,
    HeadAggregationConfig,
    NormalizationConfig,
)

class ModuleBuilder:
    """
    Constructs individual submodules for a transformer-based time series model.

    Uses the registry to resolve implementation classes for attention, feedforward,
    embedding, normalization, and head aggregation layers based on configuration.

    Args:
        config (Any): Configuration object with model hyperparameters. Needs attributes like hidden_size, feature_size, context_length, prediction_length.
    """
    def __init__(self, config):
        self.config = config

    def build_attention(self, attn_cfg: AttentionConfig) -> nn.Module:
        """
        Build an attention module from its configuration.

        Args:
            attn_cfg (AttentionConfig): Configuration for the attention mechanism.

        Returns:
            nn.Module: Instantiated attention layer.
        """
        # Ensure resolve is imported
        from temporal.registry.core import resolve
        cls = resolve("attention", attn_cfg.attention_type)
        return cls(
            embed_dim=self.config.hidden_size,
            num_heads=attn_cfg.num_heads,
            dropout=attn_cfg.dropout,
            **attn_cfg.kwargs,
        )

    def build_feedforward(self, ffn_cfg: FeedForwardConfig = None) -> nn.Module:
        """
        Build a feed-forward network module from its configuration.

        Args:
            ffn_cfg (FeedForwardConfig, optional): Configuration for the feed-forward layer.
                Defaults to the config.feedforward_config if None.

        Returns:
            nn.Module: Instantiated feed-forward network.
        """
        # Ensure resolve is imported
        from temporal.registry.core import resolve
        cfg = ffn_cfg or self.config.feedforward_config
        cls = resolve("feedforward", cfg.type)
        return cls(
            hidden_size=self.config.hidden_size,
            intermediate_size=cfg.intermediate_size,
            activation=cfg.activation,
            dropout=cfg.dropout,
            **cfg.kwargs,
        )

    def build_value_embedding(self) -> nn.Module:
        """
        Build the value embedding module for input features.

        Returns:
            nn.Module: Instantiated embedding layer for values.
        """
        # Ensure resolve is imported
        from temporal.registry.core import resolve
        cfg = self.config.value_embedding_config
        cls = resolve("embedding", cfg.type)
        return cls(
            feature_size=self.config.feature_size, 
            d_model=self.config.hidden_size,      
            **cfg.kwargs,
        )


    def build_positional_embedding(self) -> nn.Module:
        """
        Build the positional embedding module for time indices.

        Returns:
            nn.Module: Instantiated positional embedding layer.
        """
        # Ensure resolve is imported
        from temporal.registry.core import resolve
        cfg = self.config.positional_embedding_config
        cls = resolve("embedding", cfg.type)
        return cls(
            embedding_dim=self.config.hidden_size,
            **cfg.kwargs,
        )

    def build_head_aggregator(self) -> nn.Module:
        """
        Build a head aggregator to combine multiple output tokens.

        Returns:
            nn.Module: Instantiated head aggregation module.

        Raises:
            ValueError: If output_head_config.output_size is not set.
        """
        # Ensure resolve is imported
        from temporal.registry.core import resolve
        cfg = self.config.head_agg_config
        cls = resolve("head_agg", cfg.type)

        output_size = self.config.output_head_config.output_size
        if output_size is None:
            raise ValueError("output_head_config.output_size must be set to build head aggregator.")

        return cls(
            input_size=output_size,
            num_heads=self.config.output_token_lengths,
            output_size=output_size,
            **cfg.kwargs,
        )

    def build_normalization(self) -> nn.Module:
        """
        Build a normalization layer based on configuration.

        Returns:
            nn.Module: Instantiated normalization layer.
        """
        # Ensure resolve is imported
        from temporal.registry.core import resolve
        cfg = self.config.norm_config
        cls = resolve("normalization", cfg.norm_type)
        return cls(eps=cfg.eps, normalized_shape=self.config.hidden_size)
