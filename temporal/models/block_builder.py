
from temporal.registry.core import resolve
from temporal.configs.attention_config import AttentionConfig # Import specific AttentionConfig
from temporal.configs.feedforward_config import FeedForwardConfig # Import specific FeedForwardConfig
from temporal.configs.transformer_block_config import TransformerBlockConfig # Import specific TransformerBlockConfig
from temporal.configs.normalization_config import NormalizationConfig # <-- ADDED THIS IMPORT
import inspect
from typing import Type


class BlockBuilder:
    """
    Builds various sub-modules required by encoder and decoder blocks.
    This class is intended to be used internally by the TimeSeriesTransformerEncoder
    and TimeSeriesTransformerDecoder modules.
    """
    def __init__(self, config, module_builder):
        self.config = config
        self.module_builder = module_builder

    def build_attention(self, attention_config: AttentionConfig, is_cross_attention: bool = False) -> AttentionConfig:
        """
        Builds an attention module based on the provided AttentionConfig.
        This method is called from within the TransformerEncoderLayer and TransformerDecoderLayer.
        """
        # Pass the attention_config dataclass directly to the module_builder
        return self.module_builder.build_attention(attention_config)

    def build_feedforward(self, ffn_config: FeedForwardConfig) -> FeedForwardConfig:
        """
        Builds a feed-forward network based on the provided FeedForwardConfig.
        """
        # Pass the ffn_config dataclass directly to the module_builder
        return self.module_builder.build_feedforward(ffn_config)

    def build_normalization(self, norm_config: NormalizationConfig) -> NormalizationConfig:
        """
        Builds a normalization layer based on the provided NormalizationConfig.
        """
        # Pass the norm_config dataclass directly to the module_builder
        return self.module_builder.build_normalization(norm_config)
