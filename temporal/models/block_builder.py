from temporal.registry.core import resolve
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import FeedForwardConfig
from temporal.configs.transformer_block_config import (
    TransformerBlockConfig,
    EncoderBlockConfig,
    DecoderBlockConfig,
    AdaptivePatchTransformerBlockConfig # Added this import
)
from temporal.configs.normalization_config import NormalizationConfig # Ensure this is present
import inspect
from typing import Type
import torch.nn as nn # Need to import nn for type hinting nn.Module

# Assuming ModuleBuilder is imported or accessible, 
# it's usually defined in temporal.models.module_builder_helper
# For now, I'll assume module_builder_helper.ModuleBuilder is passed via constructor.


class BlockBuilder:
    """
    Builds various sub-modules required by encoder and decoder blocks.
    This class is intended to be used internally by the TimeSeriesTransformerEncoder
    and TimeSeriesTransformerDecoder modules.
    """
    def __init__(self, config, module_builder):
        self.config = config # This is the main TransformerTimeSeriesConfig
        self.module_builder = module_builder # This is an instance of ModuleBuilder (from module_builder_helper)

    def build_attention(self, attention_config: AttentionConfig) -> nn.Module:
        """
        Builds an attention module based on the provided AttentionConfig.
        This method is called from within the TransformerEncoderLayer and TransformerDecoderLayer.
        """
        # The module_builder.build_attention method is already designed to take the config
        # and handle the arguments needed by the attention module's __init__.
        return self.module_builder.build_attention(attention_config)

    def build_feedforward(self, ffn_config: FeedForwardConfig) -> nn.Module:
        """
        Builds a feed-forward network based on the provided FeedForwardConfig.
        """
        return self.module_builder.build_feedforward(ffn_config)

    def build_normalization(self, norm_config: NormalizationConfig) -> nn.Module:
        """
        Builds a normalization layer based on the provided NormalizationConfig.
        """
        return self.module_builder.build_normalization(norm_config)

    def build_block(self, block_config: TransformerBlockConfig) -> nn.Module:
        """
        Builds a transformer block (e.g., encoder layer, decoder layer, or special block)
        based on the provided TransformerBlockConfig.
        
        This method delegates to the main module_builder's _build method, 
        which handles the polymorphic instantiation and argument injection.
        """
        # The module_builder._build method is designed to take a BaseConfig
        # and extract its fields, inject common arguments (like d_model, builder),
        # and pass them to the module's __init__.
        # This covers all block types (Encoder, Decoder, AdaptivePatch)
        # as long as their configs match their module's __init__ signature.
        return self.module_builder._build(kind="block", module_config=block_config)
