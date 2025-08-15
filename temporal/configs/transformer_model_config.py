from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TypeVar, Type
import numpy as np

from transformers import PretrainedConfig

from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY
from temporal.configs.architecture_config import TransformerArchitectureConfig
from temporal.configs.attention_config import AttentionConfig, attention_config_from_dict
from temporal.configs.output_head_config import OutputHeadConfig, output_head_config_from_dict
from temporal.configs.transformer_block_config import TransformerBlockConfig, transformer_block_config_from_dict, DecoderBlockConfig, EncoderBlockConfig
from temporal.configs.embedding_config import EmbeddingConfig, embedding_config_from_dict
from temporal.configs.head_aggregation_config import HeadAggregationConfig, head_aggregation_config_from_dict
from temporal.configs.normalization_config import NormalizationConfig, normalization_config_from_dict
from temporal.configs.quantizer_config import QuantizerConfig, quantizer_config_from_dict
from temporal.configs.loss_config import LossConfig, loss_config_from_dict, PROBABILISTIC_LOSSES
from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig

T = TypeVar('T', bound='TransformerTimeSeriesConfig')

# FIX: Changed the registration name to match the model registry key.
@register_config_type("transformer")
@dataclass(frozen=True, kw_only=True)
class TransformerTimeSeriesConfig(BaseTimeSeriesConfig):
    """
    Configuration for a transformer-based time-series forecasting model.
    """
    # FIX: Changed the default type to match the model registry key.
    type: str = field(default="transformer")
    model_type: str = field(default="transformer") # Kept for consistency

    d_model: int = field(default=64)
    hidden_dropout_prob: float = field(default=0.1)
    max_position_embeddings: int = field(default=4096)
    architecture: TransformerArchitectureConfig = field(default_factory=TransformerArchitectureConfig)
    value_embedding_config: EmbeddingConfig = field(default_factory=lambda: embedding_config_from_dict({"type": "value"}))
    positional_embedding_config: EmbeddingConfig = field(default_factory=lambda: embedding_config_from_dict({"type": "sinusoidal"}))
    encoder_blocks: Optional[List[TransformerBlockConfig]] = field(default=None)
    decoder_blocks: Optional[List[TransformerBlockConfig]] = field(default=None)
    output_head_config: OutputHeadConfig = field(default_factory=lambda: output_head_config_from_dict({"type": "linear"}))
    
    layer_norm_config: NormalizationConfig = field(default_factory=lambda: normalization_config_from_dict({"type": "layer"}))
    instance_norm_config: Optional[NormalizationConfig] = field(default=None)

    head_agg_config: HeadAggregationConfig = field(default_factory=lambda: head_aggregation_config_from_dict({"type": "mean"}))
    quantizer_config: Optional[QuantizerConfig] = field(default=None)
    vocab_size: Optional[int] = field(default=None)
    decoder_start_token_id: Optional[int] = field(default=None)
    output_attentions: bool = field(default=False)
    output_hidden_states: bool = field(default=False)
    use_teacher_forcing: bool = field(default=True)
    aux_loss_weight: float = field(default=0.01)
    use_cache: bool = field(default=True)

    attention_blocks: Optional[Any] = field(default=None, repr=False, compare=False)
    feedforward_config: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.loss_config, dict):
            object.__setattr__(self, "loss_config", loss_config_from_dict(self.loss_config))
        if isinstance(self.architecture, dict):
            object.__setattr__(self, "architecture", TransformerArchitectureConfig.from_dict(self.architecture))
        
        # FIX: Ensure encoder/decoder blocks are lists if they are None. This resolves ValueErrors in the builder.
        if self.encoder_blocks is None:
            object.__setattr__(self, "encoder_blocks", [])
        if self.decoder_blocks is None:
            object.__setattr__(self, "decoder_blocks", [])
            
        super().__post_init__()
        
        if "input_dim" not in self.__dict__ and hasattr(self, 'feature_size'):
            object.__setattr__(self, "input_dim", self.feature_size)
        
        if self.attention_blocks is not None:
            print("Warning: `attention_blocks` config key is deprecated. Configure attention within `encoder_blocks`/`decoder_blocks`.")
        if self.feedforward_config is not None:
            print("Warning: `feedforward_config` key is deprecated. Configure FFN within `encoder_blocks`/`decoder_blocks`.")

        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if not (0.0 <= self.hidden_dropout_prob <= 1.0):
            raise ValueError("hidden_dropout_prob must be in [0, 1]")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be > 0")

        if not isinstance(self.architecture, TransformerArchitectureConfig):
            raise ValueError("architecture must be a TransformerArchitectureConfig instance.")
        if not isinstance(self.value_embedding_config, EmbeddingConfig):
            raise ValueError("value_embedding_config must be an EmbeddingConfig instance.")
        if not isinstance(self.positional_embedding_config, EmbeddingConfig):
            raise ValueError("positional_embedding_config must be an EmbeddingConfig instance.")
        if not all(isinstance(b, TransformerBlockConfig) for b in self.encoder_blocks):
            raise ValueError("All items in encoder_blocks must be TransformerBlockConfig instances.")
        if not all(isinstance(b, TransformerBlockConfig) for b in self.decoder_blocks):
            raise ValueError("All items in decoder_blocks must be TransformerBlockConfig instances.")
        if not isinstance(self.output_head_config, OutputHeadConfig):
            raise ValueError("output_head_config must be an OutputHeadConfig instance.")
        if not isinstance(self.layer_norm_config, NormalizationConfig):
            raise ValueError("layer_norm_config must be a NormalizationConfig instance.")
        if self.instance_norm_config is not None and not isinstance(self.instance_norm_config, NormalizationConfig):
            raise ValueError("instance_norm_config must be a NormalizationConfig instance or None.")
        if not isinstance(self.head_agg_config, HeadAggregationConfig):
            raise ValueError("head_agg_config must be a HeadAggregationConfig instance.")
        if self.quantizer_config is not None and not isinstance(self.quantizer_config, QuantizerConfig):
            raise ValueError("quantizer_config must be a QuantizerConfig instance or None.")
        if not isinstance(self.loss_config, LossConfig):
            raise ValueError("loss_config must be a LossConfig instance.")

        all_blocks = (self.encoder_blocks or []) + (self.decoder_blocks or [])
        for i, block_config in enumerate(all_blocks):
            if block_config.attention_config:
                heads = block_config.attention_config.num_heads
                if self.d_model % heads != 0:
                    raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({heads}) in self-attention of block {i}")
            if isinstance(block_config, DecoderBlockConfig) and block_config.cross_attention_config:
                cross_heads = block_config.cross_attention_config.num_heads
                if self.d_model % cross_heads != 0:
                    raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({cross_heads}) in cross-attention of block {i}")

        if self.quantizer_config and not self.vocab_size:
            print("Warning: Quantizer configured but vocab_size not set.")
        if self.architecture.layout != "encoder-only" and self.decoder_start_token_id is None and self.vocab_size is not None:
            print("Warning: Decoder model needs decoder_start_token_id if tokenized.")

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        if "norm_config" in data and "layer_norm_config" not in data:
            norm_type = data["norm_config"].get("type", "layer") if isinstance(data["norm_config"], dict) else "layer"
            if norm_type == "revin":
                data["instance_norm_config"] = data.pop("norm_config")
                if "layer_norm_config" not in data:
                    data["layer_norm_config"] = {"type": "layer"}
            else:
                data["layer_norm_config"] = data.pop("norm_config")
        
        if "attention_blocks" in data:
            print("Warning: `attention_blocks` config key is deprecated. Configure attention within `encoder_blocks`/`decoder_blocks`.")
            data.pop("attention_blocks")
        if "feedforward_config" in data:
            print("Warning: `feedforward_config` key is deprecated. Configure FFN within `encoder_blocks`/`decoder_blocks`.")
            data.pop("feedforward_config")

        if "hidden_size" in data and "d_model" not in data:
            data["d_model"] = data.pop("hidden_size")

        if "input_dim" not in data and "feature_size" in data:
            data["input_dim"] = data.pop("feature_size")
        elif "feature_size" in data and "input_dim" in data and data["feature_size"] != data["input_dim"]:
            print(f"Warning: Both 'input_dim' ({data['input_dim']}) and 'feature_size' ({data['feature_size']}) found in config dict. Preferring 'input_dim'. 'feature_size' will be ignored.")
            data.pop("feature_size")
        elif "feature_size" in data:
            data.pop("feature_size")

        if "architecture" in data and isinstance(data["architecture"], dict):
            data["architecture"] = TransformerArchitectureConfig.from_dict(data["architecture"])
        if "value_embedding_config" in data and isinstance(data["value_embedding_config"], dict):
            #data["value_embedding_config"] = embedding_config_from_dict(data["value_embedding_config"])
            feature_size = data.get("feature_size") or data.get("input_dim")
            data["value_embedding_config"] = embedding_config_from_dict(
                data["value_embedding_config"],
                feature_size=feature_size 
            )
        if "positional_embedding_config" in data and isinstance(data["positional_embedding_config"], dict):
            data["positional_embedding_config"] = embedding_config_from_dict(data["positional_embedding_config"])
        if "encoder_blocks" in data and isinstance(data["encoder_blocks"], list):
            data["encoder_blocks"] = [transformer_block_config_from_dict(b) if isinstance(b, dict) else b for b in data["encoder_blocks"]]
        if "decoder_blocks" in data and isinstance(data["decoder_blocks"], list):
            data["decoder_blocks"] = [transformer_block_config_from_dict(b) if isinstance(b, dict) else b for b in data["decoder_blocks"]]
        if "output_head_config" in data and isinstance(data["output_head_config"], dict):
            data["output_head_config"] = output_head_config_from_dict(data["output_head_config"])
        if "layer_norm_config" in data and isinstance(data["layer_norm_config"], dict):
            data["layer_norm_config"] = normalization_config_from_dict(data["layer_norm_config"])
        if "instance_norm_config" in data and data["instance_norm_config"] and isinstance(data["instance_norm_config"], dict):
            data["instance_norm_config"] = normalization_config_from_dict(data["instance_norm_config"])
        if "head_agg_config" in data and isinstance(data["head_agg_config"], dict):
            data["head_agg_config"] = head_aggregation_config_from_dict(data["head_agg_config"])
        if "quantizer_config" in data and isinstance(data["quantizer_config"], dict):
            data["quantizer_config"] = quantizer_config_from_dict(data["quantizer_config"])
        if "loss_config" in data and isinstance(data["loss_config"], dict):
            data["loss_config"] = loss_config_from_dict(data["loss_config"])
        
        return super().from_dict(data)