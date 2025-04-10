import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from temporal.modules.attentions.base_attention import DotProductAttention
from temporal.modules.attentions.time_attention import TimeSeriesAttention
from temporal.modules.attentions.timer_attention import TimerAttention
from temporal.modules.attentions.time_attention import TimeSeriesAttention



ATTENTION_REGISTRY = {
    "std": DotProductAttention,
    'time' : TimeSeriesAttention,
    'timer': TimerAttention,
    'diff':
    # "custom": MyCustomAttention,
}

import inspect

class AutoTimeSeriesAttention:
    @staticmethod
    def from_config(
        config: BaseTimeSeriesConfig,
        attention_context: str = "self",  # could be "self", "cross", "encoder"
        **kwargs
    ):
        attention_type = getattr(config, "attention_type", "base")
        attention_cls = ATTENTION_REGISTRY.get(attention_type)

        if attention_cls is None:
            raise ValueError(f"Unknown attention_type '{attention_type}'. Available: {list(ATTENTION_REGISTRY.keys())}")

        # Extract only valid kwargs
        sig = inspect.signature(attention_cls.__init__)
        accepted_keys = set(sig.parameters.keys()) - {"self"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}

        # Add derived arguments
        is_decoder = (attention_context in ["self", "cross"])
        is_cross_attention = (attention_context == "cross")

        return attention_cls(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=True,
            **filtered_kwargs,
        )
        
