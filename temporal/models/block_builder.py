from temporal.registry.core import resolve
import inspect
from temporal.registry.core import resolve


class BlockBuilder:
    def __init__(self, config, builder):
        self.config = config
        self.builder = builder

    def build_block(self, block_cfg):
        block_type = block_cfg.block_type
        block_cls = resolve("block", block_type)

        # Introspect constructor to safely map arguments
        sig = inspect.signature(block_cls.__init__)
        accepted = set(sig.parameters.keys()) - {"self"}

        init_kwargs = {}

        # === Attention
        if "attention" in accepted:
            init_kwargs["attention"] = self.builder.build_attention(block_cfg.attention_config)

        # === FFN
        if "ffn" in accepted and block_cfg.ffn_config is not None:
            init_kwargs["ffn"] = self.builder.build_feedforward(block_cfg.ffn_config)

        # === Norm
        if "norm_fn" in accepted:
            init_kwargs["norm_fn"] = self.builder.build_normalization

        # === Pass extra block-specific kwargs (e.g., for EffiTime, Hybrid, etc.)
        if "kwargs" in accepted:
            init_kwargs["kwargs"] = block_cfg.kwargs
        else:
            init_kwargs.update(block_cfg.kwargs)

        return block_cls(**init_kwargs)

