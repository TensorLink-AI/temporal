from temporal.registry.core import resolve


class BlockBuilder:
    def __init__(self, config, module_builder):
        self.config = config
        self.builder = module_builder

    def build_block(self, block_cfg):
        block_type = block_cfg.block_type
        block_cls = resolve("block", block_type)

        # Use dynamic dispatch based on class signature
        init_kwargs = {}

        if "attention" in block_cls.__init__.__code__.co_varnames:
            init_kwargs["attention"] = self.builder.build_attention(block_cfg.attention_config)

        if "ffn" in block_cls.__init__.__code__.co_varnames and block_cfg.ffn_config:
            init_kwargs["ffn"] = self.builder.build_feedforward()

        if "norm_fn" in block_cls.__init__.__code__.co_varnames:
            init_kwargs["norm_fn"] = self.builder.build_normalization

        if "config" in block_cls.__init__.__code__.co_varnames:
            init_kwargs["config"] = self.config

        # Add kwargs from config
        init_kwargs.update(block_cfg.kwargs or {})

        return block_cls(**init_kwargs)
