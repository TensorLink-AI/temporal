from temporal.registry.core import resolve
import inspect


class BlockBuilder:
    """
    Builder class for creating transformer blocks based on configuration.

    Uses a registry to resolve block classes and introspection to map
    configuration parameters safely to constructor arguments.

    Args:
        config: Global configuration object for the temporal model.
        builder: Reference to a higher-level builder that provides methods
            for constructing subcomponents (attention, feedforward, normalization).
    """
    def __init__(self, config, builder):
        self.config = config
        self.builder = builder

    def build_block(self, block_cfg):
        """
        Instantiate a block module from its configuration.

        Resolves the block class via registry, inspects its constructor to
        determine accepted parameters, and constructs init_kwargs by mapping
        available subcomponents (attention, ffn, normalization) and any
        additional block-specific kwargs.

        Args:
            block_cfg: An instance of TransformerBlockConfig containing
                attributes:
                    - block_type (str): Identifier for the block class.
                    - attention_config (AttentionConfig): Config for attention sublayer.
                    - ffn_config (FeedForwardConfig or None): Config for feed-forward sublayer.
                    - kwargs (dict): Additional keyword arguments for the block.

        Returns:
            An instantiated block class with appropriate submodules.
        """
        # Determine block class
        block_type = block_cfg.block_type
        block_cls = resolve("block", block_type)

        # Inspect constructor signature
        sig = inspect.signature(block_cls.__init__)
        accepted = set(sig.parameters.keys()) - {"self"}

        init_kwargs = {}

        # Map attention submodule if constructor accepts
        if "attention" in accepted:
            init_kwargs["attention"] = self.builder.build_attention(
                block_cfg.attention_config
            )

        # Map feed-forward submodule if accepted and configured
        if "ffn" in accepted and block_cfg.ffn_config is not None:
            init_kwargs["ffn"] = self.builder.build_feedforward(
                block_cfg.ffn_config
            )

        # Map normalization function if accepted
        if "norm_fn" in accepted:
            init_kwargs["norm_fn"] = self.builder.build_normalization

        # Pass through any extra kwargs
        if "kwargs" in accepted:
            init_kwargs["kwargs"] = block_cfg.kwargs
        else:
            init_kwargs.update(block_cfg.kwargs)

        return block_cls(**init_kwargs)
