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

        # --- Add config and builder if accepted --- 
        if "config" in accepted:
            init_kwargs["config"] = self.config
        if "builder" in accepted:
            init_kwargs["builder"] = self.builder
        # -----------------------------------------

        # Map attention submodule if constructor accepts
        # Note: The config/builder logic above handles the specific case for default blocks.
        # More generic blocks might accept 'attention' directly.
        if "attention" in accepted:
            # Ensure we don't overwrite if config/builder added it via specific logic
            if "attention" not in init_kwargs:
                # Check if the block_cfg *itself* has an attention_config
                # (As opposed to the global config passed to the builder)
                if hasattr(block_cfg, 'attention_config') and block_cfg.attention_config:
                    init_kwargs["attention"] = self.builder.build_attention(
                        block_cfg.attention_config
                    )
                # else: Maybe log a warning if attention is expected but no config provided?

        # Map feed-forward submodule if accepted and configured
        if "ffn" in accepted and block_cfg.ffn_config is not None:
             if "ffn" not in init_kwargs: # Avoid overwriting if already set by builder
                init_kwargs["ffn"] = self.builder.build_feedforward(
                    block_cfg.ffn_config
                )

        # Map normalization function if accepted
        if "norm_fn" in accepted:
            if "norm_fn" not in init_kwargs:
                init_kwargs["norm_fn"] = self.builder.build_normalization

        # Pass through any extra kwargs from the block_cfg
        if hasattr(block_cfg, 'kwargs') and block_cfg.kwargs:
            # Prioritize kwargs specifically accepted by the constructor
            ctor_kwargs = {k: v for k, v in block_cfg.kwargs.items() if k in accepted}
            init_kwargs.update(ctor_kwargs)

            # If the constructor accepts **kwargs, pass everything
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                 init_kwargs.update(block_cfg.kwargs) # Add potentially unlisted kwargs


        # --- Final instantiation --- 
        try:
            return block_cls(**init_kwargs)
        except TypeError as e:
            print(f"Error instantiating {block_cls.__name__} with kwargs: {init_kwargs}")
            print(f"Constructor signature: {sig}")
            raise e
