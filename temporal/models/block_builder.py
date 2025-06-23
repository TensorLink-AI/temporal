
from temporal.registry.core import resolve
from temporal.configs.transformer_config import AttentionConfig, FeedForwardConfig, TransformerBlockConfig
import inspect
from typing import Type
import torch.nn as nn

class BlockBuilder:
    """
    A builder class for creating Transformer blocks based on configuration.

    This class uses the central registry to resolve block classes (e.g., an
    encoder layer, a decoder layer, or a custom block) and intelligently
    constructs them by inspecting their `__init__` signature. This allows for
    flexible block designs that can accept either high-level configurations
    or pre-built sub-modules.

    Attributes:
        config: The global configuration object for the entire model.
        builder (ModuleBuilder): A reference to the main `ModuleBuilder` which
            can construct primitive components like attention or feed-forward layers.
    """
    def __init__(self, config, builder):
        """Initializes the BlockBuilder.

        Args:
            config: The main configuration object for the temporal model.
            builder: An instance of `ModuleBuilder` that provides methods
                for constructing sub-components (e.g., attention, feed-forward).
        """
        self.config = config
        self.builder = builder

    def build_block(self, block_cfg: TransformerBlockConfig) -> Type[nn.Module]:
        """
        Instantiates a single Transformer block from its specific configuration.

        This method performs the following steps:
        1.  Resolves the `block_type` from the block's configuration using the
            module registry to get the correct class.
        2.  Inspects the constructor of the resolved class to see what arguments it accepts.
        3.  Populates the arguments for the constructor. If the block's constructor
            accepts `config` and/or `builder`, this method passes the relevant
            objects. It can also build and pass primitive sub-modules (like
            attention or FFN) if the constructor is designed to accept them directly.
        4.  Passes any additional keyword arguments from the block configuration
            that match the constructor's signature, recursively building any nested
            configuration objects (e.g., AttentionConfig).
        5.  Instantiates and returns the block.

        Args:
            block_cfg: An instance of `TransformerBlockConfig` that defines the
                specific block to be built.

        Returns:
            An instantiated nn.Module representing the configured block.
        """
        block_type = block_cfg.block_type
        block_cls = resolve("block", block_type)

        # Inspect the block's constructor signature to see what it accepts.
        signature = inspect.signature(block_cls.__init__)
        accepted_params = set(signature.parameters.keys())

        init_kwargs = {}

        # If the block's constructor accepts 'config', pass it the specific
        # configuration for this block (`block_cfg`).
        if "config" in accepted_params:
            init_kwargs["config"] = block_cfg

        # If the block's constructor accepts 'builder', pass it the main
        # ModuleBuilder instance so it can build its own sub-components.
        if "builder" in accepted_params:
            init_kwargs["builder"] = self.builder

        # --- Direct Sub-module Injection (for blocks that don't use a builder) ---
        # If the block accepts 'attention' directly, build it.
        if "attention" in accepted_params:
            if hasattr(block_cfg, 'attention_config') and block_cfg.attention_config:
                init_kwargs["attention"] = self.builder.build_attention(
                    block_cfg.attention_config
                )

        # If the block accepts 'ffn' directly, build it.
        if "ffn" in accepted_params and hasattr(block_cfg, 'ffn_config') and block_cfg.ffn_config:
            init_kwargs["ffn"] = self.builder.build_feedforward(
                block_cfg.ffn_config
            )

        # Pass through any other keyword arguments from the block config
        # that are explicitly accepted by the constructor.
        if hasattr(block_cfg, 'kwargs') and block_cfg.kwargs:
            for key, value in block_cfg.kwargs.items():
                if key in accepted_params:
                    # RECURSIVE BUILD STEP: If a kwarg's value is a config object, build it.
                    if isinstance(value, AttentionConfig):
                        init_kwargs[key] = self.builder.build_attention(value)
                    elif isinstance(value, FeedForwardConfig):
                         init_kwargs[key] = self.builder.build_feedforward(value)
                    else:
                        # Otherwise, pass the value directly (e.g., for strings, ints).
                        init_kwargs[key] = value

        try:
            return block_cls(**init_kwargs)
        except TypeError as e:
            print(f"ERROR: Failed to instantiate block '{block_type}' ({block_cls.__name__}).")
            print(f"  > Constructor signature: {signature}")
            print(f"  > Provided arguments: {list(init_kwargs.keys())}")
            # Help debug by listing missing vs. unexpected arguments.
            required_params = {
                p.name for p in signature.parameters.values()
                if p.default == inspect.Parameter.empty and p.name not in init_kwargs and p.name != 'self'
            }
            if required_params:
                print(f"  > Missing required arguments: {list(required_params)}")
            raise e
