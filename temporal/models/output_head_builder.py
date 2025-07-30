from temporal.registry.core import resolve
import inspect
from typing import Type
import torch.nn as nn
from temporal.models.module_builder_helper import ModuleBuilder # Import ModuleBuilder

class OutputHeadBuilder:
    """Constructs the output head module for a time series model.

    This builder is responsible for instantiating the correct output head based on
    the model's configuration. It resolves the head's class from the registry
    and prepares the necessary arguments for its constructor, such as the model's
    hidden dimension and the required output dimension, which can vary depending
    on the head type (e.g., for point forecasts vs. quantile forecasts).

    Attributes:
        config: The main configuration object for the model.
        builder: The main module builder helper.
    """
    def __init__(self, config, builder: ModuleBuilder):
        """Initializes the OutputHeadBuilder.

        Args:
            config: The main model configuration object, which should contain
                `output_head_config` and the model's hidden dimension (`d_model`).
            builder (ModuleBuilder): The main module builder helper, for potentially
                building sub-modules within complex heads.
        """
        self.config = config
        self.builder = builder # Store the builder

    def build(self) -> nn.Module:
        """Instantiates and returns the configured output head module.

        This method resolves the appropriate output head class from the registry
        and prepares its constructor arguments. It validates that the necessary
        configuration values (e.g., `num_quantiles` for a quantile head) are
        present.

        Returns:
            An instantiated nn.Module representing the output head.

        Raises:
            ValueError: If the configuration is missing necessary parameters
                for the selected head type.
        """
        head_config = self.config.output_head_config
        head_type = head_config.type
        head_class = resolve("output_head", head_type)
        head_input_dim = getattr(self.config, 'feature_size', 1) 

        hidden_size = getattr(self.config, 'd_model', getattr(self.config, 'hidden_size', None))
        if hidden_size is None:
            raise ValueError("Config must specify 'd_model' or 'hidden_size'.")

        # Determine the required output_size based on the head type.
        output_size = self._calculate_output_size(head_type, head_config) 

        if 'patch_size' in self.config.value_embedding_config.kwargs: 
          final_hidden_size_for_head = head_input_dim

        else: 
          final_hidden_size_for_head = hidden_size


        # Prepare arguments for the head's constructor.
        # We start with the essential ones and add others from the config's kwargs.
        init_args = {
            "hidden_size": final_hidden_size_for_head,
            "output_size": output_size,
            "num_quantiles": getattr(self.config, 'num_quantiles', None),
            "feature_size": getattr(self.config, 'feature_size', 1),
        }
        
        # Add all user-defined kwargs from the head_config
        if head_config.kwargs:
            init_args.update(head_config.kwargs)

        # Add explicitly defined fields from specific head configs
        # This handles fields like num_outputs, use_tanh, tanh_scale, components, etc.
        for f_name in head_config.__dataclass_fields__:
            if f_name not in ["type", "output_size", "kwargs"]:
                init_args[f_name] = getattr(head_config, f_name)


        # Filter the arguments to only those accepted by the head's constructor.
        signature = inspect.signature(head_class.__init__)
        accepted_params = set(signature.parameters.keys())
        
        # If the constructor accepts **kwargs, pass everything. Otherwise, filter.
        if "kwargs" not in accepted_params:
            final_args = {k: v for k, v in init_args.items() if k in accepted_params}
        else:
            final_args = init_args


        try:
            return head_class(**final_args)
        except TypeError as e:
            print(f"ERROR: Failed to instantiate output head '{head_type}' ({head_class.__name__}).")
            print(f"  > Constructor signature: {signature}")
            print(f"  > Provided arguments: {list(final_args.keys())}")
            raise e

    def _calculate_output_size(self, head_type: str, head_config) -> int:
        """Calculates the required output dimension for the head's projection layer."""
        feature_size = getattr(self.config, 'feature_size', 1)

        if head_type == "linear":
            return feature_size
        elif head_type == "gaussian":
            return feature_size * 2  # Mean and std dev
        elif head_type == "t_distribution":
            return feature_size * 3  # Mean, scale, and degrees of freedom
        elif head_type == "quantile_regression": # Use the direct config field
            if not hasattr(head_config, 'num_quantiles') or head_config.num_quantiles is None:
                raise ValueError(f"Head type '{head_type}' requires 'num_quantiles' to be set in its config.")
            return feature_size * head_config.num_quantiles
        elif head_type == "distpred": # Use the direct config field
            if not hasattr(head_config, 'num_outputs') or head_config.num_outputs is None:
                raise ValueError(f"Head type '{head_type}' requires 'num_outputs' to be set in its config.")
            return feature_size * head_config.num_outputs
        elif head_type == "mixture":
            # The MixtureOutputHead calculates its own output size internally based on components.
            # We need to instantiate a dummy head to get its total_params_dim.
            from temporal.modules.heads.output_heads import MixtureOutputHead # Local import to avoid circular
            if not hasattr(head_config, 'components') or not head_config.components:
                 raise ValueError(f"Head type '{head_type}' requires 'components' to be set in its config.")
            
            # Create a dummy instance to calculate output_size based on components
            # This is a bit of a hack, but ensures output_size is correct for validation
            # before the actual build.
            dummy_head = MixtureOutputHead(hidden_size=1, components=head_config.components) # hidden_size doesn't matter for param count
            return dummy_head.output_projection.out_features

        else:
            # For other custom heads, output_size must be set explicitly.
            output_size = getattr(head_config, 'output_size', None)
            if output_size is None:
                raise ValueError(
                    f"OutputHeadConfig for a custom head of type '{head_type}' "
                    f"must specify the 'output_size' directly."
                )
            return output_size
