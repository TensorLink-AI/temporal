from temporal.registry.core import resolve
import inspect
from typing import Type
import torch.nn as nn

class OutputHeadBuilder:
    """Constructs the output head module for a time series model.

    This builder is responsible for instantiating the correct output head based on
    the model's configuration. It resolves the head's class from the registry
    and prepares the necessary arguments for its constructor, such as the model's
    hidden dimension and the required output dimension, which can vary depending
    on the head type (e.g., for point forecasts vs. quantile forecasts).

    Attributes:
        config: The main configuration object for the model.
    """
    def __init__(self, config):
        """Initializes the OutputHeadBuilder.

        Args:
            config: The main model configuration object, which should contain
                `output_head_config` and the model's hidden dimension (`d_model`).
        """
        self.config = config

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

        hidden_size = getattr(self.config, 'd_model', getattr(self.config, 'hidden_size', None))
        if hidden_size is None:
            raise ValueError("Config must specify 'd_model' or 'hidden_size'.")

        # Determine the required output_size based on the head type.
        output_size = self._calculate_output_size(head_type, head_config)

        # Prepare arguments for the head's constructor.
        # We start with the essential ones and add others from the config's kwargs.
        init_args = {
            "hidden_size": hidden_size,
            "output_size": output_size,
            "num_quantiles": getattr(self.config, 'num_quantiles', None),
            "feature_size": getattr(self.config, 'feature_size', 1),
            **(head_config.kwargs or {})
        }

        # Filter the arguments to only those accepted by the head's constructor.
        signature = inspect.signature(head_class.__init__)
        accepted_params = set(signature.parameters.keys())
        final_args = {k: v for k, v in init_args.items() if k in accepted_params}

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
        elif head_type in ("quantile_regression", "distpred"):
            num_quantiles = getattr(self.config, 'num_quantiles', None)
            num_outputs = head_config.kwargs.get('num_outputs', num_quantiles)
            if num_outputs is None:
                raise ValueError(
                    f"Head type '{head_type}' requires 'num_quantiles' in the main "
                    f"config or 'num_outputs' in the head's kwargs."
                )
            return feature_size * num_outputs
        elif head_type == "mixture":
            # The MixtureOutputHead calculates its own output size internally,
            # so we can rely on its 'output_size' kwarg if provided for validation.
            if head_config.kwargs.get("output_size") is not None:
                return head_config.kwargs["output_size"]
            # If not, it will be derived inside the head itself. We pass a dummy
            # value which will be ignored if not in the signature.
            return -1 # Placeholder, as the head itself calculates this.
        else:
            # For other custom heads, output_size must be set explicitly.
            output_size = getattr(head_config, 'output_size', None)
            if output_size is None:
                raise ValueError(
                    f"OutputHeadConfig for a custom head of type '{head_type}' "
                    f"must specify the 'output_size' directly."
                )
            return output_size
