from temporal.registry.core import resolve

class OutputHeadBuilder:
    """
    Constructs the output head module for time series models.

    Uses the registry to resolve the appropriate output head class based on configuration.
    The responsibility for building the loss function is moved to the main model builder,
    which should use `config.loss_config`.

    Args:
        config: Configuration object containing `output_head_config`, `d_model` (or `hidden_size`).
    """
    def __init__(self, config):
        self.config = config

    def build(self):
        """
        Instantiate the output head module.

        Resolves the output head class via the 'output_head' registry key and validates
        that `output_size` (or necessary components like num_quantiles/feature_size) is set
        in the config.

        Returns:
            output_head (nn.Module): The instantiated output head module.

        Raises:
            ValueError: If necessary configuration for the head is missing.
        """
        cfg = self.config.output_head_config
        head_type = cfg.type
        cls = resolve("output_head", head_type)

        # Determine required args for the head's __init__
        # Most heads need hidden_size and output_size.
        # Specific heads might need others (e.g., num_quantiles, feature_size)
        # passed via cfg.kwargs

        # --- Configuration Validation --- 
        # Use d_model if available, otherwise fall back to hidden_size for compatibility
        hidden_size = getattr(self.config, 'd_model', getattr(self.config, 'hidden_size', None))
        if hidden_size is None:
            raise ValueError("Config must specify d_model (or hidden_size)." )  

        # Define expected output_size based on head type and other config params
        output_size = None
        feature_size = getattr(self.config, 'feature_size', 1) # Default to 1 if not set
        num_quantiles = getattr(self.config, 'num_quantiles', None)

        if head_type == "linear":
            output_size = feature_size
        elif head_type == "gaussian":
            output_size = feature_size * 2
        elif head_type == "t_distribution":
             output_size = feature_size * 3 # Assuming mu, sigma, nu per feature
        elif head_type == "quantile_regression" or head_type == "distpred":
            if num_quantiles is None:
                raise ValueError(f"Head type '{head_type}' requires config.num_quantiles to be set.")
            output_size = feature_size * num_quantiles
        else:
             # For unknown or custom heads, rely on output_size being explicitly set in cfg
             output_size = cfg.output_size
             if output_size is None:
                  raise ValueError(f"OutputHeadConfig for type '{head_type}' must specify output_size explicitly.")

        # --- Instantiate Head --- 
        # Prepare arguments for the head constructor
        init_args = {
             "hidden_size": hidden_size,
             "output_size": output_size,
             # Pass necessary calculated values if the head needs them (like num_quantiles)
             "num_quantiles": num_quantiles, 
             "feature_size": feature_size,
         }
        # Add any specific kwargs from the config, potentially overriding calculated ones if needed
        init_args.update(cfg.kwargs or {}) 
        
        # Filter args to only those accepted by the specific head class __init__
        # This requires inspecting the class signature or simply trying and letting TypeError occur
        # Simpler approach: Pass all potentially relevant args and let the head handle them
        # (assuming heads use **kwargs to catch extras)
        
        # Refined approach: Pass standard args + specific kwargs from config
        final_args = {
            "hidden_size": hidden_size,
            "output_size": output_size,
            **cfg.kwargs, # Pass kwargs directly as DistPredHead now expects num_outputs in kwargs
        }

        output_head = cls(**final_args)

        # --- Loss Function Handling (REMOVED) ---
        # The loss function should be built by the main model builder using config.loss_config
        # loss_fn = output_head.get_loss_fn() # No longer call this

        return output_head # Return only the head
