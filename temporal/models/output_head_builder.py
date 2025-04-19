from temporal.registry.core import resolve


class OutputHeadBuilder:
    """
    Constructs the output head and corresponding loss function for time series models.

    Uses the registry to resolve the appropriate output head class based on configuration.

    Args:
        config: Configuration object containing an `output_head_config` attribute.
    """
    def __init__(self, config):
        self.config = config

    def build(self):
        """
        Instantiate the output head module and its loss function.

        Resolves the output head class via the 'output_head' registry key, validates
        that `output_size` is set, and retrieves the associated loss function.

        Returns:
            output_head (nn.Module): The instantiated output head module.
            loss_fn (callable): Loss function obtained from the output head.

        Raises:
            ValueError: If `output_head_config.output_size` is None.
        """
        cfg = self.config.output_head_config
        cls = resolve("output_head", cfg.type)

        output_size = cfg.output_size
        if output_size is None:
            raise ValueError("OutputHeadConfig must specify output_size explicitly.")

        output_head = cls(
            hidden_size=self.config.hidden_size,
            output_size=output_size,
            **cfg.kwargs,
        )
        loss_fn = output_head.get_loss_fn()

        return output_head, loss_fn
