from temporal.registry.core import resolve
from temporal.configs.subconfigs import OutputHeadConfig


class OutputHeadBuilder:
    def __init__(self, config):
        self.config = config

    def build(self):
        cfg: OutputHeadConfig = self.config.output_head_config
        cls = resolve("output_head", cfg.type)

        # Instantiate the head using config.hidden_size and other kwargs
        output_head = cls(
            hidden_size=self.config.hidden_size,
            output_size=self.config.num_quantiles,  # for quantile, linear, etc.
            **cfg.kwargs
        )

        # Get the matching loss function
        loss_fn = output_head.get_loss_fn()

        return output_head, loss_fn
