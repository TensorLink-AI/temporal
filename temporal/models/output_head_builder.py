from temporal.registry.core import resolve

class OutputHeadBuilder:
    def __init__(self, config):
        self.config = config

    def build(self):
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
