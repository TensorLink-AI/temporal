class EnsembleSampler:
    def __init__(self, model, dropout_enabled=True):
        self.model = model
        if dropout_enabled:
            self._enable_dropout(model)

    def _enable_dropout(self, model):
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def generate(
        self,
        ensemble_size: int,
        **generate_kwargs
    ) -> torch.Tensor:
        preds = [
            self.model.generate(**generate_kwargs)  # could be AR, multistep, etc.
            for _ in range(ensemble_size)
        ]
        return torch.stack(preds, dim=1)  # [B, N, T, D]
