from temporal.modules.losses.loss_functions import (
    QuantileLoss,
    MQLoss,
    WeightedQuantileLoss,
    KernelEnergyLoss,
    EnergyDistanceLoss,
    SpectralLoss,
    FastSoftDTWLoss
)

class TimeSeriesLoss(BaseLoss):
    def __init__(
        self,
        loss_type: str = "mse",
        quantiles: list = [0.1, 0.5, 0.9],
        output_token_len: int = 1,
        reduction: str = "mean",
        **kwargs
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.quantiles = quantiles
        self.output_token_len = output_token_len

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == "rmse":
            self.loss_fn = lambda x, y: torch.sqrt(F.mse_loss(x, y, reduction=reduction))
        elif loss_type == "quantile":
            self.loss_fn = QuantileLoss(quantile=quantiles[0], reduction=reduction)
        elif loss_type == "mq":
            self.loss_fn = MQLoss(quantiles=quantiles, reduction=reduction)
        elif loss_type == "wql":
            self.loss_fn = WeightedQuantileLoss(quantiles=quantiles, reduction=reduction)
        elif loss_type == "kernel_energy":
            self.loss_fn = KernelEnergyLoss(reduction=reduction)
        elif loss_type == "energy":
            self.loss_fn = EnergyDistanceLoss(reduction=reduction)
        elif loss_type == "spectral":
            self.loss_fn = SpectralLoss(reduction=reduction)
        elif loss_type == "softdtw":
            gamma = kwargs.get("gamma", 1.0)
            self.loss_fn = FastSoftDTWLoss(gamma=gamma, reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
