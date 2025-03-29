import torch
import torch.nn as nn
import torch.nn.functional as F
from soft_dtw_pytorch import SoftDTW
import torch.fft



class MQLoss(nn.Module):
    """
    Multi-Quantile Loss (MQ Loss) for probabilistic forecasting.
    Optionally computes CRPS approximation.
    """

    def __init__(self, quantiles, reduction="mean", use_crps=False):
        """
        Args:
            quantiles (list or Tensor): List of quantiles (e.g., [0.1, 0.5, 0.9])
            reduction (str): 'mean', 'sum', or 'none'
            use_crps (bool): If True, approximate CRPS instead of basic MQ loss
        """
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles).float())
        self.reduction = reduction
        self.use_crps = use_crps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Tensor of shape (B, T, Q) — predicted quantiles
            target: Tensor of shape (B, T) — ground truth

        Returns:
            Scalar loss (or tensor if reduction='none')
        """
        q = self.quantiles.view(1, 1, -1)  # (1, 1, Q)
        y = target.unsqueeze(-1)          # (B, T, 1)
        e = y - preds                     # (B, T, Q)

        # Pinball loss
        loss = torch.max(q * e, (q - 1) * e)

        if self.use_crps:
            # CRPS approx: add pairwise quantile disagreement penalty
            preds_sorted, _ = torch.sort(preds, dim=-1)
            diff = preds_sorted[..., 1:] - preds_sorted[..., :-1]  # (B, T, Q-1)
            crps_term = torch.mean(diff ** 2, dim=-1)              # (B, T)
            loss = loss.sum(dim=-1) + crps_term                    # (B, T)
        else:
            loss = loss.sum(dim=-1)  # (B, T)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # (B, T)



class WeightedQuantileLoss(nn.Module):
    """
    Weighted Quantile Loss (wQL)

    This loss penalizes under- and over-predictions asymmetrically,
    depending on the quantile (τ). Automatically falls back to
    unweighted QL if denominator is too small (i.e., near-zero target sum).
    """

    def __init__(self, quantiles=(0.1, 0.5, 0.9), epsilon=1e-8, reduction='mean'):
        """
        Args:
            quantiles (tuple): Quantile levels (τ) in (0, 1)
            epsilon (float): Small number to avoid divide-by-zero
            reduction (str): 'mean' (default), 'sum', or 'none'
        """
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles).float())
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Tensor of shape (B, T, Q)
            target: Tensor of shape (B, T)

        Returns:
            Scalar loss (or tensor if reduction='none')
        """
        B, T, Q = preds.shape
        assert Q == len(self.quantiles), "Mismatch between preds and quantiles"

        target = target.unsqueeze(-1)               # (B, T, 1)
        errors = target - preds                     # (B, T, Q)
        taus = self.quantiles.view(1, 1, Q)         # (1, 1, Q)

        weighted_losses = torch.max(
            taus * errors, (taus - 1) * errors
        )  # pinball loss (B, T, Q)

        # Numerator: weighted quantile loss
        num = weighted_losses.sum(dim=(0, 1))  # (Q,)

        # Denominator: sum of absolute target values (per quantile)
        denom = torch.abs(target).sum(dim=(0, 1)) + self.epsilon  # (Q,)

        wql = num / denom  # (Q,)

        if self.reduction == "mean":
            return wql.mean()
        elif self.reduction == "sum":
            return wql.sum()
        return wql  # no reduction: returns one value per quantile

def QuantileLoss(self, predictions, labels):
    """Computes quantile loss for given quantile level."""
    errors = labels - predictions
    return torch.max((self.quantile - 1) * errors, self.quantile * errors)



class KernelEnergyLoss(nn.Module):
    """
    Kernel-based loss for probabilistic forecasting using Energy Distance.
    
    Accepts:
    - Predicted samples (B, T, N): N samples per forecast
    - Target values (B, T): observed values
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Forecast samples of shape (B, T, N)
            targets: Ground truth values of shape (B, T)
        Returns:
            Energy distance-based loss (scalar or per-sample)
        """
        B, T, N = preds.shape
        y = targets.unsqueeze(-1)  # (B, T, 1)

        # Pairwise distances between samples (symmetric term)
        diff_samples = preds.unsqueeze(-1) - preds.unsqueeze(-2)  # (B, T, N, N)
        sample_term = torch.mean(torch.abs(diff_samples), dim=(2, 3))  # (B, T)

        # Distance between samples and target (cross term)
        diff_target = preds - y  # (B, T, N)
        cross_term = torch.mean(torch.abs(diff_target), dim=2)  # (B, T)

        # Energy distance per sample
        energy = 2 * cross_term - sample_term  # (B, T)

        if self.reduction == "mean":
            return energy.mean()
        elif self.reduction == "sum":
            return energy.sum()
        return energy  # (B, T)




class EnergyDistanceLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        samples: (B, S, T)  - S samples per time series
        target:  (B, T)
        """
        B, S, T = samples.shape

        # ED term 1: ||s_i - y||_2
        ed1 = torch.norm(samples - target.unsqueeze(1), dim=-1).mean(dim=1)

        # ED term 2: ||s_i - s_j||_2 between all sample pairs
        pairwise_dists = torch.norm(samples.unsqueeze(2) - samples.unsqueeze(1), dim=-1)  # (B, S, S)
        ed2 = pairwise_dists.mean(dim=(1, 2))

        ed = 2 * ed1 - ed2  # Energy Distance per sample
        return ed.mean() if self.reduction == "mean" else ed.sum()




class SpectralLoss(nn.Module):
    """
    Spectral Loss based on the L2 distance between FFTs of predictions and targets.

    Encourages models to match frequency components.

    Args:
        reduction (str): 'mean' | 'sum' | 'none'
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, T)
            targets: (B, T)
        Returns:
            Loss (scalar or per batch)
        """
        fft_pred = torch.fft.rfft(preds, dim=1)
        fft_target = torch.fft.rfft(targets, dim=1)

        loss = torch.abs(fft_pred - fft_target) ** 2
        loss = loss.sum(dim=1)  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss




class FastSoftDTWLoss(nn.Module):
    """
    Differentiable Dynamic Time Warping (Soft-DTW) Loss (pure PyTorch, batched).
    
    Args:
        gamma (float): Smoothing parameter. Lower → closer to hard DTW.
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Tensor of shape (B, T) - predicted sequence
            targets: Tensor of shape (B, T) - ground truth sequence
        Returns:
            Soft-DTW loss (scalar if reduced, else shape [B])
        """
        B, T = preds.shape
        preds = preds.unsqueeze(2)  # (B, T, 1)
        targets = targets.unsqueeze(1)  # (B, 1, T)
        D = (preds - targets).pow(2)  # (B, T, T)

        R = torch.full((B, T + 1, T + 1), float("inf"), device=preds.device)
        R[:, 0, 0] = 0.0

        for i in range(1, T + 1):
            D_i = D[:, i - 1, :]
            for j in range(1, T + 1):
                r0 = R[:, i - 1, j - 1]
                r1 = R[:, i - 1, j]
                r2 = R[:, i, j - 1]
                r = torch.stack((r0, r1, r2), dim=-1)
                softmin = -self.gamma * torch.logsumexp(-r / self.gamma, dim=-1)
                R[:, i, j] = D_i[:, j - 1] + softmin

        loss = R[:, T, T]  # (B,)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
