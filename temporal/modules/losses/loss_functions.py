import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.distributions import StudentT, LogNormal, NegativeBinomial, Normal, Categorical # Added for MixtureLoss


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



class QuantileLoss(nn.Module):
    """
    Quantile loss (a.k.a. Pinball loss) for probabilistic regression.
    
    Parameters:
    -----------
    quantile : float
        The quantile to predict, e.g. 0.1, 0.5 (median), 0.9.
    reduction : str
        One of "mean" (default), "sum", or "none".
    """
    def __init__(self, quantile: float, reduction: str = "mean"):
        super().__init__()
        assert 0 < quantile < 1, "Quantile must be between 0 and 1."
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        errors = labels - predictions
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss



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


import torch
import torch.nn as nn
import math

class SpreadPenalty(nn.Module):
    """
    Computes a penalty on the predicted quantile spread to regularize uncertainty.
    Supports asymmetric penalties (log or inverse), or symmetric log penalty around a target spread.
    """
    def __init__(
        self,
        penalty_type: str = 'log',
        epsilon: float = 1e-3,
        reduction: str = 'mean',
        target_spread: float = 1.0,
    ):
        """
        Args:
            penalty_type (str): 'log', 'inverse', or 'symmetric_log'.
            epsilon (float): Numerical stability constant.
            reduction (str): 'mean', 'sum', or 'none'.
            target_spread (float): Target spread used in 'symmetric_log' mode.
        """
        super().__init__()
        if penalty_type not in ['log', 'inverse', 'symmetric_log']:
            raise ValueError("penalty_type must be 'log', 'inverse', or 'symmetric_log'")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

        self.penalty_type = penalty_type
        self.epsilon = epsilon
        self.reduction = reduction
        self.target_spread = target_spread

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Tensor of quantile predictions, shape [B, T, Q].
                                  Assumes quantiles are sorted along Q.
        Returns:
            torch.Tensor: Scalar penalty (or tensor if reduction='none').
        """
        if preds.ndim < 3 or preds.shape[-1] < 2:
            raise ValueError(f"Expected preds shape [B, T, Q≥2], got: {preds.shape}")

        spread = preds[..., -1] - preds[..., 0]
        spread = torch.clamp(spread, min=0.0)

        if self.penalty_type == 'log':
            penalty = -torch.log(spread + self.epsilon)
        elif self.penalty_type == 'inverse':
            penalty = 1.0 / (spread + self.epsilon)
        elif self.penalty_type == 'symmetric_log':
            log_spread = torch.log(spread + self.epsilon)
            log_target = math.log(self.target_spread)
            penalty = (log_spread - log_target) ** 2
        else:
            raise RuntimeError(f"Invalid penalty_type '{self.penalty_type}'.")

        if self.reduction == 'mean':
            return penalty.mean()
        elif self.reduction == 'sum':
            return penalty.sum()
        return penalty



class MixtureLoss(nn.Module):
    def __init__(self, reduction="mean", min_df=2.0, fixed_sigma=1e-3):
        super().__init__()
        self.reduction = reduction
        self.min_df = min_df
        self.fixed_sigma = fixed_sigma

    def forward(self, preds: dict, targets: torch.Tensor, loss_mask: torch.Tensor = None):
        """
        Args:
            preds: dict of predicted parameters. It should contain:
                   - "mixture_logits": Tensor of shape [B, T, M] (M = number of components)
                   - "components": List[str] of distribution names for each component (must match keys below)
                   - Parameters for each distribution, e.g., "student_df", "student_mu", "student_scale".
                     These tensors are expected to have shape [B, T] or be broadcastable.
                     The OutputHead must ensure these keys are populated as expected by the loss.
            targets: Ground truth tensor of shape [B, T]
            loss_mask: Optional float or bool mask of shape [B, T]
        """
        logits = preds["mixture_logits"]  # [B, T, M]
        weights = F.softmax(logits, dim=-1)  # [B, T, M]

        log_probs = []
        target_for_dist = targets # Shape [B, T]

        for i, dist_name in enumerate(preds["components"]):
            if dist_name == "student_t":
                nu = F.softplus(preds["student_df"]) + self.min_df # Shape [B,T]
                mu = preds["student_mu"] # Shape [B,T]
                tau = F.softplus(preds["student_scale"]) # Shape [B,T]
                dist = StudentT(df=nu, loc=mu, scale=tau)
                log_prob = dist.log_prob(target_for_dist)
            elif dist_name == "log_normal":
                mu = preds["lognorm_mu"] # Shape [B,T]
                sigma = F.softplus(preds["lognorm_sigma"]) # Shape [B,T]
                dist = LogNormal(loc=mu, scale=sigma) # PyTorch LogNormal takes loc (mu) and scale (sigma)
                log_prob = dist.log_prob(torch.clamp(target_for_dist, min=1e-6))  # clamp for log domain
            elif dist_name == "neg_binomial":
                r = F.softplus(preds["nb_r"]) # Shape [B,T]
                p = torch.sigmoid(preds["nb_p"]) # Shape [B,T]
                dist = NegativeBinomial(total_count=r, probs=p)
                log_prob = dist.log_prob(target_for_dist)
            elif dist_name == "fixed_normal":
                mu = preds["normal_mu"] # Shape [B,T]
                sigma_val = torch.tensor(self.fixed_sigma, device=mu.device, dtype=mu.dtype)
                sigma = sigma_val.expand_as(mu) # Expand to [B,T]
                dist = Normal(loc=mu, scale=sigma)
                log_prob = dist.log_prob(target_for_dist)
            else:
                raise ValueError(f"Unknown distribution component: {dist_name}")

            log_probs.append(log_prob)  # Appending tensor of shape [B, T]

        log_probs_tensor = torch.stack(log_probs, dim=-1)  # Converts list of M tensors [B,T] to one tensor [B, T, M]
        
        # Log-sum-exp for stable mixture likelihood calculation
        # log P(y|θ) = log Σ_i w_i * P(y|θ_i) = log Σ_i exp(log w_i + log P(y|θ_i))
        # Add epsilon to weights before log to avoid log(0)
        log_weighted_log_prob = log_probs_tensor + torch.log(weights + 1e-9) 
        weighted_log_prob = torch.logsumexp(log_weighted_log_prob, dim=-1)  # Results in shape [B, T]
        
        nll = -weighted_log_prob  # Negative Log-Likelihood, shape [B, T]

        if loss_mask is not None:
            # Ensure mask is broadcastable if not identical shape, though typically [B,T] expected.
            if loss_mask.shape != nll.shape and loss_mask.ndim == nll.ndim:
                loss_mask = loss_mask.expand_as(nll) # Try to expand if dims match but sizes differ at singleton
            elif loss_mask.shape != nll.shape:
                 raise ValueError(f"loss_mask shape {loss_mask.shape} incompatible with nll shape {nll.shape}")
            
            nll = nll * loss_mask
            num_active_elements = loss_mask.sum().clamp(min=1e-9) 
        else:
            num_active_elements = torch.tensor(nll.numel(), device=nll.device, dtype=nll.dtype).clamp(min=1e-9)

        if self.reduction == "mean":
            return nll.sum() / num_active_elements
        elif self.reduction == "sum":
            return nll.sum()
        # reduction == "none" or other
        return nll
