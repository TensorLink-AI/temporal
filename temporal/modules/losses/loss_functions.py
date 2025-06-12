import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.distributions import StudentT, LogNormal, NegativeBinomial, Normal
import math

class MQLoss(nn.Module):
    """Computes the Multi-Quantile Loss (MQL) for probabilistic forecasting.

    This loss function computes the average pinball loss over a set of specified
    quantiles. It is a common metric for evaluating the accuracy of quantile
    forecasts.

    Attributes:
        quantiles (torch.Tensor): The quantiles to be evaluated.
        reduction (str): The reduction method to apply to the final loss.
        use_crps (bool): If True, adds a penalty term to approximate the CRPS.
    """
    def __init__(self, quantiles: list, reduction: str = "mean", use_crps: bool = False):
        """Initializes the MQLoss module.

        Args:
            quantiles (list): A list of quantiles to evaluate (e.g., [0.1, 0.5, 0.9]).
            reduction (str): The reduction method: 'mean', 'sum', or 'none'.
            use_crps (bool): If True, approximates CRPS by adding a spread penalty.
        """
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles).float())
        self.reduction = reduction
        self.use_crps = use_crps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the Multi-Quantile Loss.

        Args:
            preds (torch.Tensor): The predicted quantiles, shape `(B, T, Q)`.
            target (torch.Tensor): The ground truth values, shape `(B, T)`.

        Returns:
            torch.Tensor: The computed loss, as a scalar or a tensor depending
            on the reduction method.
        """
        q = self.quantiles.view(1, 1, -1)
        y = target.unsqueeze(-1)
        e = y - preds

        loss = torch.max(q * e, (q - 1) * e)

        if self.use_crps:
            preds_sorted, _ = torch.sort(preds, dim=-1)
            diff = preds_sorted[..., 1:] - preds_sorted[..., :-1]
            crps_term = torch.mean(diff ** 2, dim=-1)
            loss = loss.sum(dim=-1) + crps_term
        else:
            loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedQuantileLoss(nn.Module):
    """Computes the Weighted Quantile Loss (wQL).

    wQL is a variant of the quantile loss that is normalized by the sum of the
    absolute target values. This can be useful for stabilizing training when
    the target values have a large dynamic range.

    Attributes:
        quantiles (torch.Tensor): The quantiles to be evaluated.
        epsilon (float): A small constant to prevent division by zero.
        reduction (str): The reduction method.
    """
    def __init__(self, quantiles: tuple = (0.1, 0.5, 0.9), epsilon: float = 1e-8, reduction: str = 'mean'):
        """Initializes the WeightedQuantileLoss module.

        Args:
            quantiles (tuple): The quantile levels (τ) to evaluate, between 0 and 1.
            epsilon (float): A small value to add to the denominator for stability.
            reduction (str): The reduction method: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles).float())
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the Weighted Quantile Loss.

        Args:
            preds (torch.Tensor): Predicted quantiles, shape `(B, T, Q)`.
            target (torch.Tensor): Ground truth values, shape `(B, T)`.

        Returns:
            torch.Tensor: The computed loss.
        """
        B, T, Q = preds.shape
        if Q != len(self.quantiles):
            raise ValueError("Mismatch between number of predicted quantiles and configured quantiles.")

        target = target.unsqueeze(-1)
        errors = target - preds
        taus = self.quantiles.view(1, 1, Q)

        weighted_losses = torch.max(
            taus * errors, (taus - 1) * errors
        )

        num = weighted_losses.sum(dim=(0, 1))
        denom = torch.abs(target).sum(dim=(0, 1)) + self.epsilon
        wql = num / denom

        if self.reduction == "mean":
            return wql.mean()
        elif self.reduction == "sum":
            return wql.sum()
        return wql


class QuantileLoss(nn.Module):
    """Computes the Quantile Loss (also known as Pinball Loss).

    This loss function is used for quantile regression. It asymmetrically
    penalizes over- and under-prediction to encourage the model to output a
    specific quantile of the target distribution.

    Attributes:
        quantile (float): The target quantile, between 0 and 1.
        reduction (str): The reduction method.
    """
    def __init__(self, quantile: float, reduction: str = "mean"):
        """Initializes the QuantileLoss module.
        """
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1.")
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates the Quantile Loss.
        """
        errors = labels - predictions
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class KernelEnergyLoss(nn.Module):
    """Computes a kernel-based energy distance loss for probabilistic forecasts.

    This loss function is a proper scoring rule that encourages the distribution
    of predicted samples to match the distribution of the true data. It is
    based on the energy distance between the two distributions.
    """
    def __init__(self, reduction: str = "mean"):
        """Initializes the KernelEnergyLoss module.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the kernel-based energy loss.

        Args:
            preds (torch.Tensor): A tensor of predicted samples, shape `(B, T, N)`.
            targets (torch.Tensor): The ground truth values, shape `(B, T)`.

        Returns:
            torch.Tensor: The computed loss.
        """
        B, T, N = preds.shape
        y = targets.unsqueeze(-1)

        diff_samples = preds.unsqueeze(-1) - preds.unsqueeze(-2)
        sample_term = torch.mean(torch.abs(diff_samples), dim=(2, 3))

        diff_target = preds - y
        cross_term = torch.mean(torch.abs(diff_target), dim=2)

        energy = 2 * cross_term - sample_term

        if self.reduction == "mean":
            return energy.mean()
        elif self.reduction == "sum":
            return energy.sum()
        return energy


class EnergyDistanceLoss(nn.Module):
    """Computes the Energy Distance loss.

    This is another implementation of the energy distance, often used for
    evaluating the similarity of two distributions.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the Energy Distance.
        """
        B, S, T = samples.shape
        ed1 = torch.norm(samples - target.unsqueeze(1), dim=-1).mean(dim=1)
        pairwise_dists = torch.norm(samples.unsqueeze(2) - samples.unsqueeze(1), dim=-1)
        ed2 = pairwise_dists.mean(dim=(1, 2))
        ed = 2 * ed1 - ed2
        return ed.mean() if self.reduction == "mean" else ed.sum()


class SpectralLoss(nn.Module):
    """Computes a loss in the frequency domain.

    This loss function calculates the L2 distance between the Fast Fourier
    Transforms (FFTs) of the predictions and the targets. It encourages the
    model to match the frequency components of the target sequence.
    """
    def __init__(self, reduction: str = "mean"):
        """Initializes the SpectralLoss module.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the spectral loss.
        """
        fft_pred = torch.fft.rfft(preds, dim=1)
        fft_target = torch.fft.rfft(targets, dim=1)
        loss = torch.abs(fft_pred - fft_target) ** 2
        loss = loss.sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FastSoftDTWLoss(nn.Module):
    """Computes a differentiable approximation of Dynamic Time Warping (DTW).

    Soft-DTW is a differentiable loss function that measures the alignment
    between two time series. It can be useful for tasks where the sequences
    might be out of phase.

    Attributes:
        gamma (float): The smoothing parameter. A lower gamma makes the loss
            closer to the non-differentiable DTW.
    """
    def __init__(self, gamma: float = 1.0, reduction: str = "mean"):
        """Initializes the FastSoftDTWLoss module.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Soft-DTW loss.
        """
        B, T = preds.shape
        preds = preds.unsqueeze(2)
        targets = targets.unsqueeze(1)
        D = (preds - targets).pow(2)
        R = torch.full((B, T + 1, T + 1), float("inf"), device=preds.device)
        R[:, 0, 0] = 0.0

        for i in range(1, T + 1):
            D_i = D[:, i - 1, :]
            for j in range(1, T + 1):
                r0, r1, r2 = R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]
                r = torch.stack((r0, r1, r2), dim=-1)
                softmin = -self.gamma * torch.logsumexp(-r / self.gamma, dim=-1)
                R[:, i, j] = D_i[:, j - 1] + softmin

        loss = R[:, T, T]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SpreadPenalty(nn.Module):
    """Computes a penalty on the spread of predicted quantiles.

    This module is used to regularize the uncertainty of a probabilistic
    forecast. It can penalize excessively wide or narrow quantile ranges.
    """
    def __init__(
        self,
        penalty_type: str = 'log',
        epsilon: float = 1e-3,
        reduction: str = 'mean',
        target_spread: float = 0.0,
    ):
        """Initializes the SpreadPenalty module.

        Args:
            penalty_type (str): The type of penalty function to use ('log', 'inverse', 'symmetric_log').
            epsilon (float): A small constant for numerical stability.
            reduction (str): The reduction method for the final penalty.
            target_spread (float): The target spread value, used only for the
                'symmetric_log' penalty type.
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
        """Calculates the spread penalty.

        Args:
            preds (torch.Tensor): A tensor of quantile predictions, assumed to be
                sorted along the last dimension, shape `[B, T, Q]`.

        Returns:
            torch.Tensor: The computed penalty.
        """
        if preds.ndim < 3 or preds.shape[-1] < 2:
            raise ValueError(f"Expected preds shape [B, T, Q>=2], got: {preds.shape}")

        spread = preds[..., -1] - preds[..., 0]
        spread = torch.clamp(spread, min=0.0)

        if self.penalty_type == 'log':
            penalty = -torch.log(spread + self.epsilon)
        elif self.penalty_type == 'inverse':
            penalty = 1.0 / (spread + self.epsilon)
        elif self.penalty_type == 'symmetric_log':
            log_spread = torch.log(spread + self.epsilon)
            log_target = math.log(self.target_spread + self.epsilon if self.target_spread == 0 else self.target_spread)
            penalty = (log_spread - log_target) ** 2
        else:
            raise RuntimeError(f"Invalid penalty_type '{self.penalty_type}'.")

        if self.reduction == 'mean':
            return penalty.mean()
        elif self.reduction == 'sum':
            return penalty.sum()
        return penalty


class MixtureLoss(nn.Module):
    """Computes the Negative Log-Likelihood for a Mixture Density Network.

    This loss function is designed to work with the output of a
    `MixtureOutputHead`. It calculates the likelihood of the target values
    under a mixture of probability distributions, whose parameters are
    predicted by the model.

    Attributes:
        min_df (float): The minimum degrees of freedom for a Student's T distribution.
        fixed_sigma (float): The fixed standard deviation for a Normal distribution
            if its sigma is not predicted.
    """
    def __init__(self, reduction="mean", min_df=2.0, fixed_sigma=1e-3):
        """Initializes the MixtureLoss module.

        Args:
            reduction (str): The reduction method for the final loss.
            min_df (float): A minimum value for the degrees of freedom of the
                Student's T distribution to ensure stability.
            fixed_sigma (float): The standard deviation to use for a
                `fixed_normal` component.
        """
        super().__init__()
        self.reduction = reduction
        self.min_df = min_df
        self.fixed_sigma = fixed_sigma

    def forward(self, preds: dict, targets: torch.Tensor, loss_mask: torch.Tensor = None):
        """Calculates the mixture loss.

        Args:
            preds (dict): A dictionary of predicted parameters from a `MixtureOutputHead`.
            targets (torch.Tensor): The ground truth values.
            loss_mask (Optional[torch.Tensor]): An optional mask for the loss.

        Returns:
            torch.Tensor: The final computed loss.
        """
        logits = preds["mixture_logits"]
        weights = F.softmax(logits, dim=-1)
        log_probs = []
        target_for_dist = targets

        for i, dist_name in enumerate(preds["components"]):
            if dist_name == "student_t":
                nu = F.softplus(preds["student_df"]) + self.min_df
                mu = preds["student_mu"]
                tau = F.softplus(preds["student_scale"])
                dist = StudentT(df=nu, loc=mu, scale=tau)
                log_prob = dist.log_prob(target_for_dist)
            elif dist_name == "log_normal":
                mu = preds["lognorm_mu"]
                sigma = F.softplus(preds["lognorm_sigma"])
                dist = LogNormal(loc=mu, scale=sigma)
                log_prob = dist.log_prob(torch.clamp(target_for_dist, min=1e-6))
            elif dist_name == "neg_binomial":
                r = F.softplus(preds["nb_r"])
                p = torch.sigmoid(preds["nb_p"])
                dist = NegativeBinomial(total_count=r, probs=p)
                log_prob = dist.log_prob(target_for_dist)
            elif dist_name == "fixed_normal":
                mu = preds["normal_mu"]
                sigma_val = torch.tensor(self.fixed_sigma, device=mu.device, dtype=mu.dtype)
                sigma = sigma_val.expand_as(mu)
                dist = Normal(loc=mu, scale=sigma)
                log_prob = dist.log_prob(target_for_dist)
            else:
                raise ValueError(f"Unknown distribution component: {dist_name}")
            log_probs.append(log_prob)

        log_probs_tensor = torch.stack(log_probs, dim=-1)
        log_weighted_log_prob = log_probs_tensor + torch.log(weights + 1e-9)
        weighted_log_prob = torch.logsumexp(log_weighted_log_prob, dim=-1)
        nll = -weighted_log_prob

        if loss_mask is not None:
            if loss_mask.shape != nll.shape and loss_mask.ndim == nll.ndim:
                loss_mask = loss_mask.expand_as(nll)
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
        return nll
