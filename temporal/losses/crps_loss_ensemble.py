import torch

def crps_ensemble(
    observations: torch.Tensor,
    forecasts: torch.Tensor,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
    reduce: bool = True,  # <-- Added reduce flag
) -> torch.Tensor:
    r"""
    Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble
    of forecasts. This code is adapted from DistPred-Forecast.

    Parameters
    ----------
    observations : torch.Tensor
        The observed values. Shape can be [...], matching all but the ensemble dimension.
    forecasts : torch.Tensor
        The ensemble predictions. Must have the same shape as `observations` except
        for an additional ensemble dimension. By default, the ensemble dimension is
        assumed to be the last axis (axis = -1).
    axis : int
        The index of the ensemble dimension in `forecasts`. Defaults to -1 (last axis).
    sorted_ensemble : bool
        If True, indicates `forecasts` is already sorted in ascending order along the
        ensemble dimension. Some estimators (e.g. 'pwm') assume sorted data, so we
        automatically sort if needed and if estimator is 'pwm' or 'nrg'.
    estimator : str, optional
        Which CRPS estimator to use. Choose from:
          - "nrg": energy (a.k.a. 'nrg') form
          - "pwm": probability weighted moments form
          - "fair": the 'fair CRPS' variant
        Default is "pwm".
    reduce : bool, optional
        If True (default), returns the mean CRPS across the entire batch. If False,
        returns the CRPS for each element/timestep in the batch (shape [...]).

    Returns
    -------
    torch.Tensor
        Either a scalar CRPS value (reduce=True) or per-element CRPS values (reduce=False).

    Example
    -------
    >>> # Suppose you have forecasts => shape [batch, time, ensemble_size]
    >>> # and observations => shape [batch, time].
    >>> # by default, axis=-1 => ensemble dimension is the last axis
    >>> crps_val = crps_ensemble(observations, forecasts, axis=-1, estimator="pwm")
    >>> print(crps_val)
    """

    if estimator not in ["nrg", "pwm", "fair"]:
        raise ValueError(f"{estimator} is not a valid estimator. Must be one of ['nrg','pwm','fair'].")

    # Ensure observations have a singleton dimension for broadcasting
    # e.g., if obs is [B, T] and forecasts is [B, T, K], make obs [B, T, 1]
    if observations.shape != forecasts.shape[:-1]:
         if observations.shape == forecasts.shape[:axis] + forecasts.shape[axis+1:]:
             observations = observations.unsqueeze(axis)
         else:
            raise ValueError(
                f"Observations shape {observations.shape} must match forecast shape "
                f"{forecasts.shape} except for the ensemble dimension (axis={axis})."
            )

    # Move the ensemble axis to the last dimension if not already
    if axis != -1:
        forecasts = torch.moveaxis(forecasts, axis, -1)
        observations = torch.moveaxis(observations, axis, -1) # Also move obs if needed

    # If not sorted and needed, sort the ensemble dimension for PWM or NRG
    if (estimator in ["pwm", "nrg"]) and not sorted_ensemble:
        forecasts, _ = torch.sort(forecasts, dim=-1)

    # Call the aggregator function that handles the final CRPS calculation
    return ensemble(observations, forecasts, estimator=estimator, reduce=reduce)


def ensemble(obs: torch.Tensor, fct: torch.Tensor, estimator: str = "pwm", reduce: bool = True) -> torch.Tensor:
    """
    Compute the CRPS for a finite ensemble using the chosen estimator.

    Args:
        obs (torch.Tensor):
            Observations of shape [..., 1], matching fct except the last dimension
            which is the ensemble.
        fct (torch.Tensor):
            Forecast ensemble, shape [..., ensemble], sorted if needed for some estimators.
        estimator (str):
            "nrg", "pwm", or "fair".
        reduce (bool):
            If True, return scalar mean. If False, return tensor with shape [...].

    Returns:
        torch.Tensor:
            Either a scalar (mean) or tensor of CRPS values.

    Raises:
        ValueError: if no valid estimator is specified.
    """
    if estimator == "nrg":
        out = _crps_ensemble_nrg(obs, fct)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm(obs, fct)
    elif estimator == "fair":
        out = _crps_ensemble_fair(obs, fct)
    else:
        raise ValueError(f"Unknown estimator {estimator}")

    # Return the mean across all sample dimensions only if reduce is True.
    if reduce:
        return out.mean()
    else:
        return out # Return per-element CRPS


def _crps_ensemble_fair(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    The 'fair' version of CRPS for an ensemble:
      CRPS = mean_i |X_i - obs| - 0.5 * mean_{i != j} |X_i - X_j|

    Implementation detail:
      - M = fct.shape[-1]
      - The difference from 'nrg' is the normalization factor on the second term: M*(M-1)

    Args:
        obs: shape [..., 1], ensemble dimension last
        fct: shape [..., M], ensemble dimension last
    Returns:
        Tensor of shape [...], one CRPS value per sample, not yet averaged over batch.
    """
    M = fct.shape[-1]
    # e_1 => average over ensemble dimension of |obs - fct|
    # obs is [..., 1], fct is [..., M]. Broadcasting handles this.
    e_1 = torch.mean(torch.abs(obs - fct), dim=-1) # => shape [...]

    # e_2 => average pairwise difference among ensemble members
    diff = torch.abs(fct.unsqueeze(-2) - fct.unsqueeze(-1))  # => [..., M, M]
    # For fair CRPS, sum over pairs (M*(M-1) pairs) and divide by M*(M-1)
    # Ensure M > 1 to avoid division by zero
    if M > 1:
        e_2 = torch.sum(diff, dim=(-2, -1)) / (M * (M - 1.0))
    else:
        e_2 = torch.zeros_like(e_1) # If only 1 ensemble member, pairwise difference is 0

    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    CRPS estimator based on the 'energy' form:
       1/M sum_i |obs - x_i| - 1/(2M^2) sum_{i,j} |x_i - x_j|.

    Args:
        obs: shape [..., 1], ensemble dimension last
        fct: shape [..., M], ensemble dimension last
    Returns:
        Tensor of shape [...], CRPS per sample, not yet averaged over batch.
    """
    M = fct.shape[-1]
    # e_1 => 1/M sum |obs - x_i|
    e_1 = torch.mean(torch.abs(obs - fct), dim=-1) # => shape [...]

    # e_2 => 1/(M^2) sum_{i,j} |x_i - x_j|
    diff = torch.abs(fct.unsqueeze(-2) - fct.unsqueeze(-1))  # => shape [..., M, M]
    e_2 = torch.sum(diff, dim=(-2, -1)) / (M**2)

    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    CRPS estimator using Probability Weighted Moments (PWM).

    This code expects 'fct' is sorted along the ensemble dimension.
    If not sorted, pass sorted_ensemble=True or pre-sort it.

    The formula from DistPred paper (Eq. 13) is:
      C(F^, y) = 1/K sum_k |y^k - y| + 1/K sum_k y^k - (2/(K*(K-1))) sum_k y^k*(k-1)

    Args:
        obs: shape [..., 1], ensemble dimension last
        fct: shape [..., M], sorted ensemble dimension last
    Returns:
        Tensor of shape [...], CRPS per sample, not yet averaged over batch.
    """
    M = fct.shape[-1]
    # expected_diff => 1/M sum |obs - x_i|
    expected_diff = torch.mean(torch.abs(obs - fct), dim=-1)  # => shape [...]

    # beta_0 => mean(x_i) = 1/M sum x_i
    beta_0 = torch.mean(fct, dim=-1)

    # Term involving index-based weighting => sum_{k=1}^{M} x_k * (k-1)
    # We need indices [0, 1, ..., M-1]
    idx = torch.arange(M, device=fct.device, dtype=fct.dtype)
    # sum(fct * idx) => sum_{k=0}^{M-1} fct_k * k, which is sum_{k=1}^{M} fct_{k-1} * (k-1)
    # The paper uses index k starting from 1 for the sum, matching sorted array index + 1.
    # Let's use the paper's formula directly: (2 / (M * (M - 1))) * sum_{k=1}^{M} y_k * (k - 1)
    # where y_k is the k-th *sorted* prediction (fct is already sorted here).
    # index k runs from 1 to M => idx runs from 0 to M-1.
    # So we use sum(fct * idx) = sum_{k=0}^{M-1} fct_k * k = sum_{k=1}^{M} fct_{k-1} * (k-1)
    # Let's assume fct is indexed 0..M-1 corresponding to k=1..M in paper's formula

    # Ensure M > 1 to avoid division by zero in the normalization term
    if M > 1:
        # beta_1_term = (2 / (M * (M - 1.0))) * torch.sum(fct * idx, dim=-1)
        # Re-reading paper eq 13: C = mean(|yk-y|) + mean(yk) - (2/(K(K-1))) * sum(yk * (k-1)) 
        # where k goes from 1 to K. So idx should be k-1 => 0 to K-1. Correct.
        # Let's call fct -> yk (using 0-based indexing)
        term3 = (2.0 / (M * (M - 1.0))) * torch.sum(fct * idx, dim=-1)
    else:
        # If only one ensemble member, the third term is zero
        term3 = torch.zeros_like(beta_0)

    # CRPS = E[|Y - y|] + E[Y] - 2 * E[Y * rank(Y)] / (M-1) ? No, it's Eq 13.
    # CRPS = mean(|yk - y|) + mean(yk) - term3
    # PWM estimator: E(|X-y|) - E(|X-X'|)/2
    # The paper's eq 13 is: C = E[|Y_hat - y|] + E[Y_hat] - 2 * E[ Y_hat * (k-1) / (K-1) ] where k is index after sorting
    # So: E[|Y_hat - y|] is expected_diff
    # E[Y_hat] is beta_0
    # term3 = (2 / (K*(K-1))) * sum( yk * (k-1) )
    # CRPS = expected_diff + beta_0 - term3
    # Let's double check this calculation.
    # Original formula from paper: C(F^, y) = 1/K sum |yk - y| + 1/K sum yk - 2/(K(K-1)) sum yk*(k-1)
    # expected_diff = 1/K sum |yk - y|
    # beta_0 = 1/K sum yk
    # term3 = 2/(K(K-1)) sum yk*(k-1)
    # So, CRPS = expected_diff + beta_0 - term3 looks correct according to paper's Eq 13.

    return expected_diff + beta_0 - term3

