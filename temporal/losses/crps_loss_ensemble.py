import torch

def crps_ensemble(
    observations: torch.Tensor,
    forecasts: torch.Tensor,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
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

    Returns
    -------
    torch.Tensor
        A scalar CRPS value averaged across the entire batch. 
        (If you need a per-sample value, you may modify `ensemble(...)` below 
        to return the CRPS without taking a global `.mean()`.)

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

    # Move the ensemble axis to the last dimension if not already
    if axis != -1:
        forecasts = torch.moveaxis(forecasts, axis, -1)

    # If not sorted and needed, sort the ensemble dimension for PWM or NRG
    if (estimator in ["pwm", "nrg"]) and not sorted_ensemble:
        forecasts, _ = torch.sort(forecasts, dim=-1)

    # Call the aggregator function that handles the final CRPS calculation
    return ensemble(observations, forecasts, estimator=estimator)


def ensemble(obs: torch.Tensor, fct: torch.Tensor, estimator: str = "pwm") -> torch.Tensor:
    """
    Compute the CRPS for a finite ensemble using the chosen estimator.

    Args:
        obs (torch.Tensor):
            Observations of shape [...], matching fct except the last dimension
            which is the ensemble.
        fct (torch.Tensor):
            Forecast ensemble, shape [..., ensemble], sorted if needed for some estimators.
        estimator (str):
            "nrg", "pwm", or "fair".

    Returns:
        torch.Tensor:
            A scalar representing the mean CRPS across all items in `obs`.

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

    # Return the mean across all sample dimensions, i.e. a single scalar.
    return out.mean()


def _crps_ensemble_fair(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    The 'fair' version of CRPS for an ensemble:
      CRPS = mean_i |X_i - obs| - 0.5 * mean_{i != j} |X_i - X_j|

    Implementation detail:
      - M = fct.shape[-1]
      - The difference from 'nrg' is the normalization factor on the second term: M*(M-1)

    Args:
        obs: shape [...], no ensemble dimension
        fct: shape [..., M], ensemble dimension last
    Returns:
        Tensor of shape [...], one CRPS value per sample, not yet averaged over batch.
    """
    M = fct.shape[-1]
    # e_1 => average over ensemble dimension of |obs - fct|
    e_1 = torch.mean(torch.abs(obs.unsqueeze(-1) - fct), dim=-1)  # => shape [...]
    # e_2 => average pairwise difference among ensemble members
    #        shape [..., M, M], diagonal = 0
    diff = torch.abs(fct.unsqueeze(-2) - fct.unsqueeze(-1))  # => [..., M, M]
    e_2 = torch.mean(diff, dim=(-2, -1))  # => shape [...]
    # normalize by M*(M-1). We do e_2*(M^2 / (M*(M-1))) if we want consistent w/ 'fair', 
    # but in this original code: e_2 is 1/(M^2) times sum_{i,j} => "nrg" approach.
    # DistPred's 'fair' approach does: e_2 / (M*(M-1)) => let's replicate the code snippet exactly:
    # Actually the snippet: e_2 is sum(|xi-xj|)/(M*(M-1)) => let's do it:
    # We'll replicate the snippet from your code exactly:
    # => final is e_1 - 0.5 * e_2, but e_2 is scaled differently.
    # We'll keep it consistent with the snippet:
    # in snippet, e_1 = sum( |obs - x_i|)/M, e_2 = sum(|x_i-x_j|)/(M*(M-1)).
    # but we did e_2 = average over i,j => => 1/(M*M). So let's fix that for fair:
    e_1 = torch.sum(torch.abs(obs.unsqueeze(-1) - fct), dim=-1) / M
    # For fair approach, we re-do e_2
    e_2 = torch.sum(diff, dim=(-2, -1)) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    CRPS estimator based on the 'energy' form:
       1/M sum_i |obs - x_i| - 1/(2M^2) sum_{i,j} |x_i - x_j|.

    Args:
        obs: shape [...], no ensemble dimension
        fct: shape [..., M], ensemble dimension last
    Returns:
        Tensor of shape [...], CRPS per sample, not yet averaged over batch.
    """
    M = fct.shape[-1]
    # e_1 => 1/M sum |obs - x_i|
    e_1 = torch.sum(torch.abs(obs.unsqueeze(-1) - fct), dim=-1) / M

    # e_2 => 1/(M^2) sum_{i,j} |x_i - x_j|
    diff = torch.abs(fct.unsqueeze(-2) - fct.unsqueeze(-1))  # => shape [..., M, M]
    e_2 = torch.sum(diff, dim=(-2, -1)) / (M**2)

    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    CRPS estimator using Probability Weighted Moments (PWM).

    This code expects 'fct' is sorted along the ensemble dimension.
    If not sorted, pass sorted_ensemble=True or pre-sort it.

    The formula is:
      CRPS = E(|obs - x_i|) + β_0 - 2 * β_1,
      where:
       - E(...) is average over x_i
       - β_0 = mean(x_i)
       - β_1 = sum(i/M*(M-1)) ?

    For reference, see the DistPred-Forecast snippet or scikit-garden approach.

    Args:
        obs: shape [...], no ensemble dimension
        fct: shape [..., M], sorted ensemble dimension last
    Returns:
        Tensor of shape [...], CRPS per sample, not yet averaged over batch.
    """
    M = fct.shape[-1]
    # expected_diff => average of |obs - x_i|
    expected_diff = torch.mean(torch.abs(obs.unsqueeze(-1) - fct), dim=-1)  # => shape [...]

    # β_0 => mean of x_i
    beta_0 = torch.mean(fct, dim=-1)

    # index-based weighting => e.g. sum_{k=0}^{M-1} x_k * k / [M*(M-1)]
    # we need an index array => [0,1,..., M-1]
    idx = torch.arange(M, device=fct.device, dtype=fct.dtype)
    # (fct * idx).sum(...) => sum_{k=0}^{M-1} x_k * k
    # Then / [M*(M-1)] => average
    beta_1 = torch.sum(fct * idx, dim=-1) / (M * (M - 1.0))

    # CRPS = expected_diff + beta_0 - 2 * beta_1
    return expected_diff + beta_0 - 2.0 * beta_1
