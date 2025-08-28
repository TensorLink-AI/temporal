import torch

def crps_ensemble(
    observations: torch.Tensor,
    forecasts: torch.Tensor,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
    reduce: bool = True,
) -> torch.Tensor:
    r"""
    Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble
    of forecasts. This code is adapted from DistPred-Forecast.

    Parameters
    ----------
    observations : torch.Tensor
        The observed values. Shape must be broadcastable to forecasts shape excluding
        the ensemble dimension `axis`.
    forecasts : torch.Tensor
        The ensemble predictions. Must have an ensemble dimension specified by `axis`.
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

    Raises
    ------
    ValueError
        If estimator is invalid or shapes are incompatible.

    Example
    -------
    >>> # forecasts [B, T, K], observations [B, T]
    >>> crps_ensemble(observations, forecasts, axis=-1)
    >>> # forecasts [B, K, T], observations [B, T]
    >>> crps_ensemble(observations, forecasts, axis=1)
    """

    if estimator not in ["nrg", "pwm", "fair"]:
        raise ValueError(f"{estimator} is not a valid estimator. Must be one of ['nrg','pwm','fair'].")

    # --- Revised Shape Handling --- 
    # Get the actual axis index (handles negative index)
    actual_axis = axis if axis >= 0 else forecasts.ndim + axis
    if not (0 <= actual_axis < forecasts.ndim):
        raise ValueError(f"Invalid axis {axis} for forecasts dimension {forecasts.ndim}")

    # Check if observations needs unsqueezing
    if observations.ndim == forecasts.ndim - 1:
        # Check if shape matches forecasts excluding the ensemble dim
        expected_obs_shape = list(forecasts.shape)
        del expected_obs_shape[actual_axis]
        if list(observations.shape) == expected_obs_shape:
            observations = observations.unsqueeze(actual_axis) # Add singleton dim
        else:
            raise ValueError(
                f"Observations shape {observations.shape} does not match forecast shape "
                f"{forecasts.shape} excluding axis {axis}. Expected {tuple(expected_obs_shape)}."
            )
    elif observations.ndim == forecasts.ndim:
        # Check if the ensemble dimension in observations has size 1
        if observations.shape[actual_axis] != 1:
            raise ValueError(
                f"Observations shape {observations.shape} has same ndim as forecasts {forecasts.shape}, "
                f"but dimension at axis {axis} is not 1."
            )
        # Shapes are compatible for broadcasting, no unsqueeze needed
    else:
        raise ValueError(
            f"Observations ndim ({observations.ndim}) must be equal to or one less than "
            f"forecasts ndim ({forecasts.ndim})."
        )
    # At this point, obs shape should be broadcastable with fct shape
    # --- End Revised Shape Handling ---

    # Move the ensemble axis to the last dimension if not already
    # This simplifies the internal _crps_* functions
    if actual_axis != forecasts.ndim - 1:
        forecasts = torch.moveaxis(forecasts, actual_axis, -1)
        # Move observations dim only if it exists (was unsqueezed or already there)
        if observations.ndim == forecasts.ndim: 
             observations = torch.moveaxis(observations, actual_axis, -1)

    # If not sorted and needed, sort the ensemble dimension for PWM or NRG
    if (estimator in ["pwm", "nrg"]) and not sorted_ensemble:
        forecasts, _ = torch.sort(forecasts, dim=-1)

    # Call the aggregator function that handles the final CRPS calculation
    # Pass obs and fct with ensemble dim last
    return ensemble(observations, forecasts, estimator=estimator, reduce=reduce)


def ensemble(obs: torch.Tensor, fct: torch.Tensor, estimator: str = "pwm", reduce: bool = True) -> torch.Tensor:
    """
    Compute the CRPS for a finite ensemble using the chosen estimator.
    Assumes ensemble dimension is the *last* dimension for both obs and fct.

    Args:
        obs (torch.Tensor):
            Observations, shape [..., 1] (ensemble dim last).
        fct (torch.Tensor):
            Forecast ensemble, shape [..., ensemble], sorted if needed.
        estimator (str):
            "nrg", "pwm", or "fair".
        reduce (bool):
            If True, return scalar mean. If False, return tensor with shape [...].

    Returns:
        torch.Tensor:
            Either a scalar (mean) or tensor of CRPS values ([...]).
    """
    if obs.ndim < fct.ndim:
        obs = obs.unsqueeze(-1)
    # Ensure shapes are compatible for broadcasting along last dimension
    if not torch.broadcast_shapes(obs.shape, fct.shape):
         # This check might be redundant given the checks in crps_ensemble
         raise ValueError(f"Cannot broadcast observation shape {obs.shape} with forecast shape {fct.shape}")
         
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


# Internal functions (_crps_ensemble_fair, _crps_ensemble_nrg, _crps_ensemble_pwm)
# assume ensemble dimension is LAST

def _crps_ensemble_fair(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    Args:
        obs: shape [..., 1], ensemble dimension last
        fct: shape [..., M], ensemble dimension last
    Returns:
        Tensor of shape [...], one CRPS value per sample.
    """
    M = fct.shape[-1]
    # Broadcasting handles obs [..., 1] vs fct [..., M]
    e_1 = torch.mean(torch.abs(obs - fct), dim=-1) # => shape [...]

    diff = torch.abs(fct.unsqueeze(-2) - fct.unsqueeze(-1))  # => [..., M, M]
    if M > 1:
        # Sum over M*(M-1) pairs
        e_2 = torch.sum(diff, dim=(-2, -1)) / (M * (M - 1.0))
    else:
        e_2 = torch.zeros_like(e_1)
    return e_1 - 0.5 * e_2

def _crps_ensemble_nrg(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    Args:
        obs: shape [..., 1], ensemble dimension last
        fct: shape [..., M], ensemble dimension last
    Returns:
        Tensor of shape [...], CRPS per sample.
    """
    M = fct.shape[-1]
    # Broadcasting handles obs [..., 1] vs fct [..., M]
    e_1 = torch.mean(torch.abs(obs - fct), dim=-1) # => shape [...]

    diff = torch.abs(fct.unsqueeze(-2) - fct.unsqueeze(-1))  # => shape [..., M, M]
    # Sum over M*M pairs (including i==j where diff is 0)
    e_2 = torch.sum(diff, dim=(-2, -1)) / (M**2)
    return e_1 - 0.5 * e_2

def _crps_ensemble_pwm(obs: torch.Tensor, fct: torch.Tensor) -> torch.Tensor:
    """
    Args:
        obs: shape [..., 1], ensemble dimension last
        fct: shape [..., M], sorted ensemble dimension last
    Returns:
        Tensor of shape [...], CRPS per sample.
    """
    M = fct.shape[-1]
    # Broadcasting handles obs [..., 1] vs fct [..., M]
    expected_diff = torch.mean(torch.abs(obs - fct), dim=-1)  # => shape [...]
    beta_0 = torch.mean(fct, dim=-1)
    idx = torch.arange(M, device=fct.device, dtype=fct.dtype)

    if M > 1:
        term3 = (2.0 / (M * (M - 1.0))) * torch.sum(fct * idx, dim=-1)
    else:
        term3 = torch.zeros_like(beta_0)

    # Eq 13 from DistPred paper: C = E[|Y^ - y|] + E[Y^] - (2/(K(K-1))) * sum(y_k * (k-1))
    return expected_diff + beta_0 - term3
