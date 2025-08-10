import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union, Callable

from temporal.registry.core import register_module
from temporal.modules.heads.base_output_head import BaseOutputHead

# Note: Loss functions are generally not returned by heads anymore,
# as the main training configuration is responsible for specifying the loss.
# Imports are kept for context but could be removed.
from temporal.modules.losses.losses import TimeSeriesLoss, CRPSLoss
from temporal.modules.losses.loss_functions import QuantileLoss


@register_module("output_head", "linear")
class LinearOutputHead(BaseOutputHead):
    """
    A simple linear projection head for point forecasts.

    This head applies a single linear layer to the final hidden state of the
    model to produce a point forecast for each time step.

    Attributes:
        proj (nn.Linear): The linear projection layer.
    """
    def __init__(self, hidden_size: int, output_size: int = 1, **kwargs):
        """
        Initializes the LinearOutputHead.

        Args:
            hidden_size (int): The dimension of the input hidden state.
            output_size (int): The dimension of the output forecast.
            **kwargs: Catches any unused arguments.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the hidden state to the output forecast.

        Args:
            x (torch.Tensor): The input hidden state of shape `[B, T, hidden_size]`.

        Returns:
            torch.Tensor: The point forecast of shape `[B, T, output_size]`.
        """
        return self.proj(x)

    def get_loss_fn(self) -> Optional[Callable]:
        """Returns None, as loss is determined by the main training config."""
        return None


@register_module("output_head", "gaussian")
class GaussianHead(BaseOutputHead):
    """
    An output head for predicting parameters of a Gaussian distribution.

    This head projects the final hidden state into a mean (mu) and a
    standard deviation (sigma) for each feature, allowing for probabilistic
    forecasts. The log standard deviation is output to ensure positivity.

    Attributes:
        feature_size (int): The number of features to predict distributions for.
        proj (nn.Linear): The linear layer that projects the hidden state to
            the distribution parameters (mu and log_sigma for each feature).
    """
    def __init__(self, hidden_size: int, output_size: int = 1, **kwargs):
        """
        Initializes the GaussianHead.

        Args:
            hidden_size (int): The dimension of the input hidden state.
            output_size (int): The number of features for which to predict a
                distribution.
            **kwargs: Catches any unused arguments.
        """
        super().__init__()
        self.feature_size = output_size
        # Project to 2 parameters (mean, log_std) for each feature.
        self.proj = nn.Linear(hidden_size, self.feature_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the hidden state to the Gaussian distribution parameters.

        Args:
            x (torch.Tensor): The input hidden state of shape `[B, T, hidden_size]`.

        Returns:
            torch.Tensor: A tensor of shape `[B, T, feature_size * 2]`,
            containing the concatenated means and log standard deviations.
        """
        return self.proj(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from the predicted Gaussian distribution during generation.

        Args:
            x: [B, 1, 2F] output from forward()

        Returns:
            Sampled features: [B, 1, F]
        """
        mu, log_sigma = x.chunk(2, dim=-1)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + eps * sigma

    def sample_quantiles(self, x: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor:
        mu, log_sigma = x.chunk(2, dim=-1)
        sigma = torch.exp(log_sigma)
        q_values = torch.tensor(quantile_levels, dtype=mu.dtype, device=mu.device)
        z = torch.distributions.Normal(0, 1).icdf(q_values)  # [Q]
        mu = mu.unsqueeze(-1)      # [B, 1, F, 1]
        sigma = sigma.unsqueeze(-1)  # [B, 1, F, 1]
        return mu + sigma * z      # [B, 1, F, Q]

    def get_loss_fn(self) -> Optional[Callable]:
        """Returns None, as loss is determined by the main training config."""
        return None


@register_module("output_head", "quantile_regression")
class QuantileRegressionOutputHead(BaseOutputHead):
    """
    An output head for multi-quantile regression.

    This head projects the final hidden state to a set of predicted quantiles
    for each feature, enabling probabilistic forecasting without assuming a
    specific distribution.

    Attributes:
        num_quantiles (int): The number of quantiles to predict.
        feature_size (int): The number of features.
        proj (nn.Linear): The linear layer that projects to the quantile forecasts.
    """
    def __init__(self, hidden_size: int, output_size: int, num_quantiles: int, feature_size: int = 1, **kwargs):
        """
        Initializes the QuantileRegressionOutputHead.

        Args:
            hidden_size (int): The dimension of the input hidden state.
            output_size (int): The total output dimension, which must equal
                `num_quantiles * feature_size`.
            num_quantiles (int): The number of quantiles to predict.
            feature_size (int): The number of features per time step.
            **kwargs: Catches unused arguments from the configuration.
        """
        super().__init__()
        if output_size != num_quantiles * feature_size:
            raise ValueError(
                f"Output size mismatch: output_size ({output_size}) must equal "
                f"num_quantiles ({num_quantiles}) * feature_size ({feature_size})."
            )

        self.num_quantiles = num_quantiles
        self.feature_size = feature_size
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the hidden state to the quantile forecasts.

        Args:
            x (torch.Tensor): The input tensor of shape `[B, T, hidden_size]`.

        Returns:
            torch.Tensor: The quantile forecasts, reshaped to
            `[B, T, feature_size, num_quantiles]` for multivariate cases or
            `[B, T, num_quantiles]` for univariate cases.
        """
        projected_output = self.proj(x)
        if self.feature_size > 1:
            return projected_output.view(*projected_output.shape[:-1], self.feature_size, self.num_quantiles)
        else:
            return projected_output

    def get_loss_fn(self) -> Optional[Callable]:
        """Returns None, as loss is determined by the main training config."""
        return None


@register_module("output_head", "distpred")
class DistPredHead(BaseOutputHead):
    """
    An output head for the Distribution Prediction (DistPred) approach.

    This head is designed for models that use CRPS loss. It outputs a specified
    number of values per feature, which are treated as an ensemble or a set of
    empirical quantiles for calculating the loss.

    Attributes:
        num_outputs (int): The number of prediction values (K) per feature.
        feature_size (int): The number of features.
        use_tanh (bool): Whether to apply tanh-based output normalization.
        tanh_scale (float): Scaling factor applied after tanh.
        proj (nn.Linear): The linear projection layer.
    """

    def __init__(self, hidden_size: int, output_size: int, **kwargs):
        """
        Initializes the DistPredHead.

        Args:
            hidden_size (int): The dimension of the input hidden state.
            output_size (int): The total output dimension, must equal
                `num_outputs * feature_size`.
            **kwargs: Must contain 'num_outputs', and optionally 'feature_size',
                      'use_tanh', and 'tanh_scale'.
        """
        super().__init__()

        if 'num_outputs' not in kwargs:
            raise ValueError("DistPredHead requires 'num_outputs' to be specified in the configuration.")
        self.num_outputs = kwargs['num_outputs']
        self.feature_size = kwargs.get('feature_size', 1)

        expected_output_size = self.num_outputs * self.feature_size
        if output_size != expected_output_size:
            raise ValueError(
                f"DistPredHead output size mismatch: output_size ({output_size}) "
                f"!= num_outputs ({self.num_outputs}) * feature_size ({self.feature_size})"
            )

        self.proj = nn.Linear(hidden_size, output_size)

        # Optional tanh normalization
        self.use_tanh = kwargs.get('use_tanh', False)
        self.tanh_scale = kwargs.get('tanh_scale', 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the hidden state to the ensemble predictions.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, hidden_size].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, feature_size, num_outputs]
                          or [B, T, num_outputs] if univariate.
        """
        projected_output = self.proj(x)

        if self.use_tanh:
            projected_output = self.tanh_scale * torch.tanh(projected_output)

        return projected_output.view(*projected_output.shape[:-1], self.feature_size, self.num_outputs)

    def predict(self, x: torch.Tensor, method: Union[str, float, int] = "median") -> torch.Tensor:
        """
        Collapse an ensemble head output [B, T, K] or [B, T, F, K] to [B, T, 1] or [B, T, F].

        Args:
            x (torch.Tensor): Head output of shape [B, T, K] or [B, T, F, K].
            method (str|float|int): Collapse strategy — 'mean', 'median',
                quantile (float), or direct index (int).

        Returns:
            torch.Tensor: Collapsed prediction tensor.
        """
        is_multi = (x.ndim == 4)  # [B, T, F, K]
        K = x.shape[-1]

        if isinstance(method, str):
            if method == "mean":
                return x.mean(dim=-1, keepdim=not is_multi)
            elif method == "median":
                idx = K // 2
            else:
                raise ValueError(f"Unsupported method string: {method!r}")
        elif isinstance(method, float):
            qs = getattr(self, "quantiles", None) or getattr(self.config, "quantiles", None)
            if qs is None or len(qs) != K:
                raise ValueError(f"No valid quantiles list of length {K} found.")
            qt = torch.tensor(qs, device=x.device)
            idx = (qt - method).abs().argmin().item()
        elif isinstance(method, int):
            idx = method if method >= 0 else K + method
            if not (0 <= idx < K):
                raise IndexError(f"Index {method} out of range for K={K}")
        else:
            raise TypeError(f"method must be str|float|int, not {type(method)}")

        return x[..., idx] if is_multi else x[..., idx:idx+1]
        
    def sample_quantiles(self, x: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor:
        """
        Computes quantiles from the ensemble output.
        Args:
            x (torch.Tensor): Head output of shape [B, T, K] or [B, T, F, K].
            quantile_levels (List[float]): List of quantile levels to compute.
        Returns:
            torch.Tensor: Quantile predictions of shape [B, T, F, Q].
        """
        # Sort the ensemble members along the last dimension
        sorted_x, _ = torch.sort(x, dim=-1)

        # Ensure quantile_levels is a tensor
        q_tensor = torch.tensor(quantile_levels, device=x.device, dtype=x.dtype)
        q_tensor = q_tensor.view(1, 1, 1, -1)  # Reshape for broadcasting

        # Get the number of ensemble members
        num_outputs = sorted_x.shape[-1]

        # Calculate the indices for the desired quantiles
        indices = (q_tensor * (num_outputs - 1)).round().long()

        # Gather the quantile values
        # The shape of indices is [1, 1, 1, Q]
        # The shape of sorted_x is [B, T, F, K]
        # We need to expand indices to match the dimensions of sorted_x
        indices = indices.expand(sorted_x.shape[0], sorted_x.shape[1], sorted_x.shape[2], -1)
        
        # Gather the quantiles
        quantiles = torch.gather(sorted_x, -1, indices)

        return quantiles


    def get_loss_fn(self) -> Optional[Callable]:
        """
        Returns None — the main training config defines the loss function.
        """
        return None



@register_module("output_head", "mixture")
class MixtureOutputHead(BaseOutputHead):
    """
    An output head for Mixture Density Networks (MDNs).

    This head predicts the parameters for a mixture of several probability
    distributions (e.g., a mix of Student's T and Log-Normal). It outputs a
    dictionary of parameters required by the `MixtureLoss` function.

    Attributes:
        output_projection (nn.Linear): The linear layer that projects the hidden
            state to all required distribution and mixture parameters.
        param_indices (Dict[str, Tuple[int, int]]): A dictionary mapping parameter
            names to their slice indices in the flat output tensor.
    """
    DIST_PARAM_COUNTS = {
        "student_t": {"df": 1, "mu": 1, "scale": 1},
        "log_normal": {"mu": 1, "sigma": 1},
        "neg_binomial": {"r": 1, "p": 1},
        "normal": {"mu": 1, "sigma": 1},
        "fixed_normal": {"mu": 1}
    }
    DIST_OUTPUT_KEYS = {
        "student_t": {"df": "student_df", "mu": "student_mu", "scale": "student_scale"},
        "log_normal": {"mu": "lognorm_mu", "sigma": "lognorm_sigma"},
        "neg_binomial": {"r": "nb_r", "p": "nb_p"},
        "normal": {"mu": "normal_mu", "sigma": "normal_sigma"},
        "fixed_normal": {"mu": "normal_mu"}
    }

    def __init__(self, hidden_size: int, components: List[str], **kwargs):
        """
        Initializes the MixtureOutputHead.

        Args:
            hidden_size (int): The dimension of the input hidden state.
            components (List[str]): A list of distribution names for the mixture
                (e.g., ["student_t", "log_normal"]).
            **kwargs: Catches unused arguments.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.components = components
        self.num_components = len(components)

        if self.num_components <= 0:
            raise ValueError("MixtureOutputHead requires at least one component.")

        # Calculate the total number of parameters to predict.
        total_params_dim = 0
        self.param_indices: Dict[str, Tuple[int, int]] = {}
        current_idx = 0

        for dist_name in self.components:
            if dist_name not in self.DIST_PARAM_COUNTS:
                raise ValueError(f"Unknown distribution component '{dist_name}'. Supported: {list(self.DIST_PARAM_COUNTS.keys())}")

            param_counts_for_dist = self.DIST_PARAM_COUNTS[dist_name]
            output_keys_for_dist = self.DIST_OUTPUT_KEYS[dist_name]

            for param_key, count in param_counts_for_dist.items():
                output_key_name = output_keys_for_dist[param_key]
                self.param_indices[output_key_name] = (current_idx, current_idx + count)
                current_idx += count
                total_params_dim += count

        # Add parameters for the mixture weights (logits).
        self.mixture_logits_indices = (current_idx, current_idx + self.num_components)
        total_params_dim += self.num_components

        self.output_projection = nn.Linear(hidden_size, total_params_dim)

        if "output_size" in kwargs and kwargs["output_size"] != total_params_dim:
            raise ValueError(
                f"MixtureOutputHead: provided 'output_size' ({kwargs['output_size']}) "
                f"does not match the derived total parameter dimension ({total_params_dim})."
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Projects the hidden state to the mixture distribution parameters.

        Args:
            x (torch.Tensor): Input tensor of shape `[B, T, hidden_size]`.

        Returns:
            Dict[str, Union[torch.Tensor, List[str]]]: A dictionary of parameters
            formatted for use with `MixtureLoss`.
        """
        flat_params = self.output_projection(x)
        output_dict: Dict[str, Union[torch.Tensor, List[str]]] = {"components": self.components}

        # Extract mixture logits.
        logits_start, logits_end = self.mixture_logits_indices
        output_dict["mixture_logits"] = flat_params[..., logits_start:logits_end]

        # Extract parameters for each component distribution.
        processed_params = set()
        for dist_name in self.components:
            output_key_map = self.DIST_OUTPUT_KEYS[dist_name]
            for param_internal_name in self.DIST_PARAM_COUNTS[dist_name]:
                actual_output_key = output_key_map[param_internal_name]
                if actual_output_key not in processed_params:
                    start_idx, end_idx = self.param_indices[actual_output_key]
                    param_tensor = flat_params[..., start_idx:end_idx]
                    # Squeeze the last dimension if it's 1.
                    output_dict[actual_output_key] = param_tensor.squeeze(-1) if param_tensor.shape[-1] == 1 else param_tensor
                    processed_params.add(actual_output_key)

        return output_dict

    def get_loss_fn(self) -> Optional[Callable]:
        """Returns None, as loss is determined by the main training config."""
        return None
