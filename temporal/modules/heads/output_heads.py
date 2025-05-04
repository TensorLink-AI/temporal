import torch
import torch.nn as nn

from temporal.registry.core import register_module
from temporal.modules.heads.base_output_head import BaseOutputHead
from temporal.modules.losses.losses import TimeSeriesLoss, CRPSLoss # Import CRPSLoss as well
# Assuming QuantileLoss or MQLoss might be used by MultiQuantileHead
from temporal.modules.losses.loss_functions import QuantileLoss # Or MQLoss if that's what it uses

# ------------------------------------------------------ -
# ✅ 1. LinearOutputHead
# ------------------------------------------------------ -

@register_module("output_head", "linear")
class LinearOutputHead(BaseOutputHead):
    def __init__(self, hidden_size: int, output_size: int = 1, loss_type: str = "mse", **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_size, output_size)
        self.loss_type = loss_type # This might be ignored if builder uses main config loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    # It's better practice if the model builder selects the loss based on the main config,
    # rather than the head dictating it. This method could be removed or made optional.
    # def get_loss_fn(self):
    #     # Let the model builder decide the loss based on config.loss_type
    #     # return TimeSeriesLoss(loss_type=self.loss_type)
    #     pass


# ------------------------------------------------------ -
# ✅ 2. GaussianHead
# ------------------------------------------------------ -

@register_module("output_head", "gaussian")
class GaussianHead(BaseOutputHead):
    def __init__(self, hidden_size: int, output_size: int = 1, **kwargs): # Added output_size for consistency
        super().__init__()
        # Output mean and log_std for each output dimension
        self.proj = nn.Linear(hidden_size, output_size * 2)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [..., output_size * 2] -> reshape or split later
        return self.proj(x)

    # Let the model builder decide the loss (e.g., GaussianNLLLoss)
    # def get_loss_fn(self):
    #     # Should map to GaussianNLLLoss, handled by builder
    #     pass


# ------------------------------------------------------ -
# ✅ 3. TDistributionHead - Assuming this exists
# ------------------------------------------------------ -

# @register_module("output_head", "t_distribution")
# class TDistributionHead(BaseOutputHead):
#     def __init__(self, hidden_size: int, output_size: int = 1, **kwargs): # Added output_size
#         super().__init__()
#         # Output mu, log_sigma, log_nu for each output dimension
#         self.proj = nn.Linear(hidden_size, output_size * 3)
#         self.output_size = output_size
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.proj(x)
#
#     # Let the model builder handle loss (e.g., NegativeLogLikelihood for TDist)
#     # def get_loss_fn(self):
#     #     pass


# ------------------------------------------------------ -
# ✅ 4. MultiQuantileHead (or QuantileRegressionOutputHead)
# ------------------------------------------------------ -
# Renaming for clarity if it's the standard one
@register_module("output_head", "quantile_regression")
class QuantileRegressionOutputHead(BaseOutputHead):
    def __init__(self, hidden_size: int, output_dims: list = [1], quantiles: list = [0.5], **kwargs):
        """
        Args:
            hidden_size (int): Input hidden dimension.
            output_dims (list): List of output dimensions for each quantile.
                                Usually [feature_size] * num_quantiles for univariate.
            quantiles (list): List of target quantiles.
        """
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        # Total output dimensions = sum(output_dims), should be feature_size * num_quantiles
        total_output_dim = sum(output_dims)
        # Assuming feature_size is the first element of output_dims
        feature_size = output_dims[0] if output_dims else 1
        if total_output_dim != self.num_quantiles * feature_size:
             print(f"Warning: output_dims {output_dims} might not align with num_quantiles {self.num_quantiles} and feature_size {feature_size}")

        self.proj = nn.Linear(hidden_size, total_output_dim)
        # Store feature_size and num_quantiles for reshaping
        self.feature_size = feature_size


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, HiddenSize].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, FeatureSize, NumQuantiles]
                          or [B, T, NumQuantiles] if FeatureSize is 1.
        """
        # Project to [B, T, TotalOutputDim]
        proj_out = self.proj(x)
        # Reshape to [B, T, FeatureSize, NumQuantiles] if FeatureSize > 1
        if self.feature_size > 1 and proj_out.shape[-1] == self.feature_size * self.num_quantiles:
             return proj_out.view(*proj_out.shape[:-1], self.feature_size, self.num_quantiles)
        else:
             # Assume univariate or already correct shape [B, T, NumQuantiles]
             return proj_out # Shape: [B, T, NumQuantiles]

    # Let the builder handle loss selection (e.g. MQLoss)
    # def get_loss_fn(self):
    #     pass


# ------------------------------------------------------ -
# ✨ 5. NEW DistPredHead ✨
# ------------------------------------------------------ -

@register_module("output_head", "distpred")
class DistPredHead(BaseOutputHead):
    """
    Output head specifically for DistPred approach using CRPS loss.
    Outputs K predictions (treated as an ensemble/quantiles) per feature dimension.
    """
    def __init__(self, hidden_size: int, num_outputs: int, feature_size: int = 1, **kwargs):
        """
        Args:
            hidden_size (int): Input hidden dimension from the backbone.
            num_outputs (int): The number of ensemble predictions (K) required by CRPS.
                               This should match config.num_quantiles.
            feature_size (int): The number of features being predicted (e.g., 1 for univariate).
            **kwargs: Catches unused arguments like 'quantiles' from the config.
        """
        super().__init__()
        self.num_outputs = num_outputs # K
        self.feature_size = feature_size
        # Total output dimension = K * feature_size
        total_output_dim = num_outputs * feature_size
        self.proj = nn.Linear(hidden_size, total_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, HiddenSize].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, FeatureSize, NumOutputs]
                          or [B, T, NumOutputs] if FeatureSize is 1.
        """
        proj_out = self.proj(x) # Shape [B, T, TotalOutputDim]

        # Reshape if multivariate
        if self.feature_size > 1:
            return proj_out.view(*proj_out.shape[:-1], self.feature_size, self.num_outputs)
        else:
            # Univariate case, shape is [B, T, NumOutputs]
            return proj_out

    # This head is specifically for CRPS, but let the builder confirm/instantiate the loss
    # based on the main config's loss_type='crps'.
    # def get_loss_fn(self):
    #     # return CRPSLoss() # Builder should handle this
    #     pass
