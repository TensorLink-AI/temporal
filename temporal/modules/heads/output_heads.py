import torch
import torch.nn as nn
from typing import Optional # Added for type hinting

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
    def __init__(self, hidden_size: int, output_size: int = 1, **kwargs): # Removed loss_type default
        super().__init__()
        self.proj = nn.Linear(hidden_size, output_size)
        # Removed self.loss_type - Loss should be handled by main config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    # Removed get_loss_fn - Loss should be handled by main model builder
    # def get_loss_fn(self):
    #     pass


# ------------------------------------------------------ -
# ✅ 2. GaussianHead
# ------------------------------------------------------ -

@register_module("output_head", "gaussian")
class GaussianHead(BaseOutputHead):
    def __init__(self, hidden_size: int, output_size: int = 1, **kwargs):
        super().__init__()
        # output_size here means feature_size
        self.feature_size = output_size
        # Output mean and log_std for each feature dimension
        self.proj = nn.Linear(hidden_size, self.feature_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [..., feature_size * 2] -> reshape or split later
        return self.proj(x)

    # Removed get_loss_fn


# ------------------------------------------------------ -
# ✅ 3. TDistributionHead - Still commented out
# ------------------------------------------------------ -


# ------------------------------------------------------ -
# ✅ 4. QuantileRegressionOutputHead
# ------------------------------------------------------ -
@register_module("output_head", "quantile_regression")
class QuantileRegressionOutputHead(BaseOutputHead):
    # Changed output_dims to output_size (total dimension) and added num_quantiles
    def __init__(self, hidden_size: int, output_size: int, num_quantiles: int, feature_size: int = 1, **kwargs):
        """
        Args:
            hidden_size (int): Input hidden dimension.
            output_size (int): Total output dimension (num_quantiles * feature_size).
            num_quantiles (int): Number of quantiles to predict.
            feature_size (int): Number of features per time step.
            **kwargs: Catches unused args like 'quantiles' list from config.
        """
        super().__init__()
        # Validate consistency
        if output_size != num_quantiles * feature_size:
            raise ValueError(
                f"Output size mismatch: output_size ({output_size}) != "
                f"num_quantiles ({num_quantiles}) * feature_size ({feature_size})"
            )

        self.num_quantiles = num_quantiles
        self.feature_size = feature_size
        self.proj = nn.Linear(hidden_size, output_size)


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
        if self.feature_size > 1:
             # Ensure the last dimension matches total output dim before reshape
             if proj_out.shape[-1] == self.feature_size * self.num_quantiles:
                 return proj_out.view(*proj_out.shape[:-1], self.feature_size, self.num_quantiles)
             else:
                  raise ValueError(f"Output projection shape mismatch: {proj_out.shape[-1]} != {self.feature_size} * {self.num_quantiles}")
        else:
             # Assume univariate, shape is [B, T, NumQuantiles]
             # Ensure last dim matches num_quantiles
             if proj_out.shape[-1] == self.num_quantiles:
                 return proj_out
             else:
                  raise ValueError(f"Output projection shape mismatch: {proj_out.shape[-1]} != {self.num_quantiles}")

    # Removed get_loss_fn


# ------------------------------------------------------ -
# ✨ 5. UPDATED DistPredHead ✨
# ------------------------------------------------------ -

@register_module("output_head", "distpred")
class DistPredHead(BaseOutputHead):
    """
    Output head specifically for DistPred approach using CRPS loss.
    Outputs K predictions (treated as an ensemble/quantiles) per feature dimension.
    Accepts 'output_size' for builder compatibility but uses 'num_outputs' and 'feature_size' internally.
    """
    # Accept output_size for builder compatibility, but get essential info from kwargs
    def __init__(self, hidden_size: int, output_size: int, **kwargs):
        """
        Args:
            hidden_size (int): Input hidden dimension from the backbone.
            output_size (int): Total output dimension (must equal num_outputs * feature_size).
            **kwargs: Must contain 'num_outputs' (K) and 'feature_size'.
        """
        super().__init__()

        # Extract required args from kwargs
        if 'num_outputs' not in kwargs:
            raise ValueError("DistPredHead requires 'num_outputs' in kwargs")
        if 'feature_size' not in kwargs:
            # Default to 1 if not provided, but better to be explicit in config
            kwargs['feature_size'] = 1
            print("Warning: 'feature_size' not found in DistPredHead kwargs, defaulting to 1.")

        self.num_outputs = kwargs['num_outputs'] # K
        self.feature_size = kwargs['feature_size']

        # Validate consistency between output_size and num_outputs * feature_size
        expected_output_size = self.num_outputs * self.feature_size
        if output_size != expected_output_size:
            raise ValueError(
                f"DistPredHead output size mismatch: output_size provided ({output_size}) != "
                f"num_outputs ({self.num_outputs}) * feature_size ({self.feature_size})"
            )

        # Use the validated output_size for the projection layer
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, HiddenSize].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, FeatureSize, NumOutputs]
                          or [B, T, NumOutputs] if FeatureSize is 1.
        """
        proj_out = self.proj(x) # Shape [B, T, TotalOutputDim = FeatureSize * NumOutputs]

        # Reshape if multivariate
        if self.feature_size > 1:
            # Ensure last dimension matches before reshape
            if proj_out.shape[-1] == self.feature_size * self.num_outputs:
                return proj_out.view(*proj_out.shape[:-1], self.feature_size, self.num_outputs)
            else:
                # This shouldn't happen if __init__ validation passed
                raise RuntimeError(f"Internal shape mismatch in DistPredHead forward: {proj_out.shape[-1]} vs {self.feature_size * self.num_outputs}")
        else:
            # Univariate case, shape should be [B, T, NumOutputs]
             if proj_out.shape[-1] == self.num_outputs:
                 return proj_out
             else:
                 # This shouldn't happen if __init__ validation passed
                 raise RuntimeError(f"Internal shape mismatch in DistPredHead forward: {proj_out.shape[-1]} vs {self.num_outputs}")

    def get_loss_fn(self) -> Optional[nn.Module]:
        """Returns None to indicate the head does not determine the loss."""
        return None # Signal to builder to use main config loss
