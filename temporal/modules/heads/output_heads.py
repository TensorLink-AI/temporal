import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union # Added List, Dict, Union for MixtureOutputHead

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
   
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce the output from [B, T, K] or [B, T, F, K] → [B, T, F]
        using the median (middle quantile). This ensures the output is 
        compatible with decoder input: [B, T, feature_size]

        Args:
            x (torch.Tensor): output from self.forward()

        Returns:
            torch.Tensor: feedback tensor of shape [B, T, F]
        """
        if self.feature_size == 1:
            # x shape: [B, T, K]
            mid = self.num_outputs // 2
            return x[:, :, mid:mid+1]  # → [B, T, 1]
        else:
            # x shape: [B, T, F, K]
            mid = self.num_outputs // 2
            return x[:, :, :, mid]  # → [B, T, F]

    def get_loss_fn(self) -> Optional[nn.Module]:
        """Returns None to indicate the head does not determine the loss."""
        return None # Signal to builder to use main config loss

# ------------------------------------------------------ -
# Mixture Output Head
# ------------------------------------------------------ -
@register_module("output_head", "mixture")
class MixtureOutputHead(BaseOutputHead):
    """
    Output head for mixture density networks.
    Predicts parameters for a mixture of distributions.
    """
    # Define known parameters for each distribution type.
    # Each value is the number of raw outputs needed from the network for that param.
    DIST_PARAM_COUNTS = {
        "student_t": {"df": 1, "mu": 1, "scale": 1}, 
        "log_normal": {"mu": 1, "sigma": 1},       
        "neg_binomial": {"r": 1, "p": 1},          
        "normal": {"mu": 1, "sigma": 1},           
        "fixed_normal": {"mu": 1}                  
    }
    # Output names in the `preds` dict for the MixtureLoss
    DIST_OUTPUT_KEYS = {
        "student_t": {"df": "student_df", "mu": "student_mu", "scale": "student_scale"},
        "log_normal": {"mu": "lognorm_mu", "sigma": "lognorm_sigma"},
        "neg_binomial": {"r": "nb_r", "p": "nb_p"},
        "normal": {"mu": "normal_mu", "sigma": "normal_sigma"},
        "fixed_normal": {"mu": "normal_mu"}
    }

    def __init__(self, hidden_size: int, components: List[str], **kwargs):
        """
        Args:
            hidden_size (int): Input hidden dimension from the backbone.
            components (List[str]): List of distribution names for the mixture
                                    (e.g., ["student_t", "log_normal", "fixed_normal"]).
                                    These names must be keys in DIST_PARAM_COUNTS.
            **kwargs: Catches unused arguments. output_size from builder is not strictly needed
                      as it's derived, but can be passed for consistency checking.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.components = components
        self.num_components = len(components)

        if not self.num_components > 0:
            raise ValueError("MixtureOutputHead requires at least one component.")

        total_params_dim = 0
        self.param_indices = {} 
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
        
        self.mixture_logits_indices = (current_idx, current_idx + self.num_components)
        total_params_dim += self.num_components
        
        self.output_projection = nn.Linear(hidden_size, total_params_dim)
        
        if "output_size" in kwargs and kwargs["output_size"] != total_params_dim:
            raise ValueError(
                f"MixtureOutputHead: output_size kwarg ({kwargs['output_size']}) "
                f"does not match derived total_params_dim ({total_params_dim})."
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, HiddenSize].
        Returns:
            Dict[str, Union[torch.Tensor, List[str]]]: Dictionary of parameters for MixtureLoss.
        """
        B, T, _ = x.shape
        flat_params = self.output_projection(x)
        output_dict = {"components": self.components}
        logits_start, logits_end = self.mixture_logits_indices
        output_dict["mixture_logits"] = flat_params[..., logits_start:logits_end]
        
        processed_dist_params = set()
        for dist_name in self.components:
            param_config = self.DIST_PARAM_COUNTS[dist_name]
            output_key_map = self.DIST_OUTPUT_KEYS[dist_name]
            for param_internal_name, _ in param_config.items():
                actual_output_key = output_key_map[param_internal_name]
                if actual_output_key not in output_dict: 
                    start_idx, end_idx = self.param_indices[actual_output_key]
                    param_tensor = flat_params[..., start_idx:end_idx]
                    if param_tensor.shape[-1] == 1:
                        output_dict[actual_output_key] = param_tensor.squeeze(-1)
                    else:
                        output_dict[actual_output_key] = param_tensor
                    processed_dist_params.add(actual_output_key)
        return output_dict

    def get_loss_fn(self) -> Optional[nn.Module]:
        return None
