
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from typing import Optional
from temporal.registry.core import register_module

@register_module("normalization", "revin")
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time-series.
    It is described in https://openreview.net/forum?id=cGDAkQo1C0p

    This implementation assumes the input tensor is of shape (N, L, C), where
    N is the batch size, L is the sequence length, and C is the number of features.
    Normalization is applied over the L dimension.

    Args:
        num_features (int): The number of features or channels (C).
        eps (float): A value added for numerical stability. Default: 1e-5.
        affine (bool): If True, this module has learnable affine parameters. Default: True.
        subtract_last (bool): If True, subtracts the last element of the sequence from the input. Default: False.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Initialize affine parameters to be broadcastable for input of shape (N, L, C)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def _get_statistics(self, x, mask: Optional[torch.Tensor] = None):
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=1, keepdim=True).detach()
                self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach().clamp_min(self.eps)
            else:
                mask = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
                masked_sum = torch.sum(x * mask, dim=1, keepdim=True)
                num_non_masked = torch.sum(mask, dim=1, keepdim=True)

                all_pad = (num_non_masked == 0)
                num_non_masked = torch.clamp(num_non_masked, min=1)

                self.mean = (masked_sum / num_non_masked).detach()
                
                variance = torch.sum(((x - self.mean) * mask)**2, dim=1, keepdim=True) / num_non_masked
                self.stdev = torch.sqrt(variance + self.eps).detach().clamp_min(self.eps)

                self.mean  = torch.where(all_pad, torch.zeros_like(self.mean), self.mean)
                self.stdev = torch.where(all_pad, torch.ones_like(self.stdev), self.stdev)

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

    def transform(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Use stored statistics to normalize a new input
        if self.subtract_last:
            x_transformed = x - self.last
        else:
            x_transformed = x - self.mean
        
        x_transformed = x_transformed / self.stdev
        
        if self.affine:
            x_transformed = x_transformed * self.affine_weight
            x_transformed = x_transformed + self.affine_bias

        if mask is not None:
            mask_expanded = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            # Where mask is True, use transformed values. Where False, use original values.
            return torch.where(mask_expanded.bool(), x_transformed, x)
        
        return x_transformed


    def inverse_transform(self, x):
        return self._denormalize(x)

@register_module("normalization", "revin2d")
class RevIN2d(nn.Module):
    """
    Reversible Instance Normalization for time-series with 2D spatial dimensions.
    It is described in https://openreview.net/forum?id=cGDAkQo1C0p

    Args:
        num_features (int): The number of features or channels.
        eps (float): A value added for numerical stability. Default: 1e-5.
        affine (bool): If True, this module has learnable affine parameters. Default: True.
        subtract_last (bool): If True, subtracts the last element of the sequence from the input. Default: False.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Initialize affine parameters
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))

    def _get_statistics(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        dim2reduce = (2, 3) # H, W for a (N, C, H, W) tensor
        if self.subtract_last:
             raise NotImplementedError("`subtract_last` is not implemented for RevIN2d.")
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
                self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach().clamp_min(self.eps)
            else:
                mask = mask.to(dtype=x.dtype, device=x.device).unsqueeze(1)
                masked_sum = torch.sum(x * mask, dim=dim2reduce, keepdim=True)

                num_non_masked = torch.sum(mask, dim=dim2reduce, keepdim=True)
                all_pad = (num_non_masked == 0)
                num_non_masked = torch.clamp(num_non_masked, min=1)

                self.mean = (masked_sum / num_non_masked).detach()
                
                variance = torch.sum(((x - self.mean) * mask)**2, dim=dim2reduce, keepdim=True) / num_non_masked
                self.stdev = torch.sqrt(variance + self.eps).detach().clamp_min(self.eps)

                self.mean  = torch.where(all_pad, torch.zeros_like(self.mean), self.mean)
                self.stdev = torch.where(all_pad, torch.ones_like(self.stdev), self.stdev)


    def _normalize(self, x: torch.Tensor):
        if self.subtract_last:
            raise NotImplementedError("`subtract_last` is not implemented for RevIN2d.")
        
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            raise NotImplementedError("`subtract_last` is not implemented for RevIN2d.")
        else:
            x = x + self.mean
        return x

    def transform(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.subtract_last:
            raise NotImplementedError("`subtract_last` is not implemented for RevIN2d.")

        # Use stored statistics to normalize a new input
        x_transformed = x - self.mean
        x_transformed = x_transformed / self.stdev
        
        if self.affine:
            x_transformed = x_transformed * self.affine_weight
            x_transformed = x_transformed + self.affine_bias

        if mask is not None:
            # x is (N, C, H, W), mask is (N, H, W) -> expand to (N, 1, H, W)
            mask_expanded = mask.to(dtype=x.dtype, device=x.device).unsqueeze(1)
            return torch.where(mask_expanded.bool(), x_transformed, x)
        
        return x_transformed

    def inverse_transform(self, x):
        return self._denormalize(x)
