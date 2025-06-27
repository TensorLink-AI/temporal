
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

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time-series.
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

    def forward(self, x: torch.Tensor, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Initialize affine parameters
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(2, x.ndim))
        if self.subtract_last:
            self.last = x[:,:,-1].unsqueeze(-1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

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

    def forward(self, x: torch.Tensor, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
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

    def _get_statistics(self, x):
        dim2reduce = tuple(range(2, x.ndim))
        if self.subtract_last:
            self.last = x[:,:,-1].unsqueeze(-1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

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
