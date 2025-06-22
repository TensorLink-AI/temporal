
import torch
import torch.nn as nn

class AdaptivePatching(nn.Module):
    """
    Adaptive Patching: Increase number of patches and decrease feature dim.

    Args:
        expansion_factor (int): The factor K_i to expand patches and reduce feature dim.
    """
    def __init__(self, expansion_factor: int):
        super().__init__()
        self.K = expansion_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, D]
        Returns:
            Tensor of shape [B, N * K, D // K]
        """
        B, N, D = x.shape
        if D % self.K != 0:
            raise ValueError(f"Feature dimension ({D}) must be divisible by expansion_factor ({self.K})")
        
        new_D = D // self.K

        # Reshape [B, N, D] -> [B, N * K, D // K]
        x = x.view(B, N, self.K, new_D)
        x = x.reshape(B, N * self.K, new_D)
        return x


class PatchMerging(nn.Module):
    """
    Merge patches back: Decrease number of patches and increase feature dim.

    Args:
        merge_factor (int): The factor K_i used in patching.
    """
    def __init__(self, merge_factor: int):
        super().__init__()
        self.K = merge_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N * K, D]
        Returns:
            Tensor of shape [B, N, D * K]
        """
        B, N_K, D = x.shape
        if N_K % self.K != 0:
            raise ValueError(f"Number of patches ({N_K}) must be divisible by merge_factor ({self.K})")
        
        N = N_K // self.K
        x = x.view(B, N, self.K, D)
        x = x.reshape(B, N, D * self.K)
        return x
