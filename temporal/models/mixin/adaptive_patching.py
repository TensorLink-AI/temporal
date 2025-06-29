
import torch
import torch.nn as nn


class PatchSplitting(nn.Module):
    """
    Adaptive Patching: Increase number of patches and decrease feature dim.

    If `use_mlp` is True, it applies a full 2-layer MLP to each patch's features
    before splitting them. This allows for a rich, non-linear transformation
    to prepare features for higher-resolution processing.
    """
    def __init__(
        self,
        input_dim: int,
        expansion_factor: int,
        use_mlp: bool = True, # Changed default to True for demonstration
        mlp_expansion_factor: int = 4,
    ):
        super().__init__()
        self.K = expansion_factor
        self.use_mlp = use_mlp
        self.input_dim = input_dim

        if self.use_mlp:
            # The MLP will transform the features before they are split.
            # It will take D-dimensional features and output D-dimensional features.
            mlp_hidden_dim = input_dim * mlp_expansion_factor

            self.norm = nn.LayerNorm(input_dim)
            # The projection is now a full 2-layer MLP
            self.projection = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=input_dim # Output dim is same as input
            )
        else:
            self.norm = None
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if D != self.input_dim:
             raise ValueError(f"Input feature dimension ({D}) does not match initialized `input_dim` ({self.input_dim})")
        if D % self.K != 0:
            raise ValueError(f"Feature dimension ({D}) must be divisible by expansion_factor ({self.K})")

        # Step 1 (Conditional): Apply the MLP to transform features
        if self.use_mlp:
            x = self.norm(x)
            x = self.projection(x)

        new_D = D // self.K

        # Step 2: Reshape to split features
        x = x.view(B, N, self.K, new_D)

        # Step 3: Reshape to create a longer sequence
        x = x.reshape(B, N * self.K, new_D)
        
        return x

class PatchMerging(nn.Module):
    """
    Merge patches back.

    If `use_mlp` is True, it now uses a full 2-layer MLP to learn a rich,
    non-linear transformation of the merged patch features.
    """
    def __init__(
        self,
        input_dim: int,
        merge_factor: int,
        use_mlp: bool = True,
        mlp_expansion_factor: int = 4, # Common expansion factor for the hidden layer
    ):
        super().__init__()
        self.K = merge_factor
        self.use_mlp = use_mlp
        self.input_dim = input_dim

        if self.use_mlp:
            # The input to the MLP is the concatenated feature dimension
            mlp_input_dim = input_dim * self.K
            
            # The hidden dimension is typically a multiple of the input dimension
            mlp_hidden_dim = mlp_input_dim * mlp_expansion_factor
            
            # The final output dimension after merging (a design choice)
            # Let's stick to 2 * original input_dim as before
            mlp_output_dim = 2 * input_dim

            self.norm = nn.LayerNorm(mlp_input_dim)
            # Instantiate our new 2-layer MLP
            self.reduction = MLP(
                input_dim=mlp_input_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=mlp_output_dim
            )
        else:
            self.norm = None
            self.reduction = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_K, D = x.shape
        if D != self.input_dim:
             raise ValueError(f"Input feature dimension ({D}) does not match initialized `input_dim` ({self.input_dim})")
        if N_K % self.K != 0:
            raise ValueError(f"Number of patches ({N_K}) must be divisible by merge_factor ({self.K})")

        N = N_K // self.K
        x = x.view(B, N, self.K, D)
        x = x.reshape(B, N, self.K * D)

        if self.use_mlp:
            x = self.norm(x)
            x = self.reduction(x) # Apply the full 2-layer MLP
            
        return x