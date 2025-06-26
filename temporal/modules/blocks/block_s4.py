import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple 
from einops import rearrange

from temporal.registry.core import register_module
from temporal.modules.feedforward.standard import StandardFeedForward

@register_module("block", "s4")
class S4Layer(nn.Module):
    """
    A standalone S4D (Diagonal) layer.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        l_max: int = 1024,
        channels: int = 1,
        bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__()

        if d_model % channels != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by channels ({channels})")

        self.h = d_model // channels
        self.d_state = d_state
        self.l_max = l_max
        self.channels = channels
        self.bidirectional = bidirectional

        self._init_s4d_weights()

    def _init_s4d_weights(self):
        """Initialize the S4D-LegS parameters."""
        A, B = self.legs_init()
        C = torch.randn(self.channels, self.h, self.d_state)
        
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(torch.randn(self.channels, self.h))

        self.log_dt = nn.Parameter(torch.rand(self.channels, self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))

    def legs_init(self):
        """Initialize A and B based on Legendre polynomials (S4D-LegS)."""
        H = self.d_state
        A = torch.zeros(self.channels, self.h, H)
        B = torch.zeros(self.channels, self.h, H)
        
        q = torch.arange(H, dtype=torch.float)
        A_val = -0.5 * H * (H + 1) + q * (q + 1)
        B_val = (2 * q + 1) * (-1)**q
        
        A[:] = A_val
        B[:] = B_val
        
        return A, B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor `[batch_size, seq_len, d_model]`
        Returns:
            torch.Tensor: Output tensor `[batch_size, seq_len, d_model]`
        """
        L = x.size(1)

        dt = torch.exp(self.log_dt)
        A_bar, B_bar = self.discretize_zoh(self.A, self.B, dt)
        
        K = self.compute_kernel(A_bar, B_bar, self.C, L)
        if self.bidirectional:
            K_rev = self.compute_kernel(A_bar, B_bar, self.C, L)
            K = K + torch.fft.fft(torch.fft.ifft(K).flip(-1)).real


        y = self.fft_conv(x, K)
        y = y + rearrange(x, 'b l (c h) -> b l c h', c=self.channels) * self.D
        y = rearrange(y, 'b l c h -> b l (c h)')
        
        return y

    @staticmethod
    def discretize_zoh(A, B, dt):
        """Discretize continuous-time parameters A, B using Zero-Order Hold."""
        dt_A = dt.unsqueeze(-1) * A
        A_bar = torch.exp(dt_A)
        B_bar = torch.where(A == 0, dt.unsqueeze(-1), (A_bar - 1) / A) * B
        return A_bar, B_bar

    @staticmethod
    def compute_kernel(A_bar, B_bar, C, L: int):
        """Compute the S4 convolution kernel K."""
        powers = A_bar.unsqueeze(-1) * torch.arange(L, device=A_bar.device).view(1, 1, 1, -1)
        vandermonde = torch.exp(powers)
        K = torch.einsum('chd, chdl -> chl', C * B_bar, vandermonde)
        return K.real
    
    @staticmethod
    def fft_conv(u, K):
        """Perform convolution using FFT."""
        b, l, d = u.shape
        c, h, lk = K.shape
        
        u_c = rearrange(u, 'b l (c h) -> b c h l', c=c, h=h)
        L_fft = l + lk -1
        
        u_f = torch.fft.rfft(u_c, n=L_fft)
        K_f = torch.fft.rfft(K, n=L_fft)
        
        y_f = u_f * K_f
        y = torch.fft.irfft(y_f, n=L_fft)[..., :l]
        
        return rearrange(y, 'b c h l -> b l c h')


@register_module("block", "s4")
class S4Block(nn.Module):
    """
    A standard block that wraps the S4Layer with skip connections, normalization,
    and a feed-forward network. This structure is analogous to a Transformer block.

    Args:
        d_model (int): The main dimension of the model.
        d_state (int, optional): The latent state dimension of the S4 layer. Defaults to 64.
        ffn_dim (int, optional): The hidden dimension of the feed-forward network.
            If None, defaults to 4 * d_model.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        s4_kwargs (Dict[str, Any], optional): Additional arguments to pass to the S4Layer.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        s4_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        s4_kwargs = s4_kwargs or {}
        
        self.s4 = S4Layer(d_model=d_model, d_state=d_state, **s4_kwargs)
        self.ffn = StandardFeedForward(  # Changed arguments to match assumed StandardFeedForward __init__
            hidden_size=d_model,
            intermediate_size=ffn_dim or 4 * d_model,
            activation="gelu"
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Processes the input through the S4 block.

        Args:
            hidden_states (torch.Tensor): Input of shape `[batch, seq_len, d_model]`.
            **kwargs: Ignored, for API compatibility.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the output tensor of the same shape.
        """
        # First sub-layer: S4 layer
        residual = hidden_states
        x = self.norm1(hidden_states)
        x = self.s4(x)
        x = self.dropout1(x)
        x = x + residual

        # Second sub-layer: Feed-forward network
        residual = x
        y = self.norm2(x)
        y = self.ffn(y)
        y = self.dropout2(y)
        y = y + residual

        return (y,)
