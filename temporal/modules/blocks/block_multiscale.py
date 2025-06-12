import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from temporal.registry.core import register_module
from typing import Optional, Tuple

@register_module("block", "multiscale")
class MultiScaleBlock(nn.Module):
    """A multi-scale attention block that processes sequences at two resolutions.

    This block applies attention at two different scales:
    1.  **Fine Scale:** The input sequence is processed at its original,
        full resolution.
    2.  **Coarse Scale:** The input sequence is downsampled, processed by a
        separate attention module, and then upsampled back to the original
        resolution.

    The outputs from both scales are then fused together using a specified
    fusion method ('add', 'concat', or 'gate').

    Attributes:
        fine_attn (nn.Module): The attention module for the fine scale.
        coarse_attn (nn.Module): The attention module for the coarse scale.
        downsample_factor (int): The factor by which to downsample the sequence
            for the coarse scale.
        fusion_method (str): The method used to fuse the outputs of the two scales.
        fuse_proj (Optional[nn.Linear]): A linear layer for the 'concat' fusion method.
        gate (Optional[nn.Sequential]): A gating mechanism for the 'gate' fusion method.
    """

    def __init__(
        self,
        fine_attn: nn.Module,
        coarse_attn: nn.Module,
        downsample_factor: int = 4,       # The factor for downsampling
        fusion_method: str = "add",       # 'add', 'concat', or 'gate'
        hidden_size: int = 128,           # The hidden dimension size
        **kwargs,                         # For any other arguments
    ):
        """Initializes the MultiScaleBlock.

        Args:
            fine_attn (nn.Module): The attention module for the fine-scale processing.
            coarse_attn (nn.Module): The attention module for the coarse-scale processing.
            downsample_factor (int): The factor by which to downsample the sequence.
            fusion_method (str): The method for fusing the fine and coarse outputs.
                Supported methods are 'add', 'concat', and 'gate'.
            hidden_size (int): The hidden dimension of the input and output tensors.
                This is required for initializing projection layers for certain
                fusion methods.
            **kwargs: Additional keyword arguments (not directly used in this class).
        """
        super().__init__()
        self.fine_attn = fine_attn
        self.coarse_attn = coarse_attn
        self.downsample_factor = downsample_factor
        self.fusion_method = fusion_method

        if fusion_method == "concat":
            self.fuse_proj: Optional[nn.Linear] = nn.Linear(2 * hidden_size, hidden_size)
            self.gate: Optional[nn.Sequential] = None
        elif fusion_method == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
            self.fuse_proj = None
        else:
            self.fuse_proj = None
            self.gate = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # Mask for fine-grained attention
        output_attentions: Optional[bool] = False, # Whether to return attention weights
        **kwargs, # Pass-through arguments for attention modules
    ) -> Tuple[torch.Tensor, Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]]:
        """Performs the forward pass of the MultiScaleBlock.

        Args:
            hidden_states (torch.Tensor): The input tensor of shape `[B, L, D]`.
            attention_mask (Optional[torch.Tensor]): The attention mask for the
                fine-scale attention. Defaults to None.
            output_attentions (Optional[bool]): If True, returns the attention
                weights from both scales. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the attention modules.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]]:
                - The fused output tensor of shape `[B, L, D]`.
                - A tuple containing the attention weights from the fine and coarse
                  scales, if `output_attentions` is True.
        """
        B, L, D = hidden_states.shape
        scale_factor = self.downsample_factor

        # === Step 1: Fine-scale attention ===
        fine_output, fine_attention_weights, _ = self.fine_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )

        # === Step 2: Downsample for coarse-scale ===
        pad_len = (scale_factor - (L % scale_factor)) % scale_factor
        if pad_len > 0:
            padded_hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
        else:
            padded_hidden_states = hidden_states

        hidden_grouped = rearrange(padded_hidden_states, 'b (k s) d -> b k s d', s=scale_factor)
        coarse_scale_input = hidden_grouped.mean(dim=2)

        # === Step 3: Coarse attention ===
        coarse_output, coarse_attention_weights, _ = self.coarse_attn(
            hidden_states=coarse_scale_input,
            attention_mask=None,  # No mask needed for the downsampled sequence
            output_attentions=output_attentions,
            **kwargs
        )

        # === Step 4: Upsample coarse output ===
        coarse_upsampled = coarse_output.repeat_interleave(scale_factor, dim=1)
        if pad_len > 0:
            coarse_upsampled = coarse_upsampled[:, :L, :]

        # === Step 5: Fuse outputs ===
        if self.fusion_method == "add":
            fused_output = fine_output + coarse_upsampled
        elif self.fusion_method == "concat":
            if self.fuse_proj is None:
                raise RuntimeError("Fuse projection layer is not initialized for 'concat' method.")
            fused_output = self.fuse_proj(torch.cat([fine_output, coarse_upsampled], dim=-1))
        elif self.fusion_method == "gate":
            if self.gate is None:
                raise RuntimeError("Gating mechanism is not initialized for 'gate' method.")
            fusion_input = torch.cat([fine_output, coarse_upsampled], dim=-1)
            gate_values = self.gate(fusion_input)
            fused_output = gate_values * fine_output + (1 - gate_values) * coarse_upsampled
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        attention_weights = (fine_attention_weights, coarse_attention_weights) if output_attentions else None
        return fused_output, attention_weights
