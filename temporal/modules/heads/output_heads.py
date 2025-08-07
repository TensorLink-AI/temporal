from typing import Dict, Optional
import torch
from temporal.modules.heads.base_output_head import BaseOutputHead
from temporal.modules.losses.timeflow_loss import TimeFlowLoss
from temporal.registry.core import register_module

@register_module("output_head", "timeflow")
class TimeFlowOutputHead(BaseOutputHead):
    """
    An output head that wraps the stateful TimeFlowLoss module.
    This head is responsible for orchestrating the TimeFlow model,
    which handles both loss calculation and sample generation internally.
    """
    def __init__(
        self,
        target_channels: int,
        cond_channels: int,
        num_blocks: int,
        model_channels: int,
        num_sampling_steps: int = 10,
        input_token_len: int = 1, # from Sundial
        diffusion_batch_mul: int = 1, # from Sundial
    ):
        super().__init__()
        self.input_token_len = input_token_len
        self.output_token_len = target_channels
        self.diffusion_batch_mul = diffusion_batch_mul

        self.timeflow_loss = TimeFlowLoss(
            target_channels=target_channels,
            cond_channels=cond_channels,
            num_blocks=num_blocks,
            model_channels=model_channels,
            num_sampling_steps=num_sampling_steps,
            reduction="none" 
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        mask_y: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        if labels is None:
            return {"loss": None, "preds": None}

        bsz_h, L_h, _ = hidden_states.shape
        
        shift_labels = labels.unfold(
            dimension=-1, size=self.output_token_len, step=self.input_token_len
        )
        bsz_l, L_l, _ = shift_labels.shape

        if L_h != L_l:
            raise ValueError(f"Mismatch between hidden states patches ({L_h}) and label patches ({L_l})")

        shift_labels = shift_labels.reshape(bsz_l * L_l, -1).repeat(self.diffusion_batch_mul, 1)
        cond = hidden_states.reshape(bsz_h * L_h, -1).repeat(self.diffusion_batch_mul, 1)

        loss = self.timeflow_loss(target=shift_labels, cond=cond, loss_mask=loss_mask, mask_y=mask_y)
        
        return {
            "loss": loss.mean(),
            "preds": None,
        }

    def sample(self, hidden_states: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        cond = hidden_states[:, -1, :]
        return self.timeflow_loss.sample(cond=cond, num_samples=num_samples)
