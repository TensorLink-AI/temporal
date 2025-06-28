# temporal/modules/feedforward/moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
# Assuming StandardFeedForward is the base for experts
from temporal.modules.feedforward.standard import StandardFeedForward, _resolve_activation


@register_module("feedforward", "moe")
class MoEFeedForward(nn.Module):
    """A Mixture of Experts (MoE) Feed-Forward Network layer.

    This module replaces a standard Feed-Forward Network (FFN) with multiple
    parallel FFN "experts". A gating network learns to route each input token
    to a small subset (the top-k) of these experts. The final output for each
    token is a weighted average of the outputs from its assigned experts.

    This architecture can significantly reduce computational cost while
    increasing model capacity, as only a fraction of the experts are activated
    for any given input.

    An auxiliary load balancing loss is computed during training to encourage
    the gating network to distribute tokens evenly across all experts,

    preventing a state where only a few experts are consistently chosen.

    Attributes:
        d_model (int): The hidden dimension of the model.
        num_experts (int): The total number of expert networks.
        top_k (int): The number of experts to route each token to.
        load_balancing_coef (float): The coefficient for the auxiliary loss.
        gate (nn.Linear): The gating network that determines expert weights.
        experts (nn.ModuleList): The list of expert FFNs.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        num_experts: int = 8,
        top_k: int = 2,
        expert_intermediate_size: Optional[int] = None,
        bias: bool = True,
        load_balancing_coef: float = 0.01,
        **kwargs
    ):
        """Initializes the MoEFeedForward module.

        Args:
            hidden_size (int): The main hidden dimension of the model (d_model).
            intermediate_size (Optional[int]): A fallback for the intermediate
                size of the expert FFNs.
            activation (str): The activation function to use within experts.
            dropout (float): The dropout rate for the expert FFNs.
            num_experts (int): The total number of experts.
            top_k (int): The number of experts to route each token to.
            expert_intermediate_size (Optional[int]): The specific intermediate
                size for the expert FFNs. Takes precedence over `intermediate_size`.
            bias (bool): Whether the expert FFNs should use a bias.
            load_balancing_coef (float): The coefficient for the load balancing loss.
            **kwargs: Catches any other unused arguments.
        """
        super().__init__()
        self.d_model = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_coef = load_balancing_coef

        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot be greater than num_experts ({num_experts}).")

        # Determine the intermediate size for the expert FFNs
        ffn_intermediate_size = expert_intermediate_size if expert_intermediate_size is not None else intermediate_size
        if ffn_intermediate_size is None:
             ffn_intermediate_size = self.d_model * 4
             print(f"Warning: MoE expert intermediate size not specified, defaulting to 4*d_model={ffn_intermediate_size}")

        # Gating network: Maps a token's embedding to a logit for each expert.
        self.gate = nn.Linear(self.d_model, num_experts, bias=False)

        # Create the pool of expert networks.
        self.experts = nn.ModuleList(
            [
                StandardFeedForward(
                    hidden_size=self.d_model,
                    intermediate_size=ffn_intermediate_size,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Performs the forward pass for the MoE FFN layer.

        Args:
            hidden_states (torch.Tensor): The input tensor of shape
                `[BatchSize, SequenceLength, HiddenDim]`.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - The final hidden states after MoE processing, with the same shape as the input.
                - The auxiliary load balancing loss (a scalar tensor) if in training
                  mode, otherwise None.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        # Reshape for easier processing: [B, T, D] -> [B*T, D]
        hidden_states_flat = hidden_states.view(-1, d_model)
        num_tokens = hidden_states_flat.shape[0]

        # 1. Get routing decisions from the gating network.
        # gate_logits shape: [B*T, num_experts]
        gate_logits = self.gate(hidden_states_flat)

        # Find the top-k experts for each token.
        # top_k_weights/indices shape: [B*T, top_k]
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1, sorted=False)

        # Normalize the weights of the selected experts.
        router_weights = F.softmax(top_k_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # 2. Calculate the auxiliary load balancing loss (during training only).
        aux_loss = None
        if self.training and self.load_balancing_coef > 0:
            # This loss encourages the router to use all experts equally.
            # It's based on the formulation from the Switch Transformer paper.
            router_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts)
            expert_mask_sum = expert_mask.sum(dim=1)

            tokens_per_expert_fraction = expert_mask_sum.sum(dim=0) / num_tokens
            router_prob_per_expert = (router_probs * expert_mask_sum).sum(dim=0)
            mean_router_prob_per_expert = router_prob_per_expert / (tokens_per_expert_fraction * num_tokens + 1e-6)

            load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert_fraction * mean_router_prob_per_expert)
            aux_loss = load_balancing_loss * self.load_balancing_coef

        # 3. Dispatch tokens to experts and compute the final output.
        final_hidden_states_flat = torch.zeros_like(hidden_states_flat)
        token_indices = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(self.top_k)
        expert_indices = top_k_indices.view(-1)
        flat_router_weights = router_weights.view(-1)

        # Process tokens expert by expert for clarity.
        for i, expert in enumerate(self.experts):
            # Find all routing entries that point to the current expert.
            expert_mask = (expert_indices == i)
            if not expert_mask.any():
                continue

            # Get the indices of the tokens that should be processed by this expert.
            tokens_for_this_expert_idx = token_indices[expert_mask]

            # Ensure indices are of type long for safe indexing, especially for index_add_
            long_indices = tokens_for_this_expert_idx.long()

            # Get the weights for these tokens.
            weights_for_this_expert = flat_router_weights[expert_mask].unsqueeze(1)

            # Get the hidden states of these tokens.
            hidden_states_for_this_expert = hidden_states_flat[long_indices]

            # Run the expert on its assigned tokens.
            expert_output = expert(hidden_states_for_this_expert)

            # Add the weighted expert output to the final result tensor.
            final_hidden_states_flat.index_add_(0, long_indices, expert_output * weights_for_this_expert)

        # Reshape the output back to the original input shape.
        final_hidden_states = final_hidden_states_flat.view_as(hidden_states)

        return final_hidden_states, aux_loss
