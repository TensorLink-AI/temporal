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
    """
    Mixture of Experts Feed-Forward Network layer.

    Replaces a standard FFN with multiple parallel FFN experts and a gating
    network that routes tokens to a subset (top-k) of these experts.
    Includes a simplified auxiliary load balancing loss.
    """
    def __init__(
        self,
        # Arguments matching base_kwargs in ModuleBuilder.build_feedforward
        hidden_size: int, # This is d_model
        intermediate_size: Optional[int] = None, # Fallback if expert_intermediate_size not given
        activation: str = "gelu",
        dropout: float = 0.1,
        # MoE specific args expected from FeedForwardConfig user_kwargs
        num_experts: int = 8,
        top_k: int = 2,
        expert_intermediate_size: Optional[int] = None,
        bias: bool = True, # Bias for expert FFNs
        load_balancing_coef: float = 0.01,
        **kwargs # Catch any other unused args
    ):
        super().__init__()
        self.d_model = hidden_size # Use hidden_size as d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_coef = load_balancing_coef

        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot be greater than num_experts ({num_experts}).")

        # Determine intermediate size for experts
        ffn_intermediate_size = expert_intermediate_size if expert_intermediate_size is not None else intermediate_size
        if ffn_intermediate_size is None:
             # Default heuristic if neither intermediate_size nor expert_intermediate_size is given
             ffn_intermediate_size = self.d_model * 4
             print(f"Warning: MoE expert intermediate size not specified, defaulting to 4*d_model={ffn_intermediate_size}")

        # Gating network: Maps token embedding [D] to logits over experts [num_experts]
        self.gate = nn.Linear(self.d_model, num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList(
            [
                StandardFeedForward(
                    hidden_size=self.d_model, # Pass d_model here
                    intermediate_size=ffn_intermediate_size,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the MoE FFN layer.

        Args:
            hidden_states: Input tensor shape [BatchSize, SequenceLength, HiddenDim]

        Returns:
            Tuple containing:
                - final_hidden_states: Output tensor shape [BatchSize, SequenceLength, HiddenDim]
                - aux_loss: Scalar auxiliary load balancing loss (or None if not training/applicable).
                          Note: This loss should be scaled by the load_balancing_coef
                          outside this forward method, typically in the training loop.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, d_model) # Shape: [B*T, D]
        num_tokens = hidden_states_flat.shape[0]

        # 1. Calculate gate logits and select top-k experts
        # gate_logits shape: [B*T, num_experts]
        gate_logits = self.gate(hidden_states_flat)

        # Find top_k weights and indices per token
        # weights shape: [B*T, top_k]
        # indices shape: [B*T, top_k]
        # topk returns values and indices
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1, sorted=False)

        # Softmax over the top_k logits to get normalized weights for the chosen experts
        # Ensure float32 for softmax stability
        router_weights = F.softmax(top_k_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # 2. Calculate auxiliary load balancing loss (only during training)
        aux_loss = None
        if self.training and self.load_balancing_coef > 0:
            # Calculate softmax over *all* experts for router probabilities
            router_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float32) # [B*T, num_experts]

            # Calculate fraction of tokens assigned to each expert (f_i)
            # Create a mask indicating which experts were chosen (sum over top_k)
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts) # [B*T, top_k, num_experts]
            expert_mask_sum = expert_mask.sum(dim=1) # [B*T, num_experts], indicates if expert was chosen for token
            tokens_per_expert_fraction = expert_mask_sum.sum(dim=0) / num_tokens # [num_experts]

            # Calculate average router probability for tokens routed to each expert (P_i)
            # Sum router probs for chosen experts, divide by tokens routed to that expert
            router_prob_per_expert = (router_probs * expert_mask_sum).sum(dim=0) # Sum of probs for tokens routed to expert i
            # Avoid division by zero if an expert gets no tokens
            mean_router_prob_per_expert = router_prob_per_expert / (tokens_per_expert_fraction * num_tokens + 1e-6)

            # Aux loss: num_experts * sum(f_i * P_i) across experts (simplified from Switch Transformers)
            # Aim is to balance f_i and keep P_i somewhat high (encourage confident routing)
            # A common formulation is based on the dot product of these two vectors, scaled.
            load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert_fraction * mean_router_prob_per_expert)

            aux_loss = load_balancing_loss * self.load_balancing_coef # Apply coefficient here or in training loop


        # 3. Compute expert outputs and combine them
        # Create a flat output tensor initialized to zeros
        final_hidden_states_flat = torch.zeros_like(hidden_states_flat)

        # Create a combined index for efficient processing if possible (might need sparse ops for large scale)
        # Expand top_k indices and router weights to match the flattened token dimension
        # Example using loops (less efficient but clear): Needs optimization for performance
        token_indices = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(self.top_k)
        expert_indices = top_k_indices.view(-1) # Flat list of expert indices [B*T*top_k]
        flat_router_weights = router_weights.view(-1) # Flat list of weights [B*T*top_k]

        # Iterate through experts, process tokens routed to them
        for i, expert in enumerate(self.experts):
            # Find which entries in the flat list correspond to this expert
            expert_mask = (expert_indices == i)
            if expert_mask.any():
                # Get the original token indices for these entries
                original_token_idx = token_indices[expert_mask]
                # Get the weights for these entries
                current_weights = flat_router_weights[expert_mask].unsqueeze(1) # [num_tokens_for_expert, 1]
                # Get the hidden states for these tokens
                current_hidden_states = hidden_states_flat[original_token_idx]

                # Pass tokens through the expert
                expert_output = expert(current_hidden_states)

                # Add the weighted expert output to the final output tensor using index_add_
                # index_add_(dim, index, tensor_to_add)
                final_hidden_states_flat.index_add_(0, original_token_idx, expert_output * current_weights)

        # Reshape back to the original sequence format
        final_hidden_states = final_hidden_states_flat.view_as(hidden_states)

        return final_hidden_states, aux_loss
