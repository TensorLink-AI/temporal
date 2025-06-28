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
        gate_dropout: Optional[float] = None,
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
            gate_dropout (Optional[float]): Dropout rate to apply to the gate logits.
                                            If None or 0, no dropout is applied.
            **kwargs: Catches any other unused arguments.
        """
        super().__init__()
        self.d_model = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_coef = load_balancing_coef

        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot be greater than num_experts ({num_experts}).")

        ffn_intermediate_size = expert_intermediate_size if expert_intermediate_size is not None else intermediate_size
        if ffn_intermediate_size is None:
            ffn_intermediate_size = self.d_model * 4
            print(f"Warning: MoE expert intermediate size not specified, defaulting to 4*d_model={ffn_intermediate_size}")

        self.gate = nn.Linear(self.d_model, num_experts, bias=True)
        self.gate_dropout = nn.Dropout(gate_dropout) if gate_dropout is not None and gate_dropout > 0 else None

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
        hidden_states_flat = hidden_states.view(-1, d_model)
        num_tokens = hidden_states_flat.shape[0]

        # 1. Get routing decisions from the gating network.
        gate_logits = self.gate(hidden_states_flat)
        
        if self.training and self.gate_dropout is not None:
            gate_logits = self.gate_dropout(gate_logits)

        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1, sorted=False)
        router_weights = F.softmax(top_k_weights, dim=-1).to(hidden_states.dtype)

        # 2. Calculate the auxiliary load balancing loss (during training only).
        aux_loss = None
        if self.training and self.load_balancing_coef > 0:
            router_probs = F.softmax(gate_logits, dim=-1)
            
            expert_mask_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts)
            expert_mask_sum_per_token = expert_mask_one_hot.sum(dim=1)

            tokens_routed_to_expert = expert_mask_sum_per_token.sum(dim=0).float()
            sum_router_prob_for_expert = (router_probs * expert_mask_sum_per_token).sum(dim=0)

            fraction_tokens_routed_to_expert = tokens_routed_to_expert / num_tokens
            
            divisor = tokens_routed_to_expert + 1e-6
            mean_router_prob_per_expert = sum_router_prob_for_expert / divisor
            
            load_balancing_loss = self.num_experts * torch.sum(fraction_tokens_routed_to_expert * mean_router_prob_per_expert)
            aux_loss = load_balancing_loss * self.load_balancing_coef

        # 3. Dispatch tokens to experts and compute the final output.
        final_hidden_states_flat = torch.zeros_like(hidden_states_flat)
        
        for i, expert in enumerate(self.experts):
            mask_for_this_expert_in_topk = (top_k_indices == i)
            token_indices_for_this_expert_flat, top_k_pos_for_this_expert = torch.where(mask_for_this_expert_in_topk)

            if token_indices_for_this_expert_flat.numel() > 0:
                hidden_states_for_this_expert = hidden_states_flat[token_indices_for_this_expert_flat]
                weights_for_this_expert = router_weights[token_indices_for_this_expert_flat, top_k_pos_for_this_expert].unsqueeze(1)
                
                # --- START DEBUGGING PRINTS ---
                print(f"\n--- Debugging Expert {i} ---")
                print(f"hidden_states_for_this_expert shape: {hidden_states_for_this_expert.shape}, dtype: {hidden_states_for_this_expert.dtype}")
                print(f"weights_for_this_expert shape: {weights_for_this_expert.shape}, dtype: {weights_for_this_expert.dtype}")
                # --- END DEBUGGING PRINTS ---

                expert_output = expert(hidden_states_for_this_expert)
                
                # --- START DEBUGGING PRINTS ---
                print(f"expert_output shape: {expert_output.shape}, dtype: {expert_output.dtype}")
                # Check for unexpected scalar or integer types after expert call
                if expert_output.numel() == 1 and expert_output.dtype == torch.int:
                     print(f"!!! WARNING: expert_output is a single integer scalar: {expert_output}")
                if weights_for_this_expert.numel() == 1 and weights_for_this_expert.dtype == torch.int:
                     print(f"!!! WARNING: weights_for_this_expert is a single integer scalar: {weights_for_this_expert}")
                # --- END DEBUGGING PRINTS ---

                updates = expert_output * weights_for_this_expert
                
                final_hidden_states_flat.index_add_(0, token_indices_for_this_expert_flat.long(), updates)

        final_hidden_states = final_hidden_states_flat.view_as(hidden_states)

        return final_hidden_states, aux_loss
