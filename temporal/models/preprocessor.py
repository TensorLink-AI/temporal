
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_config import TransformerTimeSeriesConfig

class InputPreprocessor(nn.Module):
    """
    A module responsible for preprocessing raw inputs for a transformer model.

    This class encapsulates the entire input preparation pipeline, including:
    - Value embedding (with optional patching)
    - Positional embedding
    - Attention mask creation

    By centralizing this logic, it ensures that inputs are consistently
    prepared for both training/evaluation forward passes and for autoregressive
    generation.
    """

    def __init__(self, config: TransformerTimeSeriesConfig, builder: ModuleBuilder):
        """
        Initializes the InputPreprocessor.

        Args:
            config: The main model configuration object.
            builder: The module builder helper.
        """
        super().__init__()
        self.config = config
        self.value_embedding = builder.build_value_embedding()
        self.positional_embedding = builder.build_positional_embedding()
        self.layernorm_embedding = builder.build_normalization()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.is_patched = "Patch" in self.value_embedding.__class__.__name__
        self.patch_size = self.value_embedding.patch_size if self.is_patched else 1

    def process(
        self,
        input_values: torch.Tensor,
        past_key_values_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        validate_shapes: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes raw input tensors into embeddings and masks.

        Args:
            input_values (torch.Tensor): The raw input features, shape `[B, T, F]`.
            past_key_values_length (int): The length of the KV cache, used for
                creating the correct positional embeddings and attention masks.
            attention_mask (Optional[torch.Tensor]): A 2D padding mask.
            validate_shapes (bool): If True, performs assertions to check for
                shape consistency between value and positional embeddings.
            verbose (bool): If True, prints the shapes of tensors at each
                stage of the preprocessing pipeline for debugging.

        Returns:
            A dictionary containing:
            - `hidden_states` (torch.Tensor): The final processed embeddings.
            - `attention_mask` (torch.Tensor): The 4D attention mask for the model.
            - `patch_meta` (Dict): Metadata related to patching, if applicable.
        """
        if verbose: print(f"[Preprocessor] Initial input shape: {input_values.shape}")

        # --- Patching Compatibility: Pad if necessary ---
        if self.is_patched:
            current_seq_len = input_values.shape[1]
            remainder = current_seq_len % self.patch_size
            if remainder != 0:
                pad_len = self.patch_size - remainder
                input_values = nn.functional.pad(input_values, (0, 0, 0, pad_len))
                if verbose: print(f"[Preprocessor] Padded input shape for patching: {input_values.shape}")
        
        # --- Value and Positional Embedding ---
        value_embeds = self.value_embedding(input_values)
        if verbose: print(f"[Preprocessor] Value embedding shape: {value_embeds.shape}")
        
        batch_size, seq_len, _ = value_embeds.shape
        
        pos_embed = self.positional_embedding(
            batch_size=batch_size,
            seq_len=seq_len,
            past_key_values_length=past_key_values_length
        )
        if verbose: print(f"[Preprocessor] Positional embedding shape: {pos_embed.shape}")

        if validate_shapes:
            assert value_embeds.shape == pos_embed.shape, \
                f"Shape mismatch: value_embeds {value_embeds.shape} != pos_embed {pos_embed.shape}"

        hidden_states = value_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if verbose: print(f"[Preprocessor] Final hidden_states shape: {hidden_states.shape}")

        # --- Attention Mask Creation ---
        final_attention_mask = self._prepare_attention_mask(
            attention_mask,
            (batch_size, seq_len),
            hidden_states,
            past_key_values_length
        )

        return {
            "hidden_states": hidden_states,
            "attention_mask": final_attention_mask,
            "patch_meta": {"is_patched": self.is_patched, "patch_size": self.patch_size}
        }

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int
    ) -> Optional[torch.Tensor]:
        """
        Creates a 4D causal attention mask for a decoder, combining a causal
        mask with an optional padding mask.
        """
        bsz, seq_len = input_shape
        
        causal_mask = self._make_causal_mask(
            (bsz, seq_len),
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # The padding mask needs to be expanded to 4D to be combined with the causal mask.
            expanded_padding_mask = self._expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=seq_len
            ).to(inputs_embeds.device)
            # The final mask is the sum of the causal mask and the padding mask.
            causal_mask = expanded_padding_mask + causal_mask

        return causal_mask

    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0
    ) -> torch.Tensor:
        """
        Creates a lower-triangular causal mask for preventing attention to
        future tokens.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        
        # Use a standard and readable broadcasting approach to create the lower-triangular mask
        mask.masked_fill_(mask_cond[None, :] <= mask_cond[:, None], 0)
        
        if past_key_values_length > 0:
            # If a KV cache is used, the mask needs to be extended to accommodate the cached tokens.
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
            )
        # The mask is expanded to 4D to be broadcastable to the attention weights.
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(
        self,
        mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Expands a 2D padding mask to a 4D attention mask compatible with
        transformer layers.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        # The mask is inverted and filled with a large negative number where padded.
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
