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
    - Attention mask creation (padding and optional causal)

    Attention Mask Convention:
    - Input `attention_mask` (2D): `1` for valid (attendable) tokens, `0` for padded (masked out) tokens.
    - Output `attention_mask` (4D): `0` for valid tokens, `-inf` (torch.finfo(dtype).min) for masked out tokens.
      This mask is designed to be added to attention scores before softmax.
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
        # It's assumed TimeSeriesPatchEmbedding has a 'stride' attribute.
        self.patch_stride = self.value_embedding.stride if self.is_patched else 1 
        # It's assumed TimeSeriesPatchEmbedding has a 'pad_value' attribute.
        self.embedding_pad_value = getattr(self.value_embedding, 'pad_value', 0.0)


    def process(
        self,
        input_values: torch.Tensor,
        past_key_values_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None, # 1 for valid, 0 for padded (original length)
        is_causal: bool = False,
        validate_shapes: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes raw input tensors into embeddings and masks.

        Args:
            input_values (torch.Tensor): The raw input features, shape `[B, T, F]`.
            past_key_values_length (int): The length of the KV cache.
            attention_mask (Optional[torch.Tensor]): A 2D padding mask. Shape `[B, T_original]`,
                                                    where `1` indicates a valid token and `0` indicates padding.
            is_causal (bool): If True, a causal mask is created and combined with the padding mask.
            validate_shapes (bool): If True, performs shape assertions.
            verbose (bool): If True, prints shape information for debugging.

        Returns:
            A dictionary containing processed `hidden_states` and `attention_mask`.
        """
        if verbose: print(f"[Preprocessor] Initial input shape: {input_values.shape}")

        batch_size, original_seq_len, num_features = input_values.shape
        
        # --- Patching Compatibility: Pad input_values if necessary ---
        # This padding ensures input_values length is suitable for TimeSeriesPatchEmbedding's unfold operation.
        # TimeSeriesPatchEmbedding also has internal padding logic, but this pre-padding helps align.
        pad_len_for_value_embedding = 0
        if self.is_patched:
            remainder = original_seq_len % self.patch_stride # Use stride for padding calculation
            if remainder != 0:
                # Calculate padding needed to make it compatible with stride for unfold
                # (L - patch_size) % stride == 0 -> L_new = patch_size + k*stride
                # To get enough patches, L_new >= original_seq_len.
                # Simplest way is to pad such that (L - self.patch_size) is a multiple of stride
                
                # The length after padding must be `P + k*S` for some integer k, where P is patch_size, S is stride.
                # If L is current length, we need L_target such that L_target >= L and (L_target - P) % S == 0.
                
                # A robust way is to compute the number of patches needed, then the total length.
                # num_patches_needed = (original_seq_len - self.patch_size + self.patch_stride - 1) // self.patch_stride + 1
                # If original_seq_len is already less than patch_size, we need to pad to patch_size.
                if original_seq_len < self.patch_size:
                    min_len_for_patches = self.patch_size
                else:
                    # Calculate how many full strides beyond the first patch are needed
                    required_strides = (original_seq_len - self.patch_size + self.patch_stride - 1) // self.patch_stride
                    min_len_for_patches = self.patch_size + required_strides * self.patch_stride
                
                pad_len_for_value_embedding = min_len_for_patches - original_seq_len
                
                if pad_len_for_value_embedding > 0:
                    input_values = nn.functional.pad(
                        input_values, (0, 0, 0, pad_len_for_value_embedding), value=self.embedding_pad_value
                    )
                    if verbose: print(f"[Preprocessor] Padded input for patching (pre-value_embedding): {input_values.shape}")
        
        # --- Value and Positional Embedding ---
        value_embeds = self.value_embedding(input_values) # TimeSeriesPatchEmbedding handles its own internal padding if needed
        if verbose: print(f"[Preprocessor] Value embedding shape: {value_embeds.shape}")
        
        # The true sequence length for the transformer is derived from the value_embeds
        batch_size_embed, seq_len_after_patching, d_model = value_embeds.shape 
        
        pos_embed = self.positional_embedding(
            batch_size=batch_size_embed,
            seq_len=seq_len_after_patching,
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
        processed_attention_mask = None
        if attention_mask is not None:
            # 1. Convert original attention_mask to boolean for unfolding
            # Assume attention_mask is 1 for valid, 0 for padded
            bool_mask = attention_mask.to(torch.bool) # Shape [B, T_original]

            # 2. Pad the boolean mask identically to how input_values was padded before value_embedding
            if pad_len_for_value_embedding > 0:
                pad_zeros = torch.zeros(batch_size, pad_len_for_value_embedding, 
                                        device=attention_mask.device, dtype=torch.bool)
                bool_mask = torch.cat([bool_mask, pad_zeros], dim=1) # Shape [B, T_padded_pre_embedding]
            
            # 3. Apply unfold to the padded boolean mask
            # This simulates the patching on the mask
            # The length of bool_mask should now match the length of input_values passed to value_embedding
            if bool_mask.shape[1] < self.patch_size:
                # If the sequence is shorter than a patch, it's a single "patch" (which might be entirely padded)
                # In this case, unfolding will fail or yield unexpected results.
                # A single patch is formed, its validity depends on the original data.
                # Simplification: If all original tokens were masked, the patch is masked.
                patch_mask_bool = bool_mask.any(dim=-1).unsqueeze(-1) # [B, 1]
            else:
                patch_bool = bool_mask.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
                # Shape: [B, num_patches, patch_size]
            
                # 4. A patch is "valid" if **any** of its positions are valid (True)
                patch_mask_bool = patch_bool.any(dim=-1) # Shape: [B, num_patches]

            # 5. Convert back to the original attention_mask's dtype
            processed_attention_mask = patch_mask_bool.to(attention_mask.dtype)
            
            # Runtime validation: Ensure the generated mask length matches the embedded sequence length
            if validate_shapes:
                assert processed_attention_mask.shape[1] == seq_len_after_patching, \
                    f"Mismatch between generated patch mask length ({processed_attention_mask.shape[1]}) " \
                    f"and embedded sequence length ({seq_len_after_patching}). Check patching logic."
            
            if verbose: print(f"[Preprocessor] Processed attention mask shape (2D): {processed_attention_mask.shape}")
            
        final_attention_mask = self._prepare_attention_mask(
            processed_attention_mask, # This mask is now correctly aligned with seq_len_after_patching
            (batch_size_embed, seq_len_after_patching), # Use actual embedded sequence dimensions
            hidden_states,
            past_key_values_length,
            is_causal=is_causal,
        )
        if verbose and final_attention_mask is not None: 
            print(f"[Preprocessor] Final attention mask shape (4D): {final_attention_mask.shape}")

        return {
            "hidden_states": hidden_states,
            "attention_mask": final_attention_mask,
            "patch_meta": {"is_patched": self.is_patched, "patch_size": self.patch_size, "patch_stride": self.patch_stride}
        }

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor], # 2D mask (1=valid, 0=padded) aligned with embedded sequence length
        input_shape: Tuple[int, int], # (batch_size, sequence_length_after_embedding)
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        is_causal: bool,
    ) -> Optional[torch.Tensor]:
        """
        Creates a 4D attention mask, optionally causal.

        Args:
            attention_mask (Optional[torch.Tensor]): A 2D padding mask. Shape `[B, S_embedded]`,
                                                    where `1` indicates a valid token and `0` indicates padding.
                                                    This mask should already be aligned with `input_shape[1]`.
            input_shape (Tuple[int, int]): The shape of the sequence (batch_size, sequence_length_after_embedding).
            inputs_embeds (torch.Tensor): The input embeddings (used for dtype and device).
            past_key_values_length (int): The length of the past key values.
            is_causal (bool): Whether to create a causal mask.

        Returns:
            Optional[torch.Tensor]: The final 4D attention mask (0 for valid, -inf for masked).
        """
        bsz, seq_len = input_shape
        final_mask = None

        if is_causal:
            final_mask = self._make_causal_mask(
                (bsz, seq_len), 
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # Validate that the provided attention_mask matches the expected sequence length
            if attention_mask.shape[1] != seq_len:
                 raise ValueError(
                    f"Input `attention_mask` has length {attention_mask.shape[1]} "
                    f"but expected length {seq_len} based on embedded sequence. "
                    "Ensure `attention_mask` is correctly prepared (e.g., via `unfold` if patching) "
                    "before being passed to `_prepare_attention_mask`."
                )

            # Convert 2D (1=valid, 0=padded) mask to 4D (0=valid, -inf=padded) mask
            expanded_padding_mask = self._expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=seq_len
            ).to(inputs_embeds.device)
            
            if final_mask is None:
                final_mask = expanded_padding_mask
            else:
                # Combine causal mask (0 or -inf) with padding mask (0 or -inf)
                # Adding them correctly combines the masking effects (max(-inf, -inf) = -inf; max(0, -inf) = 0)
                final_mask = final_mask + expanded_padding_mask

        return final_mask

    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0
    ) -> torch.Tensor:
        """
        Creates a lower-triangular causal mask for preventing attention to
        future tokens. Returns a mask where `0` means attend and `-inf` means mask.
        """
        bsz, tgt_len = input_ids_shape
        # Initialize with -inf (mask out)
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        # Create a condition for lower-triangular part (including diagonal)
        mask_cond = torch.arange(mask.size(-1), device=device)
        # Fill lower-triangular part with 0 (attend)
        mask.masked_fill_(mask_cond[None, :] <= mask_cond[:, None], 0)
        
        if past_key_values_length > 0:
            # If there's a KV cache, prepend zeros (attend to past tokens)
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
            )
        
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(
        self,
        mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Expands a 2D padding mask to a 4D attention mask compatible with
        transformer layers.
        
        Args:
            mask (torch.Tensor): A 2D mask of shape `[B, src_len]` where `1` indicates a valid token
                                 (to be attended to) and `0` indicates a padded token (to be masked out).
            dtype (torch.dtype): The desired data type for the output mask (e.g., torch.float).
            tgt_len (Optional[int]): The target sequence length. If None, defaults to `src_len`.
                                      This allows the mask to be expanded for cross-attention if needed.

        Returns:
            torch.Tensor: A 4D mask of shape `[B, 1, tgt_len, src_len]`, where `0` means attend and
                          `torch.finfo(dtype).min` means mask out. This is suitable for adding to attention scores.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        
        # Expand the 2D mask to 4D: [B, 1, 1, src_len]
        expanded_mask = mask[:, None, None, :]
        
        # Expand to [B, 1, tgt_len, src_len] to allow broadcasting across attention heads and target sequence
        expanded_mask = expanded_mask.expand(bsz, 1, tgt_len, src_len)
        
        # Convert the mask from (1=valid, 0=padded) to (0=valid, large_negative=padded)
        # This is standard for attention masks where adding large negative values effectively "zeros out" attention scores
        # after softmax.
        inverted_mask = (1.0 - expanded_mask).to(dtype) # 1 where padded, 0 where valid
        
        # Fill the padded positions with a large negative number
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)