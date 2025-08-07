import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig # Corrected import path
# Import moved to __init__ to break circular dependency
# from temporal.modules.embedders.embedding import TimeSeriesPatchEmbedding

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
        # Import here to avoid circular dependency
        from temporal.modules.embedders.embedding import TimeSeriesPatchEmbedding
        
        self.config = config
        # Pass value_embedding_config to builder
        self.value_embedding = builder.build_value_embedding(self.config.value_embedding_config)
        # Pass positional_embedding_config to builder
        self.positional_embedding = builder.build_positional_embedding(self.config.positional_embedding_config)
        # Pass norm_config to builder
        self.layernorm_embedding = builder.build_normalization(self.config.norm_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Determine if patching is enabled based on the type of value embedding or its config
        self.is_patched = isinstance(self.value_embedding, TimeSeriesPatchEmbedding)
        
        # Access patch_size and stride safely if patching is enabled
        self.patch_size = self.value_embedding.patch_size if self.is_patched else 1
        self.patch_stride = self.value_embedding.stride if self.is_patched else 1 
        self.embedding_pad_value = getattr(self.value_embedding, 'pad_value', 0.0)
        print(f"[InputPreprocessor __init__] self.patch_size: {self.patch_size}")

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
            # This logic correctly handles multivariate time series.
            # input_values shape: [B, T, F]
            
            # 1. Transpose to bring feature dimension forward for patching
            # Shape: [B, F, T]
            x = input_values.transpose(1, 2)

            # 2. Unfold along the time dimension (now dimension 2)
            # This creates overlapping windows (patches).
            # Shape: [B, F, NumPatches, PatchSize]
            patches = x.unfold(2, self.patch_size, self.patch_stride)
            if verbose: print(f"[Preprocessor] Unfolded patches shape: {patches.shape}")

            # 3. Permute to bring NumPatches forward and group patch dimensions
            # Shape: [B, NumPatches, F, PatchSize]
            patches = patches.permute(0, 2, 1, 3)

            # 4. Flatten the last two dimensions (F and PatchSize)
            # This creates the flat patch vector that the embedding layer expects.
            B, N, F, P = patches.shape
            input_for_embedding = patches.reshape(B, N, F * P)
            if verbose: print(f"[Preprocessor] Flattened patches for embedding shape: {input_for_embedding.shape}")

        else:
            # For non-patched models, pass the input directly.
            input_for_embedding = input_values

        # --- Value and Positional Embedding ---
        # ONLY pass input_for_embedding to value_embedding
        value_embeds = self.value_embedding(input_for_embedding)
        if verbose: print(f"[Preprocessor] Value embedding shape: {value_embeds.shape}")
        
        # The true sequence length for the transformer is derived from the value_embeds
        batch_size_embed, seq_len_after_patching, d_model = value_embeds.shape 
        
        # ONLY pass batch_size, seq_len, past_key_values_length to positional_embedding
        pos_embed = self.positional_embedding(
            batch_size=batch_size_embed,
            seq_len=seq_len_after_patching,
            past_key_values_length=past_key_values_length
        )
        if verbose: print(f"[Praeprocessor] Positional embedding shape: {pos_embed.shape}")

        if validate_shapes:
            assert value_embeds.shape == pos_embed.shape, \
                f"Shape mismatch: value_embeds {value_embeds.shape} != pos_embed {pos_embed.shape}"

        hidden_states = value_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if verbose: print(f"[Preprocessor] Final hidden_states shape: {hidden_states.shape}")

        final_attention_mask = self._prepare_attention_mask(
            attention_mask,
            (batch_size_embed, seq_len_after_patching),
            hidden_states,
            past_key_values_length,
            is_causal=is_causal,
        )
        return {
            "hidden_states": hidden_states,
            "attention_mask": final_attention_mask,
            "patch_meta": {"is_patched": self.is_patched, "patch_size": self.patch_size, "patch_stride": self.patch_stride}
        }

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        is_causal: bool,
    ) -> Optional[torch.Tensor]:
        """
        Creates a 4D attention mask, handling patching internally."""
        bsz, seq_len = input_shape
        final_mask = None
        
        if is_causal:
            final_mask = self._make_causal_mask((bsz, seq_len), inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length)
        
        if attention_mask is not None:
            if self.is_patched:
                # A patch is considered valid if ANY of its original time steps were valid.
                patch_mask_bool = attention_mask.unfold(1, self.patch_size, self.patch_stride).any(dim=-1)
                processed_mask = patch_mask_bool.to(attention_mask.dtype)
            else:
                processed_mask = attention_mask

            expanded_padding_mask = self._expand_mask(processed_mask, inputs_embeds.dtype, tgt_len=seq_len).to(inputs_embeds.device)
            final_mask = expanded_padding_mask if final_mask is None else final_mask + expanded_padding_mask

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

    def _prepare_decoder_inputs_for_generation(
            self,
            patch_embeds: torch.Tensor,
            past_key_values_length: int = 0,
            attention_mask: Optional[torch.Tensor] = None,
            is_causal: bool = True,
            verbose: bool = False
        ) -> Dict[str, Any]:
            """
            Prepares a single, already-embedded patch for the decoder during generation.

            This is a lightweight version of `process` that skips the value embedding/patching
            step and is used inside the autoregressive loop.

            Args:
                patch_embeds (torch.Tensor): The patch embedding from the previous generation step.
                                            Shape: `[B, 1, d_model]`.
                past_key_values_length (int): The length of the KV cache.
                attention_mask (Optional[torch.Tensor]): A 2D padding mask. Assumed to be `[B, 1]`.
                is_causal (bool): If True, a causal mask is created.

            Returns:
                A dictionary containing processed `hidden_states` and `attention_mask`.
            """
            if verbose: print(f"[Preprocessor Gen Step] Initial patch embed shape: {patch_embeds.shape}")
            
            # This function starts with embeddings, so it skips the `value_embedding` step.
            batch_size, seq_len, d_model = patch_embeds.shape

            # Add positional encoding for the current generation step.
            # `past_key_values_length` tells the positional embedding module *which* step we are on.
            pos_embed = self.positional_embedding(
                batch_size=batch_size,
                seq_len=seq_len,
                past_key_values_length=past_key_values_length
            )
            if verbose: print(f"[Preprocessor Gen Step] Positional embedding shape: {pos_embed.shape}")

            hidden_states = patch_embeds + pos_embed
            hidden_states = self.layernorm_embedding(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if verbose: print(f"[Preprocessor Gen Step] Final hidden_states shape: {hidden_states.shape}")
            
            # Create the appropriate attention mask for this single step.
            final_attention_mask = self._prepare_attention_mask(
                attention_mask,
                (batch_size, seq_len),
                hidden_states,
                past_key_values_length,
                is_causal=is_causal
            )
            if verbose and final_attention_mask is not None: 
                print(f"[Preprocessor Gen Step] Final attention mask shape (4D): {final_attention_mask.shape}")

            return {
                "hidden_states": hidden_states,
                "attention_mask": final_attention_mask,
            }
