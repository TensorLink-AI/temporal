import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig 



class InputPreprocessor(nn.Module):
    """
    A module responsible for preprocessing raw inputs for a transformer model.

    This class encapsulates the entire input preparation pipeline, including:
    - Optional instance normalization (e.g., RevIN).
    - Value embedding (with optional patching).
    - Positional embedding.
    - Layer normalization and dropout.
    - Attention mask creation (padding and optional causal).

    Attention Mask Convention:
    - Input `attention_mask` (2D): `1` for valid (attendable) tokens, `0` for padded (masked out) tokens.
    - Output `attention_mask` (4D): `0` for valid tokens, `-inf` for masked out tokens.
    """

    def __init__(self, config: TransformerTimeSeriesConfig, builder: ModuleBuilder):
        """
        Initializes the InputPreprocessor.

        Args:
            config: The main model configuration object.
            builder: The module builder helper.
        """
        super().__init__()
        from temporal.modules.embedders.embedding import TimeSeriesPatchEmbedding
        
        self.config = config
        
        # --- Instance Normalization (e.g., RevIN) ---
        self.instance_norm = None
        if config.instance_norm_config:
            self.instance_norm = builder.build_normalization(config.instance_norm_config)
            
        self.value_embedding = builder.build_value_embedding(self.config.value_embedding_config)
        self.positional_embedding = builder.build_positional_embedding(self.config.positional_embedding_config)
        self.layernorm_embedding = builder.build_normalization(self.config.layer_norm_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.is_patched = isinstance(self.value_embedding, TimeSeriesPatchEmbedding)
        # These are now correctly inferred from the embedding layer itself if needed
        self.patch_size = getattr(self.value_embedding, 'patch_size', 1)
        self.patch_stride = getattr(self.value_embedding, 'stride', 1)

    def process(
        self,
        input_values: torch.Tensor,
        past_key_values_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        validate_shapes: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes raw input tensors into embeddings and masks.
        """
        if verbose: print(f"[Preprocessor] Initial input shape: {input_values.shape}")

        # --- Apply Instance Normalization if configured ---
        if self.instance_norm is not None:
            input_values = self.instance_norm(input_values, mode='norm', mask=attention_mask)
        
        # --- Value Embedding (Handles Patching Internally) ---
        # FIX: Removed the buggy manual patching logic. The embedding layer now handles this.
        value_embeds = self.value_embedding(input_values)
        if verbose: print(f"[Preprocessor] Value embedding shape: {value_embeds.shape}")

        batch_size_embed, seq_len_after_patching, d_model = value_embeds.shape 
        
        # --- Positional Embedding ---
        # FIX: Pass the value_embeds tensor to positional embedding. This is more robust
        # for device placement and for embeddings that modify the input tensor directly.
        # ACTION: Standardized the call to self.positional_embedding for consistency.
        pos_embed = self.positional_embedding(
            x=value_embeds,
            batch_size=batch_size_embed,
            seq_len=seq_len_after_patching,
            past_key_values_length=past_key_values_length
        )
        pos_embed= pos_embed.to(dtype=value_embeds.dtype, device=value_embeds.device)

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
            (batch_size_embed, seq_len_after_patching),
            hidden_states,
            past_key_values_length,
            is_causal=is_causal,
        )
        return {
            "hidden_states": hidden_states,
            "attention_mask": final_attention_mask,
        }
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverses the instance normalization if it was applied.
        """
        if self.instance_norm is not None:
            return self.instance_norm(data, mode='denorm')
        return data

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        is_causal: bool,
    ) -> Optional[torch.Tensor]:
        bsz, seq_len = input_shape
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        final_mask = None

        # 1) Causal mask is always built to total K length
        if is_causal:
            final_mask = self._make_causal_mask(
                (bsz, seq_len),
                dtype,
                device,
                past_key_values_length=past_key_values_length,
            )  # [B,1,T_q,(past+T_q)]

        if attention_mask is not None:
            processed_mask = attention_mask

            # (a) If you used patching for values, downsample the attention mask
            if self.is_patched:
                # [B, L] -> [B, L_patches]
                processed_mask = processed_mask.unfold(1, self.patch_size, self.patch_stride).any(dim=-1)
                processed_mask = processed_mask.to(attention_mask.dtype)

            # (b) Make sure mask lives with embeds
            processed_mask = processed_mask.to(device=device, dtype=inputs_embeds.dtype)

            # (c) ***CRITICAL***: when caching, pad LEFT with 1s for past tokens so src_len==past+T_q
            if is_causal and past_key_values_length > 0:
                # mask convention here: 1 = valid, 0 = padding
                processed_mask = torch.nn.functional.pad(
                    processed_mask, (past_key_values_length, 0), value=1.0
                )

            # (d) Expand to 4D; tgt_len stays T_q, src_len is now total_k_len
            expanded_padding_mask = self._expand_mask(
                processed_mask, dtype=dtype, tgt_len=seq_len
            ).to(device)

            final_mask = expanded_padding_mask if final_mask is None else (final_mask + expanded_padding_mask)

        return final_mask




    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0
    ) -> torch.Tensor:
        """
        Creates a lower-triangular causal mask correctly handling past_key_values.
        This is a robust implementation that directly builds the correct mask shape.
        """
        bsz, tgt_len = input_ids_shape
        total_len = tgt_len + past_key_values_length

        # Create indices for the target (query) and source (key) sequence lengths
        q_indices = torch.arange(tgt_len, device=device).view(tgt_len, 1)
        k_indices = torch.arange(total_len, device=device).view(1, total_len)

        # The query position index must be offset by the past length
        # to correctly compare with the key indices.
        # Condition for masking: key_position > query_position
        mask_cond = k_indices > (q_indices + past_key_values_length)

        # Create a float mask where masked positions are -inf and unmasked are 0.0
        mask = torch.where(mask_cond, torch.finfo(dtype).min, 0.0)

        # Expand to the 4D shape required for attention: [B, H, T_q, T_k]
        # The head dimension (H) will be broadcasted automatically.
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, total_len)

    def _expand_mask(
        self,
        mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Expands a 2D padding mask to a 4D attention mask.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
        inverted_mask = (1.0 - expanded_mask).to(dtype)
        
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
            """
            if verbose: print(f"[Preprocessor Gen Step] Initial patch embed shape: {patch_embeds.shape}")
            
            batch_size, seq_len, d_model = patch_embeds.shape

            pos_embed = self.positional_embedding(
                x=patch_embeds,
                batch_size=batch_size,
                seq_len=seq_len,
                past_key_values_length=past_key_values_length
            )
            if verbose: print(f"[Preprocessor Gen Step] Positional embedding shape: {pos_embed.shape}")

            hidden_states = patch_embeds + pos_embed
            hidden_states = self.layernorm_embedding(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if verbose: print(f"[Preprocessor Gen Step] Final hidden_states shape: {hidden_states.shape}")
            
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