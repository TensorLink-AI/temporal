import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from temporal.models.module_builder_helper import ModuleBuilder
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig


class InputPreprocessor(nn.Module):
    """
    Full input preparation:
      - Optional RevIN (instance norm)
      - Value embedding (patch or token)
      - Positional embedding
      - LayerNorm + Dropout
      - Attention mask build (padding + optional causal, cache-safe)

    Conventions:
      * Input 2D masks: 1=valid, 0=pad
      * Output attn masks: 0=valid, -inf=masked (broadcasted to [B, H=1, Tq, Tk])
    """
    def __init__(self, config: TransformerTimeSeriesConfig, builder: ModuleBuilder):
        super().__init__()
        from temporal.modules.embedders.embedding import TimeSeriesPatchEmbedding

        self.config = config

        # RevIN (optional)
        self.instance_norm = None
        if config.instance_norm_config:
            self.instance_norm = builder.build_normalization(config.instance_norm_config)

        # Embeddings
        self.value_embedding = builder.build_value_embedding(self.config.value_embedding_config)
        self.positional_embedding = builder.build_positional_embedding(self.config.positional_embedding_config)
        self.layernorm_embedding = builder.build_normalization(self.config.layer_norm_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.quantizer: Optional[BaseQuantizer] = None
        if config.quantizer_config:
            self.quantizer = builder.build_quantizer(config.quantizer_config)

        # Patch flags
        self.is_patched = isinstance(self.value_embedding, TimeSeriesPatchEmbedding)
        self.patch_size = getattr(self.value_embedding, "patch_size", 1)
        self.patch_stride = getattr(self.value_embedding, "stride", 1)

    # ---------- public helpers ----------
    def process(
        self,
        input_values: torch.Tensor,
        past_key_values_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        validate_shapes: bool = False,
        verbose: bool = False,
        *,
        skip_instance_norm: bool = False,
        freeze_norm_stats: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert raw inputs to model-ready hidden states + attention mask.

        Args:
            skip_instance_norm: True -> do not invoke RevIN at all
            freeze_norm_stats:  True -> apply stored stats (no update), False -> fit+apply
        """
        x = input_values

        # --- Instance normalization (RevIN) ---
        if self.instance_norm is not None and not skip_instance_norm:
            if freeze_norm_stats:
                x = self.instance_norm.transform(x, mask=attention_mask)
            else:
                x = self.instance_norm(x, mode="norm", mask=attention_mask, update_stats=True)

        # --- Value embedding (patch embed handles its own padding) ---
        value_embeds = self.value_embedding(x)                     # [B, T_tok, d_model]
        B, T_tok, d_model = value_embeds.shape
        quantizer_loss = None
        if self.quantizer is not None:
            quantizer_output = self.quantizer(value_embeds)
            hidden_states = quantizer_output["quantized"]
            quantizer_loss = quantizer_output.get("loss") # Can be None
        else:
            hidden_states = value_embedded
        # --- Positional embedding ---
        pos_embed = self.positional_embedding(
            x=value_embeds, batch_size=B, seq_len=T_tok, past_key_values_length=past_key_values_length
        ).to(dtype=value_embeds.dtype, device=value_embeds.device)

        hidden_states = value_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # --- Attention mask ---
        final_attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            input_shape=(B, T_tok),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
            is_causal=is_causal,
        )

        if validate_shapes and final_attention_mask is not None:
            bt, _, tq, tk = final_attention_mask.shape
            assert bt == B and tq == T_tok, f"Mask mismatch: {final_attention_mask.shape} vs (B={B},T={T_tok})"

        return {"hidden_states": hidden_states, "attention_mask": final_attention_mask}

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        if self.instance_norm is not None:
            return self.instance_norm(data, mode="denorm")
        return data

    # ---------- attention masks ----------
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

        if is_causal:
            final_mask = self._make_causal_mask((bsz, seq_len), dtype, device, past_key_values_length)

        if attention_mask is not None:
            processed_mask = attention_mask  # [B, L_raw]
            if self.is_patched:
                # Pad the 2D mask the same as the embedder pads values:
                L_raw = processed_mask.shape[1]
                pad_len = (self.patch_stride - (L_raw - self.patch_size) % self.patch_stride) % self.patch_stride
                if pad_len > 0:
                    processed_mask = F.pad(processed_mask, (0, pad_len), value=0.0)

                # Map tokens -> patches; prefer fully valid patches
                processed_mask = processed_mask.unfold(1, self.patch_size, self.patch_stride).all(dim=-1)

            # Cast to float before cache-left padding
            processed_mask = processed_mask.to(dtype=torch.float32, device=device)

            if is_causal and past_key_values_length > 0:
                # Pad left with ones for cached keys
                processed_mask = F.pad(processed_mask, (past_key_values_length, 0), value=1.0)

            expanded_padding_mask = self._expand_mask(processed_mask, dtype=dtype, tgt_len=seq_len).to(device)
            final_mask = expanded_padding_mask if final_mask is None else (final_mask + expanded_padding_mask)

        return final_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    ) -> torch.Tensor:
        bsz, tgt_len = input_ids_shape
        total_len = tgt_len + past_key_values_length

        q = torch.arange(tgt_len, device=device).view(tgt_len, 1)
        k = torch.arange(total_len, device=device).view(1, total_len)
        # Mask when key_pos > query_pos (+ past offset)
        cond = k > (q + past_key_values_length)
        mask = torch.where(cond, torch.finfo(dtype).min, torch.tensor(0.0, dtype=dtype, device=device))
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, total_len)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
        """
        2D -> 4D attention mask: [B, S] -> [B, 1, T, S]; 0=valid, -inf=masked.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)  # float mask, 1.0=valid
        inverted = (1.0 - expanded).to(dtype)  # 1.0->0.0 (valid), 0.0->1.0 (masked)
        return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)

    def _prepare_decoder_inputs_for_generation(
        self,
        patch_embeds: torch.Tensor,
        past_key_values_length: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Prepare a (possibly single-step) already-embedded patch for the decoder.
        """
        B, T, d_model = patch_embeds.shape
        pos_embed = self.positional_embedding(
            x=patch_embeds, batch_size=B, seq_len=T, past_key_values_length=past_key_values_length
        )
        pos_embed = pos_embed.to(dtype=patch_embeds.dtype, device=patch_embeds.device)

        hidden_states = patch_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        final_attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            input_shape=(B, T),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
            is_causal=is_causal,
        )
        return {"hidden_states": hidden_states, "attention_mask": final_attention_mask}
