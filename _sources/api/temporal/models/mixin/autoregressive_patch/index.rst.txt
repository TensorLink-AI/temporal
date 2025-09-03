temporal.models.mixin.autoregressive_patch
==========================================

.. py:module:: temporal.models.mixin.autoregressive_patch


Attributes
----------

.. autoapisummary::

   temporal.models.mixin.autoregressive_patch.logger


Classes
-------

.. autoapisummary::

   temporal.models.mixin.autoregressive_patch.AutoregressivePatchMixin


Module Contents
---------------

.. py:data:: logger

.. py:class:: AutoregressivePatchMixin

   Patch-based autoregressive generation that returns a consistent ForecastBundle.

   This mixin is designed for models that autoregress in a *patch latent space*
   (decoder predicts the next patch embedding), then reconstruct the native
   time-domain sequence from the generated patch embeddings, and finally apply
   probabilistic output heads (e.g., Gaussian / StudentT / MDN / DistPred).

   Assumptions about the host model
   --------------------------------
   • self.config
       - feature_size: int          (# target features/channels F)
       - d_model: int               (head input size after reconstruction)
       - prediction_length: Optional[int]
       - num_attention_heads: Optional[int]
   • self.preprocessor
       - patch_size: int
       - process(input_values, attention_mask, is_causal, ...)
       - _prepare_decoder_inputs_for_generation(patch_embeds, attention_mask, past_key_values_length, is_causal)
       - denormalize(tensor)   # optional; applied to point only
   • self.encoder (optional) and self.decoder: HuggingFace-style
       - return_dict=True objects with .last_hidden_state and .past_key_values
   • self.output_patch_reconstructor(patch_embeds) -> Tensor
       - Typically maps [B, T_patch, D_latent] -> [B, T_patch, d_model]
         which we then reshape into [B, T_native, d_model]
   • self.output_heads: nn.Module or nn.ModuleList
       - forward(x_last) returns tensor OR dict of params
       - predict(params, method=...) -> [B, T, F]
       - sample(params, **kwargs)    -> [B, T, F]   (vectorized heads only; see notes)
       - sample_quantiles(params, quantile_levels) -> [B, T, 1/F?, Q]  (normalized here)
   • optional: self.head_aggregator(list_of_head_outputs) -> head_output

   Key guarantees
   --------------
   * bundle.point is always a Tensor [B, T, F]
   * bundle.quantiles is either None or a Tensor [B, T, F, Q]
   * bundle.params is the stacked raw *primary head* outputs:
       - Tensor for Gaussian/StudentT/QuantileRegression/etc.
       - Dict[str, Tensor] for Mixture (MDN) / DistPred
   * No dicts are ever placed into bundle.quantiles (prevents `.shape` errors)

   Notes on sampling with patch generation
   ---------------------------------------
   In this patch paradigm the decoder’s AR loop happens in *latent patch space*.
   Output heads are applied *after* the full horizon is reconstructed.
   Therefore, head.sample(...) does **not** influence the decoder’s next state.
   We still allow sampling to produce a sample path as the returned `point`
   (when the head supports *vectorized* sampling on [B, T, ...] params, e.g.,
   Gaussian/StudentT). For heads that implement only one-step sampling
   (e.g., MDN sample() that returns [B,1,1]), we fall back to predict(...).


   .. py:method:: enable_dropout() -> None

      Enable dropout layers (for MC dropout).



   .. py:method:: generate(encoder_inputs: Optional[torch.Tensor] = None, decoder_inputs: Optional[torch.Tensor] = None, prediction_length: Optional[int] = None, attention_mask: Optional[torch.Tensor] = None, decoder_attention_mask: Optional[torch.Tensor] = None, use_cache: bool = True, decoder_start_token_id: Optional[Any] = None, eos_token_id: Optional[Any] = None, early_stopping: bool = False, output_attentions: bool = False, output_hidden_states: bool = False, prediction_strategy: Optional[Union[str, float, int]] = None, quantile_levels: Optional[List[float]] = None, validate_shapes: bool = True, verbose: bool = True, *, sampling: bool = False, sampling_kwargs: Optional[Dict[str, Any]] = None, store_sampled: bool = False, enable_mc_dropout: bool = False, return_raw: bool = False, post_quantiles: bool = True, return_bundle: bool = False, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor], temporal.modules.heads.forecast_types.ForecastBundle, List[Dict[str, Union[torch.Tensor, List[str]]]]]

      Patch-based autoregressive forecast.

      Steps
      -----
      1) Encode context (if encoder exists) and seed the decoder with the last
         context patch (enc-dec) or the provided decoder_inputs (dec-only).
      2) Autoregressively predict *patch embeddings* for ceil(pred_len / patch_size) steps.
      3) Reconstruct native-time embeddings and reshape to [B, T_native, d_model].
      4) Apply primary output head on the full horizon to get params (tensor/dict).
      5) Optionally compute quantiles once on the stacked params (MDN-safe).
      6) Compute a point forecast via:
            • sampling (vectorized heads only) OR
            • predict(method=prediction_strategy or 'median') OR
            • median from computed quantiles
      7) Optionally denormalize the point.
      8) Return ForecastBundle or (quantiles if tensor) else point.

      .. important::

         • sampling=True does NOT influence the decoder’s AR loop here. It only
           changes how we produce the final `point` from the stacked head params.



   .. py:method:: forecast(inputs: torch.Tensor, prediction_length: int, quantiles: Optional[List[float]] = None, **kwargs)

      Convenience wrapper around `generate` for encoder/decoder configurations.

      :param inputs: Input sequence [B, T, F] (for encoder-decoder: historical context,
                     for decoder-only: prompt).
      :type inputs: Tensor
      :param prediction_length: Number of steps to forecast in native time steps.
      :type prediction_length: int
      :param quantiles: Quantiles to compute post-hoc (e.g., [0.1, 0.5, 0.9]).
      :type quantiles: List[float] | None



