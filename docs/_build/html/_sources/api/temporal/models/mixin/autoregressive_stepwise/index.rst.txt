temporal.models.mixin.autoregressive_stepwise
=============================================

.. py:module:: temporal.models.mixin.autoregressive_stepwise


Attributes
----------

.. autoapisummary::

   temporal.models.mixin.autoregressive_stepwise.logger


Classes
-------

.. autoapisummary::

   temporal.models.mixin.autoregressive_stepwise.AutoregressiveStepwiseMixin


Module Contents
---------------

.. py:data:: logger

.. py:class:: AutoregressiveStepwiseMixin

   Stepwise autoregressive generation that returns a consistent ForecastBundle.

   This mixin assumes the host model provides:
     - self.config:
         • feature_size: int  (number of target features/channels)
         • hidden_size: int   (decoder hidden size)
         • prediction_length: Optional[int]
         • num_attention_heads: Optional[int]
     - self.preprocessor with:
         • process(input_values, attention_mask, past_key_values_length, is_causal, ...)
         • denormalize(tensor)           # optional; applied to point only
     - self.decoder (and optional self.encoder) compatible with HuggingFace-style outputs:
         • returns an object with .last_hidden_state and optionally .past_key_values
     - self.output_heads:
         • nn.Module or nn.ModuleList whose .forward(last_hidden) returns:
           - a tensor of params, or
           - a dict of params (e.g. MDN/DistPred), or
           - a list (if ModuleList) of the above per head
         • Heads should implement:
           - predict(params, method="median" | "mean" | float quantile | int index)
           - sample(params, **kwargs)                     # optional; used for AR feedback
           - sample_quantiles(params, quantile_levels)    # optional; used post-loop
     - optional: self._values_to_hidden(hidden_states) if your decoder expects a different space
     - optional: self.head_aggregator(list_of_head_outputs) -> head_output to use for feedback

   Key guarantees:
     * bundle.point is always a Tensor [B, T, F]
     * bundle.quantiles is either None or a Tensor [B, T, F, Q]
     * bundle.params is the stacked raw head outputs:
         - Tensor for Gaussian/StudentT/QuantileRegression/etc.
         - Dict[str, Tensor] for Mixture (MDN) or DistPred
     * No dicts are ever placed into bundle.quantiles (prevents `.shape` errors)


   .. py:method:: enable_dropout() -> None

      Enable dropout layers (for MC dropout).



   .. py:method:: generate(encoder_inputs: Optional[torch.Tensor] = None, decoder_inputs: Optional[torch.Tensor] = None, prediction_length: Optional[int] = None, attention_mask: Optional[torch.Tensor] = None, decoder_attention_mask: Optional[torch.Tensor] = None, use_cache: bool = True, decoder_start_token_id: Optional[Any] = None, eos_token_id: Optional[Any] = None, early_stopping: bool = False, output_attentions: bool = False, output_hidden_states: bool = False, prediction_strategy: Optional[Union[str, float, int]] = None, quantile_levels: Optional[List[float]] = None, validate_shapes: bool = True, verbose: bool = True, *, sampling: bool = True, sampling_kwargs: Optional[Dict[str, Any]] = None, store_sampled: bool = False, enable_mc_dropout: bool = False, return_raw: bool = False, post_quantiles: bool = True, return_bundle: bool = False, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor], temporal.modules.heads.forecast_types.ForecastBundle, List[Dict[str, Union[torch.Tensor, List[str]]]]]

      Stepwise autoregressive forecast.

      :param encoder_inputs: Input sequence for encoder (if model has an encoder).
      :type encoder_inputs: Tensor | None
      :param decoder_inputs: Initial decoder prompt values [B, T0, F]. If None and encoder exists,
                             uses the last encoder input step as the seed.
      :type decoder_inputs: Tensor | None
      :param prediction_length: Number of autoregressive steps to generate. Defaults to config.prediction_length.
      :type prediction_length: int | None
      :param attention_mask: Masks for encoder/decoder inputs.
      :type attention_mask: Tensor | None
      :param decoder_attention_mask: Masks for encoder/decoder inputs.
      :type decoder_attention_mask: Tensor | None
      :param use_cache: Whether to thread past_key_values through the decoder.
      :type use_cache: bool
      :param eos_token_id: If provided, enables an equality-based early stopping check on feedback values.
      :type eos_token_id: Any | None
      :param prediction_strategy:
                                  Strategy for head.predict() during feedback/storage if sampling is disabled:
                                    • "mean" or "median"
                                    • float in (0,1): quantile(e.g., 0.1 or 0.9)
                                    • int: take component/index when supported by the head (e.g., DistPred).
      :type prediction_strategy: str | float | int | None
      :param quantile_levels: Requested quantiles to compute once after the AR loop.
      :type quantile_levels: List[float] | None
      :param sampling: If True and the head implements .sample(...), feedback uses sampling.
      :type sampling: bool
      :param sampling_kwargs: Extra kwargs passed to head.sample(...), e.g., temperature/top_p/state.
      :type sampling_kwargs: dict | None
      :param store_sampled: If True, the returned `point` is the actually sampled AR path.
      :type store_sampled: bool
      :param enable_mc_dropout: If True, puts dropout layers in train mode during eval for MC sampling.
      :type enable_mc_dropout: bool
      :param return_raw: If False and preprocessor.denormalize exists, denormalize the `point`.
      :type return_raw: bool
      :param post_quantiles: If True, compute quantiles once on stacked params (preferred for MDN).
      :type post_quantiles: bool
      :param return_bundle: If True, return a ForecastBundle; else return `quantiles` (if tensor) or `point`.
      :type return_bundle: bool

      :returns:

                - ForecastBundle(point, quantiles, params, extras) if return_bundle=True
                - If not returning a bundle:
                    • quantiles tensor [B,T,F,Q] when computed
                    • else point tensor [B,T,F]
      :rtype: ForecastBundle | Tensor | Dict | list



   .. py:method:: forecast(inputs: torch.Tensor, prediction_length: int, quantiles: Optional[List[float]] = None, **kwargs)

      Convenience wrapper around `generate` for encoder/decoder configurations.

      :param inputs: Input sequence [B, T, F].
      :type inputs: Tensor
      :param prediction_length: Number of steps to forecast.
      :type prediction_length: int
      :param quantiles: Quantiles to compute post-hoc (e.g., [0.1, 0.5, 0.9]).
      :type quantiles: List[float] | None



