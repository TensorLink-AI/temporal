temporal.models.mixin.multistep
===============================

.. py:module:: temporal.models.mixin.multistep


Attributes
----------

.. autoapisummary::

   temporal.models.mixin.multistep.logger


Classes
-------

.. autoapisummary::

   temporal.models.mixin.multistep.MultiStepMixin


Module Contents
---------------

.. py:data:: logger

.. py:class:: MultiStepMixin

   Single-pass multi-step forecasting with fixed decoder context and chunked roll-forward.

   - Decoder input length is always `context_length` (default: config.context_length or inputs.shape[1]).
   - If prediction_length <= chunk_length: one decoder call (true single pass).
   - If prediction_length  > chunk_length: repeat:
       * feed last `context_length` inputs (auto-padded if shorter),
       * decode once,
       * take head on the LAST `steps` hidden states,
       * append median predictions to inputs (model input domain),
       * slide window by `steps`.
   - No per-step autoregression; exactly one decoder call per chunk.

   Requirements on the host model:
     - self.config.feature_size (and optionally .context_length, .prediction_length, .num_attention_heads)
     - self.preprocessor.process(...), optional self.preprocessor.denormalize(...)
     - self.decoder (and optional self.encoder)
     - self.output_heads (nn.Module or nn.ModuleList)
     - optional self.head_aggregator (for multi-head aggregation)


   .. py:method:: forecast_single_pass_chunked(inputs: torch.Tensor, prediction_length: int, *, quantiles: Optional[List[float]] = None, chunk_length: int = 256, context_length: Optional[int] = None, collect: str = 'auto', prediction_strategy: Optional[Union[str, float, int]] = None, validate_shapes: bool = True, verbose: bool = True, return_raw: bool = False, attention_mask: Optional[torch.Tensor] = None, pad_context_mode: str = 'left_zeros', pad_value: float = 0.0) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]

      Single-pass multi-step with fixed-length decoder context and chunked roll-forward.

      :returns:

                - collect='none'     -> aggregated tensor when possible (predict/quantiles/raw), [B, T, ...]
                - collect='distpred' -> {'paths':[B,T,F,K], 'path_logits':[B,T,K]?}
                - collect='per_chunk'-> list of per-chunk outputs (each [B, steps, ...] or a DistPred dict)
                - collect='auto'     -> stacked DistPred dict if available, else tensor



