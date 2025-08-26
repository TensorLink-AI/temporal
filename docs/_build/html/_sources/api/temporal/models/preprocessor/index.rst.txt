temporal.models.preprocessor
============================

.. py:module:: temporal.models.preprocessor


Classes
-------

.. autoapisummary::

   temporal.models.preprocessor.InputPreprocessor


Module Contents
---------------

.. py:class:: InputPreprocessor(config: temporal.configs.transformer_model_config.TransformerTimeSeriesConfig, builder: temporal.models.module_builder_helper.ModuleBuilder)

   Bases: :py:obj:`torch.nn.Module`


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


   .. py:attribute:: config


   .. py:attribute:: instance_norm
      :value: None



   .. py:attribute:: value_embedding


   .. py:attribute:: positional_embedding


   .. py:attribute:: layernorm_embedding


   .. py:attribute:: dropout


   .. py:attribute:: is_patched


   .. py:attribute:: patch_size


   .. py:attribute:: patch_stride


   .. py:method:: process(input_values: torch.Tensor, past_key_values_length: int = 0, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = False, validate_shapes: bool = False, verbose: bool = False) -> Dict[str, Any]

      Processes raw input tensors into embeddings and masks.



   .. py:method:: denormalize(data: torch.Tensor) -> torch.Tensor

      Reverses the instance normalization if it was applied.



