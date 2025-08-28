temporal.models.transformer_model
=================================

.. py:module:: temporal.models.transformer_model


Classes
-------

.. autoapisummary::

   temporal.models.transformer_model.TransformerOutput
   temporal.models.transformer_model.TransformerTemporalModel


Module Contents
---------------

.. py:class:: TransformerOutput

   A structured output class for the transformer model.

   This dataclass holds all the potential outputs of the `TransformerTemporalModel`,
   making them accessible by attribute. It provides a consistent and predictable
   output format, similar to the output objects in the Hugging Face
   Transformers library.

   .. attribute:: logits

      The final model predictions.

      :type: torch.FloatTensor

   .. attribute:: loss

      The loss, computed if targets are provided.

      :type: Optional[torch.FloatTensor]

   .. attribute:: aux_loss

      The auxiliary loss, e.g. from MoE layers.

      :type: Optional[torch.FloatTensor]

   .. attribute:: past_key_values

      The KV cache for
      accelerated decoding.

      :type: Optional[Tuple[Tuple[torch.Tensor]]]

   .. attribute:: decoder_hidden_states

      Hidden states
      of the decoder.

      :type: Optional[Tuple[torch.FloatTensor]]

   .. attribute:: decoder_attentions

      Attention weights
      from the decoder's self-attention layers.

      :type: Optional[Tuple[torch.FloatTensor]]

   .. attribute:: decoder_cross_attentions

      Attention
      weights from the decoder's cross-attention layers.

      :type: Optional[Tuple[torch.FloatTensor]]

   .. attribute:: encoder_last_hidden_state

      The last hidden
      state of the encoder.

      :type: Optional[torch.FloatTensor]

   .. attribute:: encoder_hidden_states

      Hidden states
      of the encoder.

      :type: Optional[Tuple[torch.FloatTensor]]

   .. attribute:: encoder_attentions

      Attention weights
      from the encoder's self-attention layers.

      :type: Optional[Tuple[torch.FloatTensor]]


   .. py:attribute:: logits
      :type:  torch.FloatTensor
      :value: None



   .. py:attribute:: loss
      :type:  Optional[torch.FloatTensor]
      :value: None



   .. py:attribute:: aux_loss
      :type:  Optional[torch.FloatTensor]
      :value: None



   .. py:attribute:: past_key_values
      :type:  Optional[Tuple[Tuple[torch.Tensor]]]
      :value: None



   .. py:attribute:: decoder_hidden_states
      :type:  Optional[Tuple[torch.FloatTensor]]
      :value: None



   .. py:attribute:: decoder_attentions
      :type:  Optional[Tuple[torch.FloatTensor]]
      :value: None



   .. py:attribute:: decoder_cross_attentions
      :type:  Optional[Tuple[torch.FloatTensor]]
      :value: None



   .. py:attribute:: encoder_last_hidden_state
      :type:  Optional[torch.FloatTensor]
      :value: None



   .. py:attribute:: encoder_hidden_states
      :type:  Optional[Tuple[torch.FloatTensor]]
      :value: None



   .. py:attribute:: encoder_attentions
      :type:  Optional[Tuple[torch.FloatTensor]]
      :value: None



   .. py:method:: __getitem__(key: str) -> Any

      Allows dictionary-style access to attributes.



   .. py:method:: __setitem__(key: str, value: Any)

      Allows dictionary-style setting of attributes.



   .. py:method:: keys() -> List[str]

      Returns the names of the attributes.



   .. py:method:: to_dict() -> Dict[str, Any]

      Converts the dataclass to a dictionary.



.. py:class:: TransformerTemporalModel(config: temporal.configs.transformer_model_config.TransformerTimeSeriesConfig, encoder: Optional[torch.nn.Module] = None, decoder: Optional[torch.nn.Module] = None, output_heads: Optional[torch.nn.Module] = None, head_aggregator: Optional[torch.nn.Module] = None, loss_fn: Optional[callable] = None, builder: Optional[temporal.models.module_builder_helper.ModuleBuilder] = None)

   Bases: :py:obj:`temporal.models.mixin.autoregressive.AutoregressiveDispatchMixin`, :py:obj:`temporal.models.mixin.autoregressive_patch.AutoregressivePatchMixin`, :py:obj:`temporal.models.mixin.autoregressive_stepwise.AutoregressiveStepwiseMixin`, :py:obj:`temporal.models.mixin.multistep.MultiStepMixin`, :py:obj:`temporal.models.base_model.BaseTemporalModel`


   A concrete implementation of a transformer-based temporal model.

   This class assembles the encoder, decoder, and output heads into a cohesive
   model. It defines the main `forward` pass, handling the flow of data through
   the different components. It also inherits generation capabilities from the
   `AutoregressiveMixin` and `MultiStepMixin` classes.


   .. py:attribute:: builder
      :value: None



   .. py:attribute:: preprocessor


   .. py:attribute:: output_patch_reconstructor
      :value: None



   .. py:method:: forward(encoder_inputs: Optional[torch.Tensor] = None, decoder_inputs: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, decoder_attention_mask: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, loss_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, validate_shapes: bool = False, verbose: bool = False, x_raw: Optional[torch.Tensor] = None) -> TransformerOutput

      Performs a forward pass through the entire transformer model.

      :param encoder_inputs: Inputs for the encoder,
                             shape `[B, L_enc, F_enc]`.
      :type encoder_inputs: Optional[torch.Tensor]
      :param decoder_inputs: Inputs for the decoder,
                             shape `[B, L_dec, F_dec]`.
      :type decoder_inputs: Optional[torch.Tensor]
      :param attention_mask: A 2D padding mask for the
                             encoder inputs, shape `[B, L_enc]`.
      :type attention_mask: Optional[torch.Tensor]
      :param decoder_attention_mask: A 2D padding mask for
                                     the decoder inputs, shape `[B, L_dec]`.
      :type decoder_attention_mask: Optional[torch.Tensor]
      :param targets: The ground truth values for loss
                      calculation, shape `[B, L_dec, F_out]`.
      :type targets: Optional[torch.Tensor]
      :param loss_mask: An optional mask to apply to the loss
                        calculation. Its shape should be broadcastable to the shape of `logits`.
      :type loss_mask: Optional[torch.Tensor]
      :param past_key_values: A cache of key-value states for
                              efficient autoregressive decoding.
      :type past_key_values: Optional[Tuple]
      :param use_cache: If True, the model will return the
                        updated `past_key_values`.
      :type use_cache: Optional[bool]
      :param output_attentions: If True, returns attention weights.
      :type output_attentions: Optional[bool]
      :param output_hidden_states: If True, returns hidden states
                                   from all layers.
      :type output_hidden_states: Optional[bool]
      :param validate_shapes: If True, performs assertions inside the
                              preprocessor to check for shape consistency.
      :type validate_shapes: bool
      :param verbose: If True, prints detailed shape information from
                      the preprocessor for debugging.
      :type verbose: bool
      :param x_raw: Raw input for de-stationary attention.
      :type x_raw: Optional[torch.Tensor]

      :returns: A structured object containing the model's outputs.
      :rtype: TransformerOutput



