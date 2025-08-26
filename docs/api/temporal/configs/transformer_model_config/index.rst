temporal.configs.transformer_model_config
=========================================

.. py:module:: temporal.configs.transformer_model_config


Attributes
----------

.. autoapisummary::

   temporal.configs.transformer_model_config.T


Classes
-------

.. autoapisummary::

   temporal.configs.transformer_model_config.TransformerTimeSeriesConfig


Module Contents
---------------

.. py:data:: T

.. py:class:: TransformerTimeSeriesConfig(*, output_hidden_states: bool = False, output_attentions: bool = False, return_dict: bool = True, torchscript: bool = False, torch_dtype: Optional[Union[str, torch.dtype]] = None, pruned_heads: Optional[dict[int, list[int]]] = None, tie_word_embeddings: bool = True, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, is_decoder: bool = False, cross_attention_hidden_size: Optional[int] = None, add_cross_attention: bool = False, tie_encoder_decoder: bool = False, architectures: Optional[list[str]] = None, finetuning_task: Optional[str] = None, id2label: Optional[dict[int, str]] = None, label2id: Optional[dict[str, int]] = None, num_labels: Optional[int] = None, task_specific_params: Optional[dict[str, Any]] = None, problem_type: Optional[str] = None, tokenizer_class: Optional[str] = None, prefix: Optional[str] = None, bos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, sep_token_id: Optional[int] = None, decoder_start_token_id: Optional[int] = None, **kwargs)

   Bases: :py:obj:`temporal.configs.basetimeseriesconfig.BaseTimeSeriesConfig`


   Configuration for a transformer-based time-series forecasting model.


   .. py:attribute:: type
      :type:  str
      :value: 'transformer'



   .. py:attribute:: model_type
      :type:  str
      :value: 'transformer'



   .. py:attribute:: d_model
      :type:  int
      :value: 64



   .. py:attribute:: hidden_dropout_prob
      :type:  float
      :value: 0.1



   .. py:attribute:: max_position_embeddings
      :type:  int
      :value: 4096



   .. py:attribute:: architecture
      :type:  temporal.configs.architecture_config.TransformerArchitectureConfig


   .. py:attribute:: value_embedding_config
      :type:  temporal.configs.embedding_config.EmbeddingConfig


   .. py:attribute:: positional_embedding_config
      :type:  temporal.configs.embedding_config.EmbeddingConfig


   .. py:attribute:: encoder_blocks
      :type:  Optional[List[temporal.configs.transformer_block_config.TransformerBlockConfig]]
      :value: None



   .. py:attribute:: decoder_blocks
      :type:  Optional[List[temporal.configs.transformer_block_config.TransformerBlockConfig]]
      :value: None



   .. py:attribute:: output_head_config
      :type:  temporal.configs.output_head_config.OutputHeadConfig


   .. py:attribute:: layer_norm_config
      :type:  temporal.configs.normalization_config.NormalizationConfig


   .. py:attribute:: instance_norm_config
      :type:  Optional[temporal.configs.normalization_config.NormalizationConfig]
      :value: None



   .. py:attribute:: head_agg_config
      :type:  temporal.configs.head_aggregation_config.HeadAggregationConfig


   .. py:attribute:: quantizer_config
      :type:  Optional[temporal.configs.quantizer_config.QuantizerConfig]
      :value: None



   .. py:attribute:: vocab_size
      :type:  Optional[int]
      :value: None



   .. py:attribute:: decoder_start_token_id
      :type:  Optional[int]
      :value: None



   .. py:attribute:: output_attentions
      :type:  bool
      :value: False


      Whether or not the model should returns all attentions.

      :type: `bool`


   .. py:attribute:: output_hidden_states
      :type:  bool
      :value: False



   .. py:attribute:: use_teacher_forcing
      :type:  bool
      :value: True



   .. py:attribute:: aux_loss_weight
      :type:  float
      :value: 0.01



   .. py:attribute:: use_cache
      :type:  bool
      :value: True



   .. py:attribute:: attention_blocks
      :type:  Optional[Any]
      :value: None



   .. py:attribute:: feedforward_config
      :type:  Optional[Any]
      :value: None



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



   .. py:method:: from_dict(data: Dict[str, Any]) -> T
      :classmethod:


      Creates a dataclass instance from a dictionary. It's designed to be
      forward-compatible by ignoring unknown keys and allowing new fields
      to be added to dataclasses with default values.



