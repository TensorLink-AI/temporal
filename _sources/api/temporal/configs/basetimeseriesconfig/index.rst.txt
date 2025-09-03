temporal.configs.basetimeseriesconfig
=====================================

.. py:module:: temporal.configs.basetimeseriesconfig


Attributes
----------

.. autoapisummary::

   temporal.configs.basetimeseriesconfig.T


Classes
-------

.. autoapisummary::

   temporal.configs.basetimeseriesconfig.BaseTimeSeriesConfig


Module Contents
---------------

.. py:data:: T

.. py:class:: BaseTimeSeriesConfig(*, output_hidden_states: bool = False, output_attentions: bool = False, return_dict: bool = True, torchscript: bool = False, torch_dtype: Optional[Union[str, torch.dtype]] = None, pruned_heads: Optional[dict[int, list[int]]] = None, tie_word_embeddings: bool = True, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, is_decoder: bool = False, cross_attention_hidden_size: Optional[int] = None, add_cross_attention: bool = False, tie_encoder_decoder: bool = False, architectures: Optional[list[str]] = None, finetuning_task: Optional[str] = None, id2label: Optional[dict[int, str]] = None, label2id: Optional[dict[str, int]] = None, num_labels: Optional[int] = None, task_specific_params: Optional[dict[str, Any]] = None, problem_type: Optional[str] = None, tokenizer_class: Optional[str] = None, prefix: Optional[str] = None, bos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, sep_token_id: Optional[int] = None, decoder_start_token_id: Optional[int] = None, **kwargs)

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`, :py:obj:`transformers.PretrainedConfig`


   Configuration for a base time series forecasting model.

   Inherits from :class:`transformers.PretrainedConfig` to enable saving/loading
   via the Hugging Face ecosystem. This dataclass also integrates with the
   local `BaseConfig` for `to_dict` and `from_dict` consistency.


   .. py:attribute:: type
      :type:  str
      :value: 'base_time_series_config'



   .. py:attribute:: feature_size
      :type:  int
      :value: 1



   .. py:attribute:: context_length
      :type:  int
      :value: 128



   .. py:attribute:: prediction_length
      :type:  int
      :value: 12



   .. py:attribute:: quantiles
      :type:  Optional[List[float]]
      :value: []



   .. py:attribute:: output_token_lengths
      :type:  int
      :value: 1



   .. py:attribute:: loss_config
      :type:  temporal.configs.loss_config.LossConfig


   .. py:attribute:: use_dynamic_features
      :type:  bool
      :value: False



   .. py:attribute:: use_static_features
      :type:  bool
      :value: False



   .. py:attribute:: autoregressive
      :type:  bool
      :value: True



   .. py:attribute:: is_decoder
      :type:  bool
      :value: False



   .. py:attribute:: target_dim
      :type:  Optional[int]
      :value: None



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



   .. py:method:: to_dict() -> Dict[str, Any]

      Converts the dataclass instance to a dictionary, handling nested BaseConfig objects.



   .. py:method:: from_dict(data: Dict[str, Any]) -> T
      :classmethod:


      Creates a dataclass instance from a dictionary. It's designed to be
      forward-compatible by ignoring unknown keys and allowing new fields
      to be added to dataclasses with default values.



