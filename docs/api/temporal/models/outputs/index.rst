temporal.models.outputs
=======================

.. py:module:: temporal.models.outputs


Classes
-------

.. autoapisummary::

   temporal.models.outputs.EncoderLayerOutput
   temporal.models.outputs.DecoderLayerOutput


Module Contents
---------------

.. py:class:: EncoderLayerOutput

   .. py:attribute:: hidden_states
      :type:  torch.Tensor


   .. py:attribute:: attention_weights
      :type:  Optional[torch.Tensor]
      :value: None



   .. py:attribute:: aux_loss
      :type:  Optional[torch.Tensor]
      :value: None



.. py:class:: DecoderLayerOutput

   .. py:attribute:: hidden_states
      :type:  torch.Tensor


   .. py:attribute:: self_attention_weights
      :type:  Optional[torch.Tensor]
      :value: None



   .. py:attribute:: cross_attention_weights
      :type:  Optional[torch.Tensor]
      :value: None



   .. py:attribute:: past_key_value
      :type:  Optional[Tuple[torch.Tensor, torch.Tensor]]
      :value: None



   .. py:attribute:: aux_loss
      :type:  Optional[torch.Tensor]
      :value: None



