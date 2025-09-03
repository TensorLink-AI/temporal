temporal.modules.blocks.block_patch_transformer
===============================================

.. py:module:: temporal.modules.blocks.block_patch_transformer


Classes
-------

.. autoapisummary::

   temporal.modules.blocks.block_patch_transformer.PatchTransformBlock


Module Contents
---------------

.. py:class:: PatchTransformBlock(config: temporal.configs.transformer_block_config.AdaptivePatchTransformerBlockConfig, builder: temporal.models.module_builder_helper.ModuleBuilder, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A transformer block that wraps another transformer layer, applying patch
   splitting/merging before and after.


   .. py:attribute:: expansion_factor


   .. py:attribute:: order


   .. py:attribute:: transformer_layer


   .. py:method:: forward(hidden_states: torch.Tensor, attention_mask=None, past_key_value=None, **kwargs) -> temporal.models.outputs.DecoderLayerOutput


   .. py:method:: forward_split_first(hidden_states, attention_mask, **kwargs)


   .. py:method:: forward_merge_first(hidden_states, attention_mask, **kwargs)


