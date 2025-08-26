temporal.modules.heads.timeflow_head
====================================

.. py:module:: temporal.modules.heads.timeflow_head


Classes
-------

.. autoapisummary::

   temporal.modules.heads.timeflow_head.TimeFlowTimestepEmbedder
   temporal.modules.heads.timeflow_head.TimeFlowResBlock
   temporal.modules.heads.timeflow_head.TimeFlowFinalLayer
   temporal.modules.heads.timeflow_head.TimeFlowMLPAdaLN
   temporal.modules.heads.timeflow_head.TimeFlowHead


Module Contents
---------------

.. py:class:: TimeFlowTimestepEmbedder(emb_dim: int, freq_emb_dim: int = 256)

   Bases: :py:obj:`torch.nn.Module`


   Embeds scalar timesteps into a vector for TimeFlowLoss.


   .. py:attribute:: freq_emb_dim
      :value: 256



   .. py:attribute:: mlp


   .. py:method:: forward(t: torch.Tensor) -> torch.Tensor


.. py:class:: TimeFlowResBlock(channels: int)

   Bases: :py:obj:`torch.nn.Module`


   A single residual block with AdaLN modulation for TimeFlowMLPAdaLN.


   .. py:attribute:: in_ln


   .. py:attribute:: mlp


   .. py:attribute:: mod


   .. py:method:: forward(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor


.. py:class:: TimeFlowFinalLayer(model_channels: int, out_channels: int)

   Bases: :py:obj:`torch.nn.Module`


   Final projection layer with AdaLN for TimeFlowMLPAdaLN.


   .. py:attribute:: norm


   .. py:attribute:: linear


   .. py:attribute:: mod


   .. py:method:: forward(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor


.. py:class:: TimeFlowMLPAdaLN(in_channels: int, model_channels: int, out_channels: int, cond_channels: int, num_blocks: int)

   Bases: :py:obj:`torch.nn.Module`


   The MLP backbone for TimeFlowLoss.
   Projects inputs → model_channels, applies num_blocks ResBlocks, then final projection.


   .. py:attribute:: input_proj


   .. py:attribute:: time_embed


   .. py:attribute:: cond_proj


   .. py:attribute:: blocks


   .. py:attribute:: final


   .. py:method:: forward(x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor


.. py:class:: TimeFlowHead(target_channels: int, cond_channels: int, num_blocks: int, model_channels: int, **kwargs)

   Bases: :py:obj:`temporal.modules.heads.base_output_head.BaseOutputHead`


   An abstract base class for all model output heads.

   This class defines the common interface that all output head modules must
   adhere to. An output head is responsible for taking the final hidden state
   from the model's backbone and transforming it into the desired output format
   (e.g., a point forecast, a probability distribution).

   It also defines a method for retrieving the appropriate loss function
   to be used with the head's output.


   .. py:attribute:: net


   .. py:attribute:: num_sampling_steps


   .. py:attribute:: num_samples_for_quantiles


   .. py:method:: forward(cond: torch.Tensor, targets: torch.Tensor) -> torch.Tensor

      Processes the model's final hidden state to produce the output.

      This method must be implemented by all subclasses.

      :param hidden_state: The final hidden state from the model's
                           backbone, typically of shape `[batch_size, seq_len, d_model]`.
      :type hidden_state: torch.Tensor

      :returns:

                The model's final output, with its shape and meaning
                    determined by the specific head implementation.
      :rtype: torch.Tensor



   .. py:method:: sample(cond: torch.Tensor, num_samples: int = 1) -> torch.Tensor

      Generates samples via simple Euler discretization.



   .. py:method:: sample_quantiles(cond: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor

      Computes quantiles from generated samples.

      :param cond: The conditioning tensor of shape [B, C].
      :type cond: torch.Tensor
      :param quantile_levels: List of quantile levels to compute.
      :type quantile_levels: List[float]

      :returns: Quantile predictions of shape [B, Q, C].
      :rtype: torch.Tensor



