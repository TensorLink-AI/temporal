temporal.modules.attentions.destationary
========================================

.. py:module:: temporal.modules.attentions.destationary


Classes
-------

.. autoapisummary::

   temporal.modules.attentions.destationary.Projector


Module Contents
---------------

.. py:class:: Projector(enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3)

   Bases: :py:obj:`torch.nn.Module`


   MLP to learn the De-stationary factors


   .. py:attribute:: series_conv


   .. py:attribute:: backbone


   .. py:method:: forward(x, stats)


