temporal.modules.norm.layer_norm
================================

.. py:module:: temporal.modules.norm.layer_norm


Classes
-------

.. autoapisummary::

   temporal.modules.norm.layer_norm.LayerNorm


Module Contents
---------------

.. py:class:: LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, **kwargs)

   Bases: :py:obj:`torch.nn.LayerNorm`


   Applies Layer Normalization over a mini-batch of inputs.

   This layer implements the operation as described in
   the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

   .. math::
       y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

   The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
   is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
   is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
   the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
   :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
   :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
   The variance is calculated via the biased estimator, equivalent to
   `torch.var(input, unbiased=False)`.

   .. note::
       Unlike Batch Normalization and Instance Normalization, which applies
       scalar scale and bias for each entire channel/plane with the
       :attr:`affine` option, Layer Normalization applies per-element scale and
       bias with :attr:`elementwise_affine`.

   This layer uses statistics computed from input data in both training and
   evaluation modes.

   :param normalized_shape: input shape from an expected input
                            of size

                            .. math::
                                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                                    \times \ldots \times \text{normalized\_shape}[-1]]

                            If a single integer is used, it is treated as a singleton list, and this module will
                            normalize over the last dimension which is expected to be of that specific size.
   :type normalized_shape: int or list or torch.Size
   :param eps: a value added to the denominator for numerical stability. Default: 1e-5
   :param elementwise_affine: a boolean value that when set to ``True``, this module
                              has learnable per-element affine parameters initialized to ones (for weights)
                              and zeros (for biases). Default: ``True``.
   :param bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
                :attr:`elementwise_affine` is ``True``). Default: ``True``.

   .. attribute:: weight

      the learnable weights of the module of shape
      :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
      The values are initialized to 1.

   .. attribute:: bias

      the learnable bias of the module of shape
      :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
      The values are initialized to 0.

   Shape:
       - Input: :math:`(N, *)`
       - Output: :math:`(N, *)` (same shape as input)

   Examples::

       >>> # NLP Example
       >>> batch, sentence_length, embedding_dim = 20, 5, 10
       >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
       >>> layer_norm = nn.LayerNorm(embedding_dim)
       >>> # Activate module
       >>> layer_norm(embedding)
       >>>
       >>> # Image Example
       >>> N, C, H, W = 20, 5, 10, 10
       >>> input = torch.randn(N, C, H, W)
       >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
       >>> # as shown in the image below
       >>> layer_norm = nn.LayerNorm([C, H, W])
       >>> output = layer_norm(input)

   .. image:: ../_static/img/nn/layer_norm.jpg
       :scale: 50 %



