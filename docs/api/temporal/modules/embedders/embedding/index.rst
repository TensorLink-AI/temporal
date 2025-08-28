temporal.modules.embedders.embedding
====================================

.. py:module:: temporal.modules.embedders.embedding


Classes
-------

.. autoapisummary::

   temporal.modules.embedders.embedding.BaseEmbedding
   temporal.modules.embedders.embedding.TimeSeriesValueEmbedding
   temporal.modules.embedders.embedding.FlexibleValueEmbedding
   temporal.modules.embedders.embedding.SinusoidalPositionalEmbedding
   temporal.modules.embedders.embedding.TimeSeriesPatchEmbedding
   temporal.modules.embedders.embedding.RotaryPositionalEmbedding
   temporal.modules.embedders.embedding.TimeSeriesGlobalEmbedding
   temporal.modules.embedders.embedding.LearnedAbsolutePositionalEmbedding
   temporal.modules.embedders.embedding.ShawRelativePositionalBias
   temporal.modules.embedders.embedding.FourierFeatureEmbedding
   temporal.modules.embedders.embedding.Time2VecEmbedding
   temporal.modules.embedders.embedding.ALiBiPositionalBias
   temporal.modules.embedders.embedding.BucketedRelativeBias
   temporal.modules.embedders.embedding.ConvolutionalPositionalEmbedding
   temporal.modules.embedders.embedding.TimeDeltaEmbedding
   temporal.modules.embedders.embedding.StackedPositionalEmbedding
   temporal.modules.embedders.embedding.NoneEmbedding
   temporal.modules.embedders.embedding.S4PositionalEmbedding
   temporal.modules.embedders.embedding.WaveletPositionalEmbedding


Functions
---------

.. autoapisummary::

   temporal.modules.embedders.embedding.rotate_half
   temporal.modules.embedders.embedding.apply_rotary_pos_emb


Module Contents
---------------

.. py:function:: rotate_half(x: torch.Tensor) -> torch.Tensor

.. py:function:: apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]

.. py:class:: BaseEmbedding(d_model: int)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: d_model


   .. py:method:: forward(*args, **kwargs)
      :abstractmethod:



.. py:class:: TimeSeriesValueEmbedding(feature_size: int, d_model: int, use_value_norm: bool = False)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: value_projection


   .. py:attribute:: value_norm
      :value: None



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: FlexibleValueEmbedding(*, d_model: int, input_dims: Union[int, Sequence[int]], proj_builder: Callable[[int, int, Dict[str, Any]], torch.nn.Module] = None, proj_kwargs: Dict[str, Any] = None, use_layer_norm: bool = False)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: layer_norm
      :value: None



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: SinusoidalPositionalEmbedding(d_model: int, max_seq_len: int = 2048)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: d_model


   .. py:method:: forward(x: torch.Tensor | None = None, *, batch_size: int | None = None, seq_len: int | None = None, past_key_values_length: int = 0, **kwargs) -> torch.Tensor

      Two calling conventions:
        1) emb(x): returns x + PE  (shape [B, T, D])
        2) emb(batch_size=..., seq_len=..., [x=...]): returns PE only (shape [B, T, D])
           (x is ignored in this mode; it is accepted for API symmetry)



.. py:class:: TimeSeriesPatchEmbedding(patch_size: int, feature_size: int, d_model: int, stride: Optional[int] = None, pad_value: float = 0.0, use_mlp: bool = False, mlp_hidden_size: Optional[int] = None, output_patch_size: Optional[int] = None)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: patch_size


   .. py:attribute:: stride


   .. py:attribute:: pad_value
      :value: 0.0



   .. py:attribute:: flat_size


   .. py:attribute:: output_patch_size
      :value: None



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Correctly pads, unfolds, and projects the input tensor into patches.
      :param x: Input tensor of shape [B, L, F].

      :returns: Tensor of shape [B, num_patches, d_model].



.. py:class:: RotaryPositionalEmbedding(d_model: int, max_seq_len: int = 2048, base: int = 10000)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: max_seq_len
      :value: 2048



   .. py:attribute:: base
      :value: 10000



   .. py:method:: forward(x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]

      Infers seq_len from input and returns the cos/sin caches.
      :param x: A dummy tensor of shape [B, H, L, D] to infer device.
      :param seq_len: The sequence length.

      :returns: Tuple of (cos, sin), each of shape [seq_len, d_model].



.. py:class:: TimeSeriesGlobalEmbedding(seq_len: int, feature_size: int, d_model: int)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: seq_len


   .. py:attribute:: feature_size


   .. py:attribute:: global_projection


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: LearnedAbsolutePositionalEmbedding(d_model: int, max_seq_len: int = 2048)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: max_seq_len
      :value: 2048



   .. py:attribute:: embedding


   .. py:method:: forward(x: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor


.. py:class:: ShawRelativePositionalBias(num_heads: int, max_distance: int = 128)

   Bases: :py:obj:`BaseEmbedding`


   Learnable relative positional biases as in Shaw et al.


   .. py:attribute:: max_distance
      :value: 128



   .. py:attribute:: relative_bias


   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      Compute relative bias tensor for attention scores.

      :param batch_size: Batch size.
      :param seq_len: Sequence length.

      :returns: Tensor of shape [1, num_heads, seq_len, seq_len].



.. py:class:: FourierFeatureEmbedding(d_model: int, num_features: int = 16)

   Bases: :py:obj:`BaseEmbedding`


   Embed positions using random Fourier features.


   .. py:attribute:: num_features
      :value: 16



   .. py:attribute:: proj


   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      Embed positions into Fourier feature space.

      :param batch_size: Batch size.
      :param seq_len: Sequence length.

      :returns: Tensor of shape [B, seq_len, d_model].



.. py:class:: Time2VecEmbedding(d_model: int, use_cos: bool = True)

   Bases: :py:obj:`BaseEmbedding`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: use_cos
      :value: True



   .. py:attribute:: linear


   .. py:attribute:: periodic


   .. py:method:: forward(batch_size: int, seq_len: int, past_key_values_length: int = 0, **_)


.. py:class:: ALiBiPositionalBias(num_heads: int, max_seq_len: int = 2048)

   Bases: :py:obj:`BaseEmbedding`


   Attention with linear biases (ALiBi) instead of positional embeddings.


   .. py:attribute:: max_seq_len
      :value: 2048



   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      Generate ALiBi bias tensor for causal attention.

      :param batch_size: Batch size.
      :param seq_len: Sequence length.

      :returns: Tensor [1, num_heads, seq_len, seq_len].



.. py:class:: BucketedRelativeBias(num_heads: int, num_buckets: int = 32, max_distance: int = 128)

   Bases: :py:obj:`BaseEmbedding`


   Bucketed relative positional biases.


   .. py:attribute:: num_buckets
      :value: 32



   .. py:attribute:: max_distance
      :value: 128



   .. py:attribute:: relative_buckets


   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      Compute bucketed relative bias.

      :param batch_size: Batch size.
      :param seq_len: Sequence length.

      :returns: Tensor [1, num_heads, seq_len, seq_len].



.. py:class:: ConvolutionalPositionalEmbedding(d_model: int, kernel_size: int = 3, max_seq_len: int = 2048)

   Bases: :py:obj:`BaseEmbedding`


   Convolutional enhancement of sinusoidal embeddings.


   .. py:attribute:: base


   .. py:attribute:: conv


   .. py:method:: forward(batch_size: int, seq_len: int, past_key_values_length: int = 0) -> torch.Tensor

      Apply conv to base positional embeddings.

      :param batch_size: Batch size.
      :param seq_len: Sequence length.
      :param past_key_values_length: Offset index.

      :returns: Tensor [B, seq_len, d_model].



.. py:class:: TimeDeltaEmbedding(d_model: int, hidden_dim: int = 64)

   Bases: :py:obj:`BaseEmbedding`


   Embedding for time delta features via an MLP.


   .. py:attribute:: mlp


   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      Embed relative time deltas.

      :param batch_size: Batch size.
      :param seq_len: Sequence length.

      :returns: Tensor [B, seq_len, d_model].



.. py:class:: StackedPositionalEmbedding(d_model: int, embedding_configs: List[Dict[str, Any]], builder: temporal.models.module_builder_helper.ModuleBuilder)

   Bases: :py:obj:`BaseEmbedding`


   Wrapper to stack multiple positional embeddings sequentially by summing them.
   This wrapper intelligently passes arguments to its sub-modules, preventing
   errors when sub-modules have different forward signatures.


   .. py:attribute:: embeddings


   .. py:method:: forward(**kwargs) -> torch.Tensor

      Sum outputs of configured embeddings, intelligently passing only supported arguments.

      :param \*\*kwargs: A dictionary of arguments that might be needed by any of the
                         sub-embeddings. This can include `batch_size`, `seq_len`,
                         `past_key_values_length`, `x` (for RoPE), `device`, etc.

      :returns: Tensor of shape [B, seq_len, d_model] representing the summed embeddings.



.. py:class:: NoneEmbedding(d_model: int, **kwargs)

   Bases: :py:obj:`BaseEmbedding`


   A placeholder embedding that returns a zero tensor on the correct device.
   This effectively disables the positional embedding when used in a model
   configuration.


   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      Returns a zero tensor of the correct shape and on the correct device.

      :param batch_size: The batch size of the input.
      :param seq_len: The sequence length of the input.
      :param \*\*kwargs: Additional arguments (ignored).

      :returns: A zero tensor of shape [batch_size, seq_len, d_model] on the
                correct device.



.. py:class:: S4PositionalEmbedding(d_model: int, kernel_size: int = 512, max_seq_len: int = 4096)

   Bases: :py:obj:`BaseEmbedding`


   S4-inspired positional embedding using a learned long convolution kernel.
   Projects identity input over time (like a learned filter bank) and returns
   a [B, T, d_model] positional signal.


   .. py:attribute:: kernel_size
      :value: 512



   .. py:attribute:: max_seq_len
      :value: 4096



   .. py:attribute:: filter


   .. py:attribute:: skip


   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      Generate positional embeddings.

      :param batch_size: Batch size.
      :param seq_len: Sequence length.

      :returns: Tensor of shape [B, T, d_model].



.. py:class:: WaveletPositionalEmbedding(d_model: int, wavelet: str = 'db4', level: int = 3, max_seq_len: int = 2048, **kwargs)

   Bases: :py:obj:`BaseEmbedding`


   Wavelet-based positional embedding that projects identity-position impulses
   through a learned multi-resolution filter bank (DWT basis).

   This is a fixed-time-index embedding (like sinusoidal) but with multi-scale
   locality and better modeling of transients.

   :param d_model: Output embedding dimension.
   :type d_model: int
   :param wavelet: Wavelet type (e.g., 'db1', 'haar', 'coif1').
   :type wavelet: str
   :param level: Decomposition level (e.g., 3).
   :type level: int
   :param max_seq_len: Maximum supported sequence length.
   :type max_seq_len: int


   .. py:attribute:: wavelet
      :value: 'db4'



   .. py:attribute:: level
      :value: 3



   .. py:attribute:: max_seq_len
      :value: 2048



   .. py:attribute:: input_dim


   .. py:attribute:: proj


   .. py:method:: forward(batch_size: int, seq_len: int, **kwargs) -> torch.Tensor

      :param batch_size: Batch size.
      :param seq_len: Current sequence length.

      :returns: Tensor of shape [B, T, d_model]



