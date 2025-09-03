temporal.modules.norm.dynamic_revin
===================================

.. py:module:: temporal.modules.norm.dynamic_revin


Classes
-------

.. autoapisummary::

   temporal.modules.norm.dynamic_revin.DynamicRevIN


Module Contents
---------------

.. py:class:: DynamicRevIN(num_features: int, eps: float = 1e-05, affine_mode: Union[str, Dict] = 'fixed')

   Bases: :py:obj:`torch.nn.Module`


   Dynamic RevIN for channels-last tensors [B, L, F] with switchable affine:
     - 'fixed'            : learnable γ, β (broadcast [1,1,F])
     - {'type':'dynamic', 'mapper':'linear' | 'mlp', 'hidden_dim':16, 'gamma_positive':True}

   Modes:
     - mode='norm'      : compute stats on x, compute affine, normalize & store stats/affine
     - mode='transform' : normalize using last stored stats/affine (no recompute)
     - mode='denorm'    : invert last stored affine + stats to original scale


   .. py:attribute:: num_features


   .. py:attribute:: eps
      :value: 1e-05



   .. py:attribute:: cfg
      :value: 'fixed'



   .. py:attribute:: affine_type


   .. py:attribute:: gamma_positive


   .. py:method:: forward(x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None) -> torch.Tensor


   .. py:method:: transform(x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor


   .. py:method:: inverse_transform(x: torch.Tensor) -> torch.Tensor


