temporal.models.mixin.autoregressive
====================================

.. py:module:: temporal.models.mixin.autoregressive


Attributes
----------

.. autoapisummary::

   temporal.models.mixin.autoregressive.logger


Classes
-------

.. autoapisummary::

   temporal.models.mixin.autoregressive.AutoregressiveDispatchMixin


Module Contents
---------------

.. py:data:: logger

.. py:class:: AutoregressiveDispatchMixin

   A mixin that dynamically dispatches calls to the correct implementation
   (patch-based or stepwise) by inspecting the model's components at runtime.

   This should be placed *first* in the inheritance list of a model.


   .. py:method:: forecast(inputs: torch.Tensor, prediction_length: int, quantiles: Optional[List[float]] = None, **kwargs) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]

      A user-friendly, dispatching forecast method.



   .. py:method:: generate(*args, **kwargs) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]

      Dispatches the expert-level `generate` call to the correct implementation.



