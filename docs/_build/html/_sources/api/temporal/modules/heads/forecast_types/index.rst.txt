temporal.modules.heads.forecast_types
=====================================

.. py:module:: temporal.modules.heads.forecast_types


Attributes
----------

.. autoapisummary::

   temporal.modules.heads.forecast_types.Tensor
   temporal.modules.heads.forecast_types.Params


Classes
-------

.. autoapisummary::

   temporal.modules.heads.forecast_types.HeadExtras
   temporal.modules.heads.forecast_types.ForecastBundle


Module Contents
---------------

.. py:data:: Tensor

.. py:data:: Params

.. py:class:: HeadExtras

   .. py:attribute:: paths
      :type:  Optional[Tensor]
      :value: None



   .. py:attribute:: path_logits
      :type:  Optional[Tensor]
      :value: None



   .. py:attribute:: components
      :type:  Optional[List[str]]
      :value: None



   .. py:attribute:: meta
      :type:  Dict[str, Any]


.. py:class:: ForecastBundle

   A unified container for anything a caller might reasonably want.
   Shapes:
     point:      [B,T,F]
     quantiles:  [B,T,F,Q] or None
     params:     Tensor or Dict[str,Tensor] you actually predicted from


   .. py:attribute:: point
      :type:  Tensor


   .. py:attribute:: quantiles
      :type:  Optional[Tensor]


   .. py:attribute:: params
      :type:  Optional[Params]


   .. py:attribute:: extras
      :type:  HeadExtras


