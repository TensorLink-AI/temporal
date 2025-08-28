temporal.configs.base_config
============================

.. py:module:: temporal.configs.base_config


Attributes
----------

.. autoapisummary::

   temporal.configs.base_config.T
   temporal.configs.base_config.CONFIG_REGISTRY


Classes
-------

.. autoapisummary::

   temporal.configs.base_config.BaseConfig


Functions
---------

.. autoapisummary::

   temporal.configs.base_config.register_config_type


Module Contents
---------------

.. py:data:: T

.. py:data:: CONFIG_REGISTRY
   :type:  Dict[str, Type[BaseConfig]]

.. py:function:: register_config_type(config_type: str)

   A decorator to register configuration dataclasses with a given type string.


.. py:class:: BaseConfig

   Base class for all immutable configuration dataclasses.
   Provides common serialization/deserialization methods.


   .. py:attribute:: type
      :type:  str


   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: to_dict() -> Dict[str, Any]

      Converts the dataclass instance to a dictionary, handling nested BaseConfig objects.



   .. py:method:: from_dict(data: Dict[str, Any]) -> T
      :classmethod:


      Creates a dataclass instance from a dictionary. It's designed to be
      forward-compatible by ignoring unknown keys and allowing new fields
      to be added to dataclasses with default values.



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



