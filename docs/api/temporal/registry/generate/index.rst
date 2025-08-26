temporal.registry.generate
==========================

.. py:module:: temporal.registry.generate


Attributes
----------

.. autoapisummary::

   temporal.registry.generate.GENERATE_REGISTRY


Functions
---------

.. autoapisummary::

   temporal.registry.generate.register_generate
   temporal.registry.generate.resolve_generate
   temporal.registry.generate.list_generate_registered


Module Contents
---------------

.. py:data:: GENERATE_REGISTRY
   :type:  Dict[str, Type]

.. py:function:: register_generate(name: str) -> Callable

   A decorator to register a generation method class.

   This decorator adds a class, which should implement a generation strategy
   (e.g., autoregressive decoding, parallel sampling), to the global generation
   registry. This allows different generation methods to be selected by name.

   .. rubric:: Example

   @register_generate("autoregressive")
   class AutoregressiveGenerator:
       ...

   :param name: The unique name for the generation method.
   :type name: str

   :returns: The wrapper function that performs the registration.
   :rtype: Callable


.. py:function:: resolve_generate(name: str) -> Type

   Retrieves a registered generation method class from the registry.

   :param name: The name of the generation method to resolve.
   :type name: str

   :returns: The registered generation class.
   :rtype: Type

   :raises KeyError: If the `name` is not registered.


.. py:function:: list_generate_registered() -> List[str]

   Lists all registered generation method names.

   :returns: A list of the names of all registered generation methods.
   :rtype: List[str]


