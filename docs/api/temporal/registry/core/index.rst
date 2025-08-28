temporal.registry.core
======================

.. py:module:: temporal.registry.core


Attributes
----------

.. autoapisummary::

   temporal.registry.core.MODULE_REGISTRY


Functions
---------

.. autoapisummary::

   temporal.registry.core.register_module
   temporal.registry.core.resolve
   temporal.registry.core.list_registered


Module Contents
---------------

.. py:data:: MODULE_REGISTRY
   :type:  Dict[str, Dict[str, Type]]

.. py:function:: register_module(kind: str, name: str) -> Callable

   A decorator to register a module class in the global registry.

   This decorator is the primary mechanism for adding new components (like
   attention mechanisms, feed-forward networks, etc.) to the framework's
   registry, making them available to be instantiated from a configuration file.

   .. rubric:: Example

   @register_module("attention", "my_custom_attention")
   class MyCustomAttention(nn.Module):
       ...

   :param kind: The category of the module (e.g., "attention", "loss").
                This must be a pre-defined key in the `MODULE_REGISTRY`.
   :type kind: str
   :param name: The unique name for the module within its kind.
   :type name: str

   :returns: The wrapper function that performs the registration.
   :rtype: Callable


.. py:function:: resolve(kind: str, name: str) -> Type

   Retrieves a registered module class from the registry.

   This function is used by the model builders to look up and instantiate the
   appropriate class based on a name provided in a configuration.

   :param kind: The category of the module to resolve.
   :type kind: str
   :param name: The name of the module to resolve.
   :type name: str

   :returns: The registered module class.
   :rtype: Type

   :raises ValueError: If the `kind` does not exist in the registry.
   :raises KeyError: If the `name` is not registered for the given `kind`.


.. py:function:: list_registered(kind: str) -> List[str]

   Lists all registered module names for a given kind.

   This is a helper function useful for debugging and introspection, allowing
   a user to see what modules are available for a particular category.

   :param kind: The category of modules to list.
   :type kind: str

   :returns: A list of the names of all registered modules of the given kind.
   :rtype: List[str]


