# temporal/registry/core.py
from typing import Dict, Type, Callable, List

# The global registry dictionary that stores all registered modules.
MODULE_REGISTRY: Dict[str, Dict[str, Type]] = {
    "attention": {},
    "feedforward": {},
    "embedding": {},
    "head_agg": {},
    "normalization": {},
    "loss": {},
    "block": {},
    "output_head": {},
}

def register_module(kind: str, name: str) -> Callable:
    """
    A decorator to register a module class in the global registry.

    This decorator is the primary mechanism for adding new components (like
    attention mechanisms, feed-forward networks, etc.) to the framework's
    registry, making them available to be instantiated from a configuration file.

    Example:
        @register_module("attention", "my_custom_attention")
        class MyCustomAttention(nn.Module):
            ...

    Args:
        kind (str): The category of the module (e.g., "attention", "loss").
            This must be a pre-defined key in the `MODULE_REGISTRY`.
        name (str): The unique name for the module within its kind.

    Returns:
        Callable: The wrapper function that performs the registration.
    """
    def wrapper(cls: Type) -> Type:
        """The actual wrapper that registers the class."""
        if kind not in MODULE_REGISTRY:
            raise ValueError(f"Unknown module kind: '{kind}'. Must be one of {list(MODULE_REGISTRY.keys())}")
        if name in MODULE_REGISTRY[kind]:
            print(f"Warning: Module '{name}' of kind '{kind}' is being overridden.")
        MODULE_REGISTRY[kind][name] = cls
        return cls
    return wrapper

def resolve(kind: str, name: str) -> Type:
    """
    Retrieves a registered module class from the registry.

    This function is used by the model builders to look up and instantiate the
    appropriate class based on a name provided in a configuration.

    Args:
        kind (str): The category of the module to resolve.
        name (str): The name of the module to resolve.

    Returns:
        Type: The registered module class.

    Raises:
        ValueError: If the `kind` does not exist in the registry.
        KeyError: If the `name` is not registered for the given `kind`.
    """
    if kind not in MODULE_REGISTRY:
        raise ValueError(f"Unknown module kind: '{kind}'. Cannot resolve module.")
    if name not in MODULE_REGISTRY[kind]:
        available = list_registered(kind)
        raise KeyError(f"Module not found for kind='{kind}', name='{name}'. Available modules: {available}")
    return MODULE_REGISTRY[kind][name]

def list_registered(kind: str) -> List[str]:
    """
    Lists all registered module names for a given kind.

    This is a helper function useful for debugging and introspection, allowing
    a user to see what modules are available for a particular category.

    Args:
        kind (str): The category of modules to list.

    Returns:
        List[str]: A list of the names of all registered modules of the given kind.
    """
    return list(MODULE_REGISTRY.get(kind, {}).keys())
