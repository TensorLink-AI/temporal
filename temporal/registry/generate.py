# temporal/registry/generate.py
from typing import Dict, Type, Callable, List

# The global registry dictionary that stores all registered generation methods.
GENERATE_REGISTRY: Dict[str, Type] = {}

def register_generate(name: str) -> Callable:
    """
    A decorator to register a generation method class.

    This decorator adds a class, which should implement a generation strategy
    (e.g., autoregressive decoding, parallel sampling), to the global generation
    registry. This allows different generation methods to be selected by name.

    Example:
        @register_generate("autoregressive")
        class AutoregressiveGenerator:
            ...

    Args:
        name (str): The unique name for the generation method.

    Returns:
        Callable: The wrapper function that performs the registration.
    """
    def wrapper(cls: Type) -> Type:
        """The actual wrapper that registers the class."""
        if name in GENERATE_REGISTRY:
            print(f"Warning: Generation method '{name}' is being overridden.")
        GENERATE_REGISTRY[name] = cls
        return cls
    return wrapper

def resolve_generate(name: str) -> Type:
    """
    Retrieves a registered generation method class from the registry.

    Args:
        name (str): The name of the generation method to resolve.

    Returns:
        Type: The registered generation class.

    Raises:
        KeyError: If the `name` is not registered.
    """
    if name not in GENERATE_REGISTRY:
        available = list_generate_registered()
        raise KeyError(f"Generation method not found: '{name}'. Available methods: {available}")
    return GENERATE_REGISTRY[name]

def list_generate_registered() -> List[str]:
    """
    Lists all registered generation method names.

    Returns:
        List[str]: A list of the names of all registered generation methods.
    """
    return list(GENERATE_REGISTRY.keys())
