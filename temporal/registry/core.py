# temporal/registry/core.py

MODULE_REGISTRY = {
    "attention": {},
    "feedforward": {},
    "embedding": {},
    "head_agg": {},
    "normalization": {},
    "loss": {},
    "block": {},
    "output_head": {},  # ✅ NEW

}

def register_module(kind: str, name: str):
    """
    Decorator to register a module under a specific kind and name.
    Example: @register_module("attention", "full")
    """
    def wrapper(cls):
        if kind not in MODULE_REGISTRY:
            raise ValueError(f"Unknown module kind: {kind}")
        MODULE_REGISTRY[kind][name] = cls
        return cls
    return wrapper

def resolve(kind: str, name: str):
    """
    Returns the registered class for the given kind and name.
    Raises KeyError if not found.
    """
    if kind not in MODULE_REGISTRY:
        raise ValueError(f"Unknown module kind: {kind}")
    if name not in MODULE_REGISTRY[kind]:
        raise KeyError(f"Module not found: kind={kind}, name={name}")
    return MODULE_REGISTRY[kind][name]

def list_registered(kind: str):
    return list(MODULE_REGISTRY.get(kind, {}).keys())
