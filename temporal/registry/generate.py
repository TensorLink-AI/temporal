# temporal/registry/generate.py
GENERATE_REGISTRY = {}

def register_generate(name):
    def wrapper(cls):
        GENERATE_REGISTRY[name] = cls
        return cls
    return wrapper

def resolve_generate(name):
    return GENERATE_REGISTRY[name]
