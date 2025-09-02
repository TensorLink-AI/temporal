from dataclasses import dataclass, asdict, fields, field
from typing import Type, TypeVar, Dict, Any, Optional, get_origin, get_args

T = TypeVar('T', bound='BaseConfig')

# Global registry for configuration classes
CONFIG_REGISTRY: Dict[str, Type['BaseConfig']] = {}

def register_config_type(config_type: str):
    """
    A decorator to register configuration dataclasses with a given type string.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        CONFIG_REGISTRY[config_type] = cls
        return cls
    return decorator


def _maybe_to_config(val: Any) -> Any:
    """
    Recursively convert dictionaries/lists that look like configs into BaseConfig instances.
    Works for values that appear inside kwargs as well as field values.
    """
    if isinstance(val, dict):
        t = val.get("type")
        if isinstance(t, str) and t in CONFIG_REGISTRY:
            return CONFIG_REGISTRY[t].from_dict(val)  # delegate to that class' loader
        # Not a registered config dict; still recurse into nested structures
        return {k: _maybe_to_config(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_maybe_to_config(v) for v in val]
    return val


@dataclass(frozen=True, kw_only=True)
class BaseConfig:
    """
    Base class for all immutable configuration dataclasses.
    Provides common serialization/deserialization methods.
    """
    type: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary, handling nested BaseConfig objects."""
        data = asdict(self)
        # Recurse over fields to convert nested configs to dicts
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, BaseConfig):
                data[f.name] = v.to_dict()
            elif isinstance(v, list):
                data[f.name] = [item.to_dict() if isinstance(item, BaseConfig) else item for item in v]
            elif isinstance(v, dict) and f.name == "kwargs":
                # Also convert any BaseConfig instances stored inside kwargs
                converted = {}
                for k, item in v.items():
                    if isinstance(item, BaseConfig):
                        converted[k] = item.to_dict()
                    elif isinstance(item, list):
                        converted[k] = [x.to_dict() if isinstance(x, BaseConfig) else x for x in item]
                    else:
                        converted[k] = item
                data[f.name] = converted
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Creates a dataclass instance from a dictionary. It's designed to be
        forward-compatible by ignoring unknown keys and allowing new fields
        to be added to dataclasses with default values.
        """
        # Resolve the concrete class via registry if possible
        if "type" in data and data["type"] in CONFIG_REGISTRY:
            target_cls = CONFIG_REGISTRY[data["type"]]
        else:
            target_cls = cls

        # Known constructor keys for the chosen class
        known_keys = {f.name for f in fields(target_cls) if f.init}

        # --- FIX: Unpack kwargs before initialization (top-level wins) ---
        init_data = dict(data)  # shallow copy
        kwargs_data = init_data.pop("kwargs", {}) or {}
        if not isinstance(kwargs_data, dict):
            # tolerate non-dict kwargs silently
            kwargs_data = {}
        # Merge kwargs first, then top-level (top-level has priority)
        init_data = {**kwargs_data, **init_data}

        # Partition into known vs unknown (unknown → will be placed back into kwargs)
        known_data: Dict[str, Any] = {}
        unknown_data: Dict[str, Any] = {}
        for k, v in init_data.items():
            if k in known_keys:
                known_data[k] = v
            else:
                unknown_data[k] = v

        # If the class has a kwargs field, preserve all unknown keys there
        if "kwargs" in known_keys:
            # Recursively convert any nested config-like dicts inside kwargs
            known_data["kwargs"] = _maybe_to_config(unknown_data)
        elif unknown_data:
            print(f"Warning: Discarding unknown keys for {target_cls.__name__}: {list(unknown_data.keys())}")

        # Recursively build nested configs for declared fields (excluding kwargs which we handled)
        for f in fields(target_cls):
            name = f.name
            if name not in known_data or name == "kwargs":
                continue

            field_value = known_data[name]
            field_type = f.type

            # Normalize Optional[T] → T
            origin = get_origin(field_type)
            if origin is Optional:
                args = [a for a in get_args(field_type) if a is not type(None)]
                field_type = args[0] if args else field_type
                origin = get_origin(field_type)

            # If this field itself is a config, delegate using its from_dict
            if isinstance(field_value, dict) and hasattr(field_type, "from_dict") and issubclass(field_type, BaseConfig):
                known_data[name] = field_type.from_dict(field_value)

            # Lists of configs
            elif isinstance(field_value, list):
                list_item_type = None
                args = get_args(field_type) if get_args(field_type) else ()
                if args:
                    list_item_type = args[0]
                built_list = []
                for item in field_value:
                    if isinstance(item, dict) and list_item_type and hasattr(list_item_type, "from_dict") and issubclass(list_item_type, BaseConfig):
                        built_list.append(list_item_type.from_dict(item))
                    else:
                        built_list.append(_maybe_to_config(item))  # still upgrade any nested configs
                known_data[name] = built_list
            else:
                # Upgrade any nested config-like dicts even for plain fields (dict-of-configs etc.)
                known_data[name] = _maybe_to_config(field_value)

        return target_cls(**known_data)

    def __post_init__(self):
        """
        Method for post-initialization validation. Subclasses should override this.
        """
        pass
