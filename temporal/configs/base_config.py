from dataclasses import dataclass, asdict, fields, field
from typing import Type, TypeVar, Dict, Any, Optional

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
        for f in fields(self):
            if isinstance(getattr(self, f.name), BaseConfig):
                data[f.name] = getattr(self, f.name).to_dict()
            elif isinstance(getattr(self, f.name), list):
                data[f.name] = [
                    item.to_dict() if isinstance(item, BaseConfig) else item
                    for item in getattr(self, f.name)
                ]
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Creates a dataclass instance from a dictionary. It's designed to be
        forward-compatible by ignoring unknown keys and allowing new fields
        to be added to dataclasses with default values.
        """
        if "type" in data and data["type"] in CONFIG_REGISTRY:
            target_cls = CONFIG_REGISTRY[data["type"]]
        else:
            target_cls = cls

        # Get all fields of the target dataclass that can be initialized.
        # This is the set of "known" parameters for the target version.
        known_keys = {f.name for f in fields(target_cls) if f.init}

        # Separate known keys from unknown keys. Unknown keys will be stored in 'kwargs'.
        known_data = {k: v for k, v in data.items() if k in known_keys}
        unknown_data = {k: v for k, v in data.items() if k not in known_keys}

        # If the target class has a 'kwargs' field, store the unknown data there.
        if 'kwargs' in known_keys:
            known_data['kwargs'] = unknown_data
        elif unknown_data:
            # If there are unknown keys but no 'kwargs' field, you might want to log this.
            print(f"Warning: Discarding unknown keys for {target_cls.__name__}: {list(unknown_data.keys())}")


        # Recursively build nested configs
        for f in fields(target_cls):
            if f.name in known_data:
                field_value = known_data[f.name]
                field_type = f.type

                # Handle Optional[Type]
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                    field_type = field_type.__args__[0]

                if isinstance(field_value, dict) and hasattr(field_type, "from_dict") and issubclass(field_type, BaseConfig):
                    known_data[f.name] = field_type.from_dict(field_value)
                elif isinstance(field_value, list):
                    list_items = []
                    # Determine the type of list elements
                    list_item_type = None
                    if hasattr(f.type, '__args__') and f.type.__args__:
                        list_item_type = f.type.__args__[0]
                    
                    for item in field_value:
                        if isinstance(item, dict) and list_item_type and hasattr(list_item_type, "from_dict") and issubclass(list_item_type, BaseConfig):
                            list_items.append(list_item_type.from_dict(item))
                        else:
                            list_items.append(item)
                    known_data[f.name] = list_items

        return target_cls(**known_data)

    def __post_init__(self):
        """
        Method for post-initialization validation. Subclasses should override this.
        """
        pass