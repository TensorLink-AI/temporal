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
    type: str # This field is required and will be keyword-only
    kwargs: Dict[str, Any] = field(default_factory=dict) # Added kwargs to BaseConfig

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary, handling nested BaseConfig objects."""
        data = asdict(self)
        # Recursively convert nested BaseConfig objects to dictionaries
        for f in fields(self):
            if isinstance(getattr(self, f.name), BaseConfig):
                data[f.name] = getattr(self, f.name).to_dict()
            elif isinstance(getattr(self, f.name), list):
                # Handle lists of BaseConfig objects
                data[f.name] = [
                    item.to_dict() if isinstance(item, BaseConfig) else item
                    for item in getattr(self, f.name)
                ]
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Creates a dataclass instance from a dictionary, using the registry for polymorphic types.
        """
        if "type" in data and data["type"] in CONFIG_REGISTRY:
            target_cls = CONFIG_REGISTRY[data["type"]]
        else:
            target_cls = cls

        # Filter out keys not in the constructor's signature
        # This handles cases where `data` might contain extra keys (e.e., from old configs)
        # For kw_only=True dataclasses, all fields are in init=True by default unless specified
        valid_keys = {f.name for f in fields(target_cls) if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        # Recursively build nested configs if they are dicts
        for f in fields(target_cls):
            if f.name in filtered_data and isinstance(filtered_data[f.name], dict):
                # If the field's type is a BaseConfig subclass, use its from_dict
                if hasattr(f.type, "from_dict") and issubclass(f.type, BaseConfig):
                    filtered_data[f.name] = f.type.from_dict(filtered_data[f.name])
                # Handle Optional[BaseConfig]
                elif hasattr(f.type, '__origin__') and f.type.__origin__ is Optional and \
                     hasattr(f.type.__args__[0], "from_dict") and issubclass(f.type.__args__[0], BaseConfig):
                    filtered_data[f.name] = f.type.__args__[0].from_dict(filtered_data[f.name])

            elif f.name in filtered_data and isinstance(filtered_data[f.name], list):
                # Handle lists of nested configs
                list_items = []
                # Determine the type of list elements from the field's type annotation
                list_field_type = None
                if hasattr(f.type, '__args__') and f.type.__args__:
                    # This assumes homogeneous lists of BaseConfig types
                    list_field_type = f.type.__args__[0]

                for item in filtered_data[f.name]:
                    if isinstance(item, dict):
                        if "type" in item and item["type"] in CONFIG_REGISTRY:
                            list_items.append(CONFIG_REGISTRY[item["type"]].from_dict(item))
                        elif list_field_type and hasattr(list_field_type, "from_dict") and issubclass(list_field_type, BaseConfig):
                            # Fallback if item dict doesn't have a 'type' but list is typed
                            list_items.append(list_field_type.from_dict(item))
                        else:
                            list_items.append(item)
                    else:
                        list_items.append(item)
                filtered_data[f.name] = list_items

        return target_cls(**filtered_data)

    def __post_init__(self):
        """
        Method for post-initialization validation. Subclasses should override this.
        """
        pass
