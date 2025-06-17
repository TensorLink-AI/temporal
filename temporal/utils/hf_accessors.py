# temporal/utils/hf_accessors.py
import os
import json
import torch
from typing import Optional, Type

# Optional safetensors support
try:
    from safetensors.torch import save_file as _safetensors_save
    from safetensors.torch import load_file as _safetensors_load
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

# Added for Hub interaction
try:
    from huggingface_hub import (
        upload_folder,
        snapshot_download,
        create_repo,
        HfApi
    )
    from huggingface_hub.utils import RepositoryNotFoundError
    _HAS_HUGGINGFACE_HUB = True
except ImportError:
    _HAS_HUGGINGFACE_HUB = False


def save_hf(
    model: torch.nn.Module,
    config,
    save_directory: str,
    safe: bool = False,
    repo_id: Optional[str] = None,
    commit_message: Optional[str] = "Save model using custom save_hf",
    private: bool = False,
    token: Optional[str] = None,
    push_to_hub: bool = False
):
    """
    Saves a model and its configuration to a directory, with optional Hub upload.

    This function saves a model's state dictionary and its configuration file
    in a format that is compatible with the Hugging Face ecosystem. It can
    also create a new repository on the Hub and upload the saved files.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        config: The configuration object for the model. It should have a
            `to_dict()` method.
        save_directory (str): The local directory where the model and config
            will be saved.
        safe (bool): If True, saves the model weights using `safetensors`.
            Otherwise, uses `torch.save` (pickle format). Defaults to False.
        repo_id (Optional[str]): The ID of the repository on the Hugging Face
            Hub (e.g., 'username/repo_name'). Required if `push_to_hub` is True.
        commit_message (Optional[str]): The commit message for the upload.
        private (bool): Whether to create the repository as private if it
            doesn't exist. This is ignored if the repo already exists.
        token (Optional[str]): Your Hugging Face API token. If None, it uses
            the token from the environment or login cache.
        push_to_hub (bool): If True, the function will push the `save_directory`
            to the specified `repo_id` on the Hub.

    Raises:
        ImportError: If `safetensors` or `huggingface_hub` is required but not installed.
        ValueError: If `push_to_hub` is True but `repo_id` is not provided.
    """
    os.makedirs(save_directory, exist_ok=True)

    config_dict = config.to_dict()
    config_dict["model_type"] = "transformer"

    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    weights_name = "model.safetensors" if safe else "pytorch_model.bin"
    weights_path = os.path.join(save_directory, weights_name)
    if safe:
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors library is required to save in safe format")
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        _safetensors_save(sd, weights_path)
    else:
        torch.save(model.state_dict(), weights_path)

    print(f"Model saved locally to {save_directory}")

    if push_to_hub:
        if not _HAS_HUGGINGFACE_HUB:
             raise ImportError("huggingface_hub library is required to push to Hub. Please install it (`pip install huggingface_hub`).")
        if not repo_id:
            raise ValueError("`repo_id` must be specified when `push_to_hub=True`.")

        api = HfApi(token=token)

        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            print(f"Repository '{repo_id}' already exists on the Hub.")
        except RepositoryNotFoundError:
            print(f"Repository '{repo_id}' not found. Attempting to create it...")
            try:
                create_repo(
                    repo_id=repo_id,
                    token=token,
                    private=private,
                    repo_type="model",
                    exist_ok=False
                )
                print(f"Successfully created repository '{repo_id}'.")
            except Exception as create_e:
                print(f"Error creating repository '{repo_id}': {create_e}")
                raise create_e
        except Exception as e:
            print(f"Error checking repository status: {e}")
            raise e

        print(f"Pushing contents of {save_directory} to repository: {repo_id}...")
        try:
            api_url = upload_folder(
                folder_path=save_directory,
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
                repo_type="model"
            )
            print(f"Push successful. Model uploaded to: {api_url}")
        except Exception as e:
            print(f"Error pushing to Hub repository '{repo_id}': {e}")
            print("Please ensure you have write permissions and the token is valid.")
            raise e


def load_hf(
    model_name_or_path: str,
    model_cls: Type[torch.nn.Module],
    config_cls,
    safe: bool = False,
    map_location="cpu",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    token: Optional[str] = None,
    **model_kwargs
):
    """
    Loads a model and configuration from a local path or the Hugging Face Hub.

    This function can load models saved with `save_hf`. It automatically handles
    downloading files from the Hub if the `model_name_or_path` is a repository ID.

    Args:
        model_name_or_path (str): The path to a local directory or a repository
            ID on the Hugging Face Hub.
        model_cls (Type[torch.nn.Module]): The class of the model to instantiate.
            It is expected to have an `__init__(self, config, **kwargs)` signature.
        config_cls: The configuration class for the model. It should have a
            `from_pretrained` or `from_dict` method.
        safe (bool): If True, prioritizes loading `model.safetensors`.
        map_location (str): The device to load the model weights onto (e.g., 'cpu', 'cuda:0').
        cache_dir (Optional[str]): The directory for caching downloaded Hub files.
        force_download (bool): If True, forces a re-download from the Hub.
        token (Optional[str]): Your Hugging Face API token for private repos.
        **model_kwargs: Additional keyword arguments to pass to the model's constructor.

    Returns:
        torch.nn.Module: An instance of `model_cls` with the loaded weights.
    """
    load_path = model_name_or_path
    is_local = os.path.isdir(load_path)
    resolved_from_hub = False

    if not is_local:
        if not _HAS_HUGGINGFACE_HUB:
            raise ImportError("huggingface_hub library is required to load from Hub. Please install it (`pip install huggingface_hub`).")

        print(f"Attempting to download '{model_name_or_path}' from Hugging Face Hub...")
        try:
            load_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                repo_type="model"
            )
            print(f"Files downloaded to cache: {load_path}")
            resolved_from_hub = True
        except Exception as e:
            raise ValueError(f"Could not download repository '{model_name_or_path}' from Hub. Please ensure it's a valid repository ID and you have access.") from e
    else:
        print(f"Loading from local directory: {load_path}")

    config_path = os.path.join(load_path, "config.json")
    if not os.path.exists(config_path):
         raise FileNotFoundError(f"Config file 'config.json' not found in {load_path}.")

    try:
        cfg = config_cls.from_pretrained(load_path, token=token if resolved_from_hub else None)
        print("Config loaded using from_pretrained.")
    except (AttributeError, TypeError, Exception):
         print("from_pretrained failed or not available for config, falling back to manual load.")
         with open(config_path, "r") as f:
             cfg_dict = json.load(f)
         if hasattr(config_cls, "from_dict"):
              cfg = config_cls.from_dict(cfg_dict)
         else:
              try:
                  cfg = config_cls(**cfg_dict)
              except TypeError as e:
                  raise ValueError(f"Could not instantiate config class {config_cls.__name__} from dictionary. Ensure it has a from_dict method or accepts the config keys as __init__ arguments.") from e

    model = model_cls(cfg, **model_kwargs)

    weights_name = "model.safetensors" if safe else "pytorch_model.bin"
    weights_path = os.path.join(load_path, weights_name)

    if not os.path.exists(weights_path):
         alt_weights_name = "pytorch_model.bin" if safe else "model.safetensors"
         alt_weights_path = os.path.join(load_path, alt_weights_name)
         if os.path.exists(alt_weights_path):
             print(f"Warning: Requested {'safetensors' if safe else 'pytorch'} but found {alt_weights_name}. Loading found file.")
             weights_path = alt_weights_path
             safe = not safe
         else:
            location_msg = f"in downloaded repository '{model_name_or_path}'" if resolved_from_hub else f"in local directory {load_path}"
            raise FileNotFoundError(f"Could not find weight file '{weights_name}' or '{alt_weights_name}' {location_msg}.")


    print(f"Loading weights from: {weights_path}")
    if safe:
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors is not installed, but is required to load .safetensors files.")
        sd = _safetensors_load(weights_path, device=map_location)
    else:
        sd = torch.load(weights_path, map_location=map_location)

    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
         print(f"Error loading state_dict: {e}")
         print("This can happen if the model architecture definition does not match the saved weights.")
         raise

    print(f"Model loaded successfully from {model_name_or_path}{' (via Hub)' if resolved_from_hub else ' (local)'}.")
    return model
