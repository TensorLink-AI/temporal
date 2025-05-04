# temporal/utils/hf_accessors.py
import os
import json
import torch
from typing import Optional

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
    config, # Expecting an instance like TransformerTimeSeriesConfig here
    save_directory: str,
    safe: bool = False,
    # --- New parameters for Hugging Face Hub ---
    repo_id: Optional[str] = None,
    commit_message: Optional[str] = "Save model using custom save_hf",
    private: bool = False, # Used ONLY when creating the repo if it doesn't exist
    token: Optional[str] = None, # Use HF_TOKEN env var or login if None
    push_to_hub: bool = False # Set to True to enable pushing
    # --- End new parameters ---
):
    """
    Save a PyTorch model + HF-compatible config to a directory,
    optionally creating and/or pushing to HF Hub.

    Args:
        model: Your nn.Module (state_dict will be saved).
        config: An instance of PretrainedConfig (e.g. TransformerTimeSeriesConfig).
                Its model_type might be overridden.
        save_directory: Path to write files into locally.
        safe: If True, uses safetensors. Otherwise uses torch.save.
        repo_id (Optional[str]): Repository ID on Hugging Face Hub (e.g., 'your-username/your-model-name').
                                 Required if push_to_hub is True.
        commit_message (Optional[str]): Commit message for the Hub upload.
        private (bool): If True, creates the repository as private if it doesn't exist.
                        This argument is IGNORED if the repository already exists.
        token (Optional[str]): Hugging Face API token. Uses logged-in user or HF_TOKEN env var if None.
        push_to_hub (bool): If True, attempts to create the repo (if needed) and uploads
                          the `save_directory` to the specified `repo_id` after saving locally.
    """
    os.makedirs(save_directory, exist_ok=True)

    # 1) Get config dict and potentially OVERRIDE model_type
    config_dict = config.to_dict()
    # Ensure model_type matches the registered name if needed by your builder
    if "model_type" not in config_dict or config_dict["model_type"] is None:
         # Example: Set a default if missing, adjust as needed
         # config_dict["model_type"] = "transformer"
         pass # Or raise an error if it's mandatory

    # Your existing modification (ensure this is correct for your use case)
    config_dict["model_type"] = "transformer"

    # 2) Save config.json
    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # 3) Save weights
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

    # --- Push to Hub ---
    if push_to_hub:
        if not _HAS_HUGGINGFACE_HUB:
             raise ImportError("huggingface_hub library is required to push to Hub. Please install it (`pip install huggingface_hub`).")
        if not repo_id:
            raise ValueError("`repo_id` must be specified when `push_to_hub=True`.")

        api = HfApi(token=token)

        # Check if repo exists, create if it doesn't
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
                    exist_ok=False # Don't error if it was created between check and now
                )
                print(f"Successfully created repository '{repo_id}'.")
            except Exception as create_e:
                print(f"Error creating repository '{repo_id}': {create_e}")
                raise create_e # Re-raise creation error
        except Exception as e: # Catch other potential errors during repo_info check
            print(f"Error checking repository status: {e}")
            raise e

        # Proceed with upload
        print(f"Pushing contents of {save_directory} to repository: {repo_id}...")
        try:
            api_url = upload_folder(
                folder_path=save_directory,
                repo_id=repo_id,
                commit_message=commit_message,
                # private=private, # Removed argument - visibility set at creation or on Hub
                token=token,
                repo_type="model" # Assuming it's a model
            )
            print(f"Push successful. Model uploaded to: {api_url}")
        except Exception as e:
            # Provide more context in case of upload error after creation attempt
            print(f"Error pushing to Hub repository '{repo_id}': {e}")
            print("Please ensure you have write permissions and the token is valid.")
            raise e
    # --- End Push to Hub ---

# ... (keep load_hf function below)

def load_hf(
    model_name_or_path: str, # Changed from save_directory
    model_cls,
    config_cls,
    safe: bool = False, # Check for safetensors file if True
    map_location="cpu",
    # --- New parameters for Hugging Face Hub ---
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    token: Optional[str] = None, # Use HF_TOKEN env var or login if None
    # --- End new parameters ---
    **model_kwargs
):
    """
    Load a model + config saved with `save_hf`, from local path or HF Hub.

    Args:
        model_name_or_path (str): Can be either:
            - Path to a local directory containing config.json and model weights.
            - A Hugging Face Hub repository ID (e.g., 'your-username/your-model-name').
        model_cls: Class of your model, signature __init__(config, **model_kwargs).
        config_cls: Config class with `.from_dict()` or `.from_pretrained()` capability.
        safe (bool): If True, attempts to load `model.safetensors`, otherwise `pytorch_model.bin`.
        map_location: Passed to torch.load (if not using safetensors).
        cache_dir (Optional[str]): Path to Hugging Face cache directory for downloads.
        force_download (bool): Whether to force download from Hub, even if cached.
        token (Optional[str]): Hugging Face API token for private repos.
        model_kwargs: Extra args forwarded to model_cls(config, **model_kwargs).

    Returns:
        model: An instance of model_cls with loaded weights.
    """
    load_path = model_name_or_path
    is_local = os.path.isdir(load_path)
    resolved_from_hub = False

    # --- Download from Hub if not a local directory ---
    if not is_local:
        if not _HAS_HUGGINGFACE_HUB:
            raise ImportError("huggingface_hub library is required to load from Hub. Please install it (`pip install huggingface_hub`).")

        print(f"Attempting to download '{model_name_or_path}' from Hugging Face Hub...")
        try:
            # snapshot_download downloads the whole repo content
            load_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                repo_type="model" # Assuming it's a model
            )
            print(f"Files downloaded to cache: {load_path}")
            resolved_from_hub = True
        except Exception as e:
            raise ValueError(f"Could not download repository '{model_name_or_path}' from Hub. Please ensure it's a valid repository ID and you have access.") from e
    else:
        print(f"Loading from local directory: {load_path}")
    # --- End Hub download ---

    # --- Load Config ---
    config_path = os.path.join(load_path, "config.json")
    if not os.path.exists(config_path):
         raise FileNotFoundError(f"Config file 'config.json' not found in {load_path}.")

    try:
        # Try HF's loading first if available and config_cls supports it
        # Note: This might fail if config_cls isn't a true HF PretrainedConfig subclass
        # Use token if loading potentially private config from hub
        cfg = config_cls.from_pretrained(load_path, token=token if resolved_from_hub else None)
        print("Config loaded using from_pretrained.")
    except (AttributeError, TypeError, Exception): # Catch broad exceptions as from_pretrained might fail variously
         print("from_pretrained failed or not available for config, falling back to manual load.")
         # Fallback to manual JSON loading
         with open(config_path, "r") as f:
             cfg_dict = json.load(f)
         # Try from_dict if available, otherwise assume constructor works
         if hasattr(config_cls, "from_dict"):
              cfg = config_cls.from_dict(cfg_dict)
         else:
              # This assumes your config_cls can be initialized from a dict directly
              try:
                  cfg = config_cls(**cfg_dict)
              except TypeError as e:
                  raise ValueError(f"Could not instantiate config class {config_cls.__name__} from dictionary. Ensure it has a from_dict method or accepts the config keys as __init__ arguments.") from e
    # --- End Load Config ---


    # 2) Instantiate model
    # Pass the loaded config object to your model class constructor
    model = model_cls(cfg, **model_kwargs)


    # 3) Load weights
    weights_name = "model.safetensors" if safe else "pytorch_model.bin"
    weights_path = os.path.join(load_path, weights_name)

    if not os.path.exists(weights_path):
         # If the primary weight file isn't found, try the alternative
         alt_weights_name = "pytorch_model.bin" if safe else "model.safetensors"
         alt_weights_path = os.path.join(load_path, alt_weights_name)
         if os.path.exists(alt_weights_path):
             print(f"Warning: Requested {'safetensors' if safe else 'pytorch'} but found {alt_weights_name}. Loading found file.")
             weights_path = alt_weights_path
             safe = not safe # Update safe flag based on found file
         else:
            # If loading from hub, give a specific message about potential missing files in repo
            location_msg = f"in downloaded repository '{model_name_or_path}'" if resolved_from_hub else f"in local directory {load_path}"
            raise FileNotFoundError(f"Could not find weight file '{weights_name}' or '{alt_weights_name}' {location_msg}.")


    print(f"Loading weights from: {weights_path}")
    if safe:
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed; cannot load safe tensors.")
        # Load using safetensors
        sd = _safetensors_load(weights_path, device=map_location) # Pass map_location to device
    else:
        # torch.load allows specifying map_location
        sd = torch.load(weights_path, map_location=map_location)

    # Load the state dict into the instantiated model
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
         print(f"Error loading state_dict: {e}")
         print("This might happen if the model architecture definition does not match the saved weights.")
         # You might want to add more specific error handling here if needed
         raise

    print(f"Model loaded successfully from {model_name_or_path}{' (via Hub)' if resolved_from_hub else ' (local)'}.")
    return model
