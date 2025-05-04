# temporal/utils/hf_accessors.py

import os
import json
import torch

# Optional safetensors support
try:
    from safetensors.torch import save_file as _safetensors_save
    from safetensors.torch import load_file as _safetensors_load
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


def save_hf(
    model: torch.nn.Module,
    config, # Expecting an instance like TransformerTimeSeriesConfig here
    save_directory: str,
    safe: bool = False
):
    """
    Save a PyTorch model + HF-compatible config to a directory.

    Args:
        model:      Your nn.Module (state_dict will be saved).
        config:     An instance of PretrainedConfig (e.g. TransformerTimeSeriesConfig).
                    Its model_type might be overridden.
        save_directory: Path to write files into.
        safe:       If True, uses safetensors (requires the safetensors library).
                    Otherwise uses torch.save.
    """
    os.makedirs(save_directory, exist_ok=True)

    # --- MODIFICATION START ---
    # 1) Get config dict and OVERRIDE model_type
    config_dict = config.to_dict()
    # Explicitly set the model type the builder/registry expects for this model
    # This should match the key used in @register_generate in transformer_model.py
    # Assuming the key is 'transformer'
    config_dict["model_type"] = "transformer"
    # --- MODIFICATION END ---

    # 2) Save config.json using the MODIFIED dict
    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, "w") as f:
        # Dump the modified dictionary
        json.dump(config_dict, f, indent=2)

    # 3) Save weights
    weights_path = os.path.join(save_directory, "model.safetensors" if safe else "pytorch_model.bin")
    if safe:
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors library is required to save in safe format")
        # ensure CPU tensors
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        _safetensors_save(sd, weights_path)
    else:
        torch.save(model.state_dict(), weights_path)

    print(f"Model saved to {save_directory}")


def load_hf(
    save_directory: str,
    model_cls,
    config_cls,
    safe: bool = False,
    map_location="cpu",
    **model_kwargs
):
    """
    Load a model + config saved with `save_hf`.

    Args:
        save_directory: Path where config.json and weights live.
        model_cls:      Class of your model, signature __init__(config, **model_kwargs).
        config_cls:     HF config class with `.from_pretrained()` or `.from_dict()`.
        safe:           If True, attempts to load `model.safetensors`.
        map_location:   Passed to torch.load.
        model_kwargs:   Extra args forwarded to model_cls(config, **model_kwargs).

    Returns:
        model:          An instance of model_cls with loaded weights.
    """
    # 1) Load config
    from transformers import PretrainedConfig, AutoConfig

    # Use HF’s loading if available
    try:
        cfg = config_cls.from_pretrained(save_directory)
    except Exception:
        # fallback to manual read
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = config_cls.from_dict(cfg_dict)

    # 2) Instantiate model
    model = model_cls(cfg, **model_kwargs)

    # 3) Load weights
    if safe:
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed; cannot load safe tensors.")
        weights_path = os.path.join(save_directory, "model.safetensors")
        sd = _safetensors_load(weights_path)
    else:
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        sd = torch.load(weights_path, map_location=map_location)

    model.load_state_dict(sd)
    return model
