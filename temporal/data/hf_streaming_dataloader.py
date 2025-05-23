import os
import json
import io
import urllib.request
import torch
import pandas as pd
import pyarrow as pa
from dateutil.relativedelta import relativedelta  # Keep if actually used for freq, though not in current logic
from huggingface_hub import list_repo_files, hf_hub_url
from torch.utils.data import IterableDataset, DataLoader

# Default cache directory, can be overridden by config
DEFAULT_CACHE_DIR = "/mnt/data/hf_cache"

class HFStreamingDataConfig:
    """
    Configuration for the Hugging Face streaming dataset.
    """
    def __init__(self,
                 hf_dataset_name: str = "liamsbhoo/GiftEvalPretrain",
                 hf_repo_type: str = "dataset",
                 context_length: int = 48,
                 prediction_length: int = 12,
                 stride: int = 1,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 manifest_filename: str = "manifest.json",
                 target_key: str = "target",
                 dynamic_feats_key: str = "past_feat_dynamic_real",
                 start_timestamp_key: str = "start",
                 freq_key: str = "freq",
                 force_reload_manifest: bool = False,
                 **kwargs): # For future expansion or unhandled params
        self.hf_dataset_name = hf_dataset_name
        self.hf_repo_type = hf_repo_type
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.cache_dir = cache_dir
        self.manifest_filename = manifest_filename
        self.manifest_fp = os.path.join(self.cache_dir, self.manifest_filename)
        self.target_key = target_key
        self.dynamic_feats_key = dynamic_feats_key
        self.start_timestamp_key = start_timestamp_key
        self.freq_key = freq_key
        self.force_reload_manifest = force_reload_manifest
        self.kwargs = kwargs

    def to_dict(self):
        return self.__dict__

class FlexibleWindowedDataset(IterableDataset):
    def __init__(self, config: HFStreamingDataConfig):
        super().__init__()
        self.config = config
        self.shards = self._load_shards_list()

    def _load_shards_list(self):
        """Loads the list of .arrow files from the Hugging Face Hub dataset repository."""
        os.makedirs(self.config.cache_dir, exist_ok=True)
        if not self.config.force_reload_manifest and os.path.exists(self.config.manifest_fp):
            try:
                with open(self.config.manifest_fp, "r") as f:
                    all_files = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Manifest file {self.config.manifest_fp} is corrupted. Reloading.")
                all_files = self._fetch_and_cache_repo_files()
        else:
            all_files = self._fetch_and_cache_repo_files()
        
        return [f for f in all_files if f.endswith(".arrow")]

    def _fetch_and_cache_repo_files(self):
        """Fetches file list from Hugging Face Hub and caches it."""
        print(f"Fetching file list for {self.config.hf_dataset_name}...")
        try:
            all_files = list_repo_files(self.config.hf_dataset_name, repo_type=self.config.hf_repo_type)
            with open(self.config.manifest_fp, "w") as f:
                json.dump(all_files, f)
            return all_files
        except Exception as e:
            print(f"Error fetching or caching repository files for {self.config.hf_dataset_name}: {e}")
            # Fallback to an empty list if fetching fails, so the dataset doesn't break entirely,
            # though it will be empty.
            return []


    def __iter__(self):
        for shard_file in self.shards:
            try:
                url   = hf_hub_url(self.config.hf_dataset_name, shard_file, repo_type=self.config.hf_repo_type)
                buf   = urllib.request.urlopen(url).read() # Consider adding timeout
                table = pa.ipc.open_stream(io.BytesIO(buf)).read_all()
                batch_data = table.to_pydict()
            except Exception as e:
                print(f"Skipping shard {shard_file} due to error during download/read: {e}")
                continue

            num_series = len(batch_data.get(self.config.target_key, []))
            targets    = batch_data.get(self.config.target_key, [])
            dyn_feats_all  = batch_data.get(self.config.dynamic_feats_key, [None]*num_series)
            starts_all     = batch_data.get(self.config.start_timestamp_key, [None]*num_series)
            freqs_all      = batch_data.get(self.config.freq_key,  [None]*num_series)

            for j in range(num_series):
                series_target = targets[j]
                if series_target is None:
                    continue
                
                series_tensor = torch.tensor(series_target, dtype=torch.float32)
                L, P = self.config.context_length, self.config.prediction_length
                
                if series_tensor.size(0) < L + P:
                    continue

                series_dyn_feats_list = dyn_feats_all[j]
                series_dyn_feats = torch.tensor(series_dyn_feats_list, dtype=torch.float32) if series_dyn_feats_list is not None else None

                start_time = starts_all[j]
                freq_str   = freqs_all[j]
                full_ts    = None
                if start_time is not None and freq_str is not None:
                    pd_freq = freq_str.replace("T", "min") # "T" is standard for minutes in pandas freq strings
                    try:
                        full_ts = pd.date_range(start=str(start_time), periods=series_tensor.size(0), freq=pd_freq)
                    except ValueError: # Handles invalid start/freq
                        try:
                            # Fallback to inferring frequency if the provided one fails
                            full_ts = pd.date_range(start=str(start_time), periods=series_tensor.size(0), freq="infer")
                        except Exception as ts_err:
                            # print(f"Could not generate timestamps for series {j} in {shard_file}: {ts_err}")
                            full_ts = None


                for i in range(0, series_tensor.size(0) - L - P + 1, self.config.stride):
                    window_ts = list(full_ts[i : i + L]) if full_ts is not None else None
                    
                    # Dynamic features handling:
                    # This assumes `series_dyn_feats` are either static (e.g., [num_features])
                    # or already aligned/windowed if they are time-varying (e.g., [num_features, total_series_length]).
                    # If `series_dyn_feats` is [num_features, total_series_length], it should be sliced:
                    # current_dyn_feats = series_dyn_feats[:, i : i + L] if series_dyn_feats is not None and series_dyn_feats.ndim == 2 else series_dyn_feats
                    # For simplicity, current code passes the whole `series_dyn_feats` if available.
                    # Modify as per actual structure of your dynamic features.
                    current_dyn_feats = series_dyn_feats 
                    if series_dyn_feats is not None and series_dyn_feats.ndim == 2 and series_dyn_feats.shape[-1] == series_tensor.size(0):
                        # Assuming [num_dynamic_features, sequence_length]
                        current_dyn_feats = series_dyn_feats[:, i : i+L]
                    elif series_dyn_feats is not None and series_dyn_feats.ndim == 1:
                         # Assuming static features for the entire series [num_dynamic_features]
                         pass # Keep as is

                    yield {
                        "input_ids":     series_tensor[i : i + L],
                        "labels":        series_tensor[i + L : i + L + P],
                        "dynamic_feats": current_dyn_feats,
                        "timestamps":    window_ts,
                    }

def get_flexible_windowed_dataloader(
        data_config: HFStreamingDataConfig,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        collate_fn_override = None, # Allow custom collate_fn
        **dataloader_kwargs # Other DataLoader arguments
    ):
    """
    Creates a DataLoader for the FlexibleWindowedDataset.
    """
    dataset = FlexibleWindowedDataset(config=data_config)
    
    def default_collate_fn(batch_items):
        # Filter out items where essential data might be None (e.g. due to processing errors)
        valid_batch_items = [item for item in batch_items if item["input_ids"] is not None and item["labels"] is not None]
        if not valid_batch_items: # All items in batch were invalid
            # This can happen if all series in a fetched set of windows are too short, or other errors.
            # Depending on strictness, could raise error or return None/empty batch.
            # Returning None and handling it in training loop might be one option.
            # For now, if collate gets an empty list, it will error out with torch.stack.
            # It's better to ensure `FlexibleWindowedDataset` yields valid items,
            # or handle this case explicitly if empty batches are possible.
            # If an empty list is passed to torch.stack it will raise an error.
            # Consider what to do if valid_batch_items is empty.
            # For now, if valid_batch_items is empty, it will likely cause an error later.
            # One could return a dict of empty tensors or specific flags.
             return { # Return a valid empty structure
                "input_ids": torch.empty(0, data_config.context_length),
                "labels": torch.empty(0, data_config.prediction_length),
                "dynamic_feats": [],
                "timestamps": [],
            }


        input_ids_list = [item["input_ids"] for item in valid_batch_items]
        labels_list = [item["labels"] for item in valid_batch_items]
        
        dynamic_feats_list = [item["dynamic_feats"] for item in valid_batch_items]
        collated_dynamic_feats = []
        if any(df is not None for df in dynamic_feats_list):
            # Attempt to stack if all are tensors and shapes match, else keep as list
            # This part needs careful handling based on whether dynamic_feats are per-window or per-series static
            if all(isinstance(df, torch.Tensor) for df in dynamic_feats_list if df is not None):
                # Example: if all are (context_length, num_features) or (num_features)
                try:
                    # Pad if necessary, or ensure they are same shape if stacking
                    # For now, assume they can be stacked directly or model handles list of tensors
                    non_none_dfs = [df for df in dynamic_feats_list if df is not None]
                    if non_none_dfs:
                         # If features are like [L, D_feat] or [D_feat]
                         # Stacking works if all items have same shape.
                         # If shapes vary (e.g. some are [L,D1] others [L,D2] or some None), this will fail.
                         # A more robust collate would pad or handle lists.
                         # This example assumes dynamic_feats for each item in batch are already correctly shaped
                         # and can be stacked or handled as a list of tensors by the model.
                        if all(df.shape == non_none_dfs[0].shape for df in non_none_dfs):
                            collated_dynamic_feats = torch.stack([df if df is not None else torch.zeros_like(non_none_dfs[0]) for df in dynamic_feats_list]) # Example: fill None with zeros
                        else: # shapes differ, keep as list
                            collated_dynamic_feats = dynamic_feats_list
                    else: # all dynamic_feats were None
                        collated_dynamic_feats = [None] * len(valid_batch_items)

                except RuntimeError as e: # Stacking failed (e.g. different shapes)
                    print(f"Note: Could not stack dynamic_feats, keeping as list. Error: {e}")
                    collated_dynamic_feats = dynamic_feats_list
            else: # Mix of types or not all tensors (that are not None)
                collated_dynamic_feats = dynamic_feats_list
        else: # All are None
            collated_dynamic_feats = [None] * len(valid_batch_items)


        timestamps_list = [item["timestamps"] for item in valid_batch_items]

        return {
            "input_ids":     torch.stack(input_ids_list),
            "labels":        torch.stack(labels_list),
            "dynamic_feats": collated_dynamic_feats, # Can be list of Tensors/None, or a stacked Tensor
            "timestamps":    timestamps_list,
        }

    chosen_collate_fn = collate_fn_override if collate_fn_override is not None else default_collate_fn

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0, # persistent_workers only if num_workers > 0
        collate_fn=chosen_collate_fn,
        **dataloader_kwargs
    )
    return loader

if __name__ == "__main__":
    print("Starting dataloader sanity check...")
    
    # 1. Create a configuration object
    data_cfg = HFStreamingDataConfig(
        context_length=48,
        prediction_length=12,
        stride=24, # Example: larger stride
        cache_dir="./.hf_cache_test", # Use a local test cache
        force_reload_manifest=False # Set to True to refetch from HF Hub
    )
    print(f"Using data configuration: {data_cfg.to_dict()}")

    # 2. Get the dataloader instance
    my_loader = get_flexible_windowed_dataloader(
        data_config=data_cfg,
        batch_size=4,
        num_workers=0 # Easier for debugging, set > 0 for performance
    )
    
    print("DataLoader instance created. Attempting to load a batch...")
    
    try:
        batch_count = 0
        for i, batch in enumerate(my_loader):
            if i >= 1: # Check first batch
                break
            batch_count +=1
            print(f"--- Batch {i+1} ---")
            print("Batch loaded successfully.")
            if batch["input_ids"].numel() == 0:
                print("Warning: Loaded an empty batch. This might be due to all series in shards being too short or filtering.")
                continue

            print("Input IDs shape:", batch["input_ids"].shape)
            print("Labels shape:", batch["labels"].shape)
            
            # Print dynamic feats info
            if isinstance(batch["dynamic_feats"], torch.Tensor):
                print("Dynamic feats stacked, shape:", batch["dynamic_feats"].shape)
            elif isinstance(batch["dynamic_feats"], list):
                print(f"Dynamic feats is a list of {len(batch['dynamic_feats'])} items.")
                if batch["dynamic_feats"] and batch["dynamic_feats"][0] is not None:
                    if isinstance(batch["dynamic_feats"][0], torch.Tensor):
                        print("First dynamic feat in list is a Tensor, shape:", batch["dynamic_feats"][0].shape)
                    else:
                         print("First dynamic feat in list type:", type(batch["dynamic_feats"][0]))
                elif batch["dynamic_feats"] and batch["dynamic_feats"][0] is None:
                     print("First dynamic feat in list is None.")

            # Print timestamps info
            print(f"Timestamps is a list of {len(batch['timestamps'])} items.")
            if batch["timestamps"] and batch["timestamps"][0] is not None:
                print("Timestamps for first window (first 5):", batch["timestamps"][0][:5])
            elif batch["timestamps"] and batch["timestamps"][0] is None:
                print("Timestamps for first window are None.")
        
        if batch_count == 0:
            print("No batches were loaded. Check dataset source or filtering criteria.")

    except Exception as e:
        print(f"Error during sanity check: {e}")
        print("This could be due to network issues, problems with the Hugging Face dataset, or configuration errors.")
        import traceback
        traceback.print_exc()

    print("Dataloader sanity check finished.")
