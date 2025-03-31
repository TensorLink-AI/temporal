import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, List, Dict, Any, Union, Tuple,Callable
import math
from datetime import datetime

#Placeholder
def generate_time_features(timestamps, freq): 
    return None 

class TimeSeriesIterableDataset(IterableDataset):
    """
    A streaming dataset that:
      1. Converts each row into multiple AR examples by sliding windows of length `config.context_length`.
      2. Optionally applies a user-provided `transform` to the target sequence.
      3. Optionally applies a user-provided `transform_dynamic` to dynamic features.
      4. Can do Reversible Instance Normalization (ReVIN) on the target if `revin=True`.
    
    Expects `dataset` to be a list of dicts like:
      {
        'target': [float, float, ...],
        'feat_dynamic_real': optional 2D array or list,
        'feat_static_cat': optional 1D array or list,
        'freq': optional string (e.g., '1H'),
        'start_date': optional string or datetime,
        'item_id': optional
      }
    """
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        config: Any,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform_dynamic: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        revin: bool = False,
        stride: int = 1,
    ):
        """
        Args:
            dataset: list of data samples (dict).
            config: must have .context_length (int).
            transform: a callable applied to the `target` tensor,
                       e.g., log-diff, min-max scaling, etc.
            transform_dynamic: a callable applied to dynamic_features 
                               (shape [seq_len, feat_dim]) if desired.
            revin: whether to apply Reversible Instance Normalization on the target
                   per window: (x - mean) / std
            stride: how many steps to advance the sliding window each time.
        """
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.transform = transform
        self.transform_dynamic = transform_dynamic
        self.revin = revin
        self.stride = stride
        
        # Cache dataset length for convenience
        self._length = len(dataset)

    def __len__(self):
        return self._length

    def __iter__(self):
        # For each item in the dataset, produce multiple sub-windows,
        # and yield them one by one.
        for idx, item in enumerate(self.dataset):
            windowed_samples = self._process_item(idx, item)
            for sample in windowed_samples:
                yield sample

    def _process_item(self, idx: int, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Return a list of AR training examples (windows) from a single item.
        Each window is `config.context_length` steps.
        """
        try:
            # 1) Extract target
            target = item.get("target", None)
            if target is None:
                return []
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)

            # 2) Apply the transformation if provided
            #    e.g. log-diff, min-max scaling, etc.
            if self.transform is not None:
                target = self.transform(target)

            # 3) Possibly skip if not enough data
            context_len = getattr(self.config, 'context_length', 48)
            full_length = len(target)
            if full_length < context_len:
                return []

            # 4) Prepare dynamic features
            dynamic_features = None
            if 'feat_dynamic_real' in item and item['feat_dynamic_real'] is not None:
                dyn = item['feat_dynamic_real']
                if not isinstance(dyn, torch.Tensor):
                    dyn = torch.tensor(dyn, dtype=torch.float32)
                # Match length with target (truncate if longer)
                if dyn.shape[0] > full_length:
                    dyn = dyn[: full_length]
                # Optional user-provided transform for dynamic feats
                if self.transform_dynamic is not None:
                    dyn = self.transform_dynamic(dyn)
                dynamic_features = dyn

            # 5) Prepare static categorical features
            static_cat_features = None
            if 'feat_static_cat' in item and item['feat_static_cat'] is not None:
                cat_vals = item['feat_static_cat']
                if not isinstance(cat_vals, torch.Tensor):
                    cat_vals = torch.tensor(cat_vals, dtype=torch.long)
                static_cat_features = cat_vals

            # 6) Possibly generate time-based features
            #    if 'freq' and 'start_date' exist
            if 'freq' in item and item['freq'] is not None:
                freq = item['freq']
                start_date = item.get('start_date', datetime.now())
                # Build timestamps for the length of target
                timestamps = pd.date_range(start=start_date, periods=full_length, freq=freq)
                time_feats = generate_time_features(timestamps, freq)
                if time_feats is not None:
                    if dynamic_features is not None:
                        dynamic_features = torch.cat([dynamic_features, time_feats], dim=1)
                    else:
                        dynamic_features = time_feats

            # 7) Slide across 'target' in windows of length `context_len`
            windows = []
            for start_idx in range(0, full_length - context_len + 1, self.stride):
                end_idx = start_idx + context_len
                # window of length context_len
                window_target = target[start_idx:end_idx]

                window_dyn = None
                if dynamic_features is not None:
                    window_dyn = dynamic_features[start_idx:end_idx, :]

                # -- Reversible Instance Normalization if requested --
                # We'll store the mean/std used for each window in case the user
                # wants to invert it later. In training, usually we don't invert, but
                # you might want to store these stats.
                revin_stats = None
                if self.revin:
                    mean = window_target.mean(dim=0, keepdim=True)
                    std = window_target.std(dim=0, keepdim=True)
                    std = torch.where(std == 0, torch.tensor(1.0, device=std.device), std)  # avoid div by 0
                    window_target = (window_target - mean) / std
                    revin_stats = {
                        "mean": mean,
                        "std": std,
                    }

                sample = {
                    "input_ids": window_target,
                    "dynamic_features": window_dyn,
                    "static_cat_features": static_cat_features,
                    "item_id": item.get("item_id", f"item_{idx}"),
                    "freq": item.get("freq", "1D"),
                }

                if revin_stats is not None:
                    sample["revin_stats"] = revin_stats

                windows.append(sample)

            return windows

        except Exception as e:
            print(f"Error in _process_item({idx}): {str(e)}")
            return []




def timeseries_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate streamed samples into a single batch with padding if needed."""
    # Filter out empty or invalid samples
    samples = [s for s in samples if s and 'input_ids' in s]
    if not samples:
        return {}
    
    # We'll handle up to 3 main things: input_ids, dynamic_features, static_cat_features
    # Step 1: Find max length
    lengths = [s['input_ids'].shape[0] for s in samples]
    max_len = max(lengths)

    # Prepare storage
    batch_input_ids = []
    batch_attention_mask = []
    batch_dynamic_feats = []
    batch_dynamic_mask = []
    batch_static_cat = []
    
    # Pad input_ids
    for s in samples:
        seq_len = s['input_ids'].shape[0]
        # pad the input_ids
        padded_input = torch.cat([
            s['input_ids'],
            torch.zeros(max_len - seq_len, dtype=torch.float32)
        ])
        batch_input_ids.append(padded_input)

        # attention mask
        attn_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        batch_attention_mask.append(attn_mask)
    
    # Handle dynamic features
    has_dyn = any(s['dynamic_features'] is not None for s in samples)
    if has_dyn:
        # We assume each sample has shape [seq_len, feature_dim]
        # We find the max feature_dim from the first or a check
        # For simplicity, assume consistent feature_dim
        first_dyn = None
        for s in samples:
            if s['dynamic_features'] is not None:
                first_dyn = s['dynamic_features']
                break
        feature_dim = first_dyn.shape[1] if first_dyn is not None else 0

        for s in samples:
            if s['dynamic_features'] is None:
                # all zeros
                dyn_pad = torch.zeros(max_len, feature_dim)
                batch_dynamic_feats.append(dyn_pad)
                dyn_mask = torch.zeros(max_len, dtype=torch.long)
                batch_dynamic_mask.append(dyn_mask)
                continue
            
            seq_len = s['dynamic_features'].shape[0]
            # pad dynamic
            needed = max_len - seq_len
            if needed > 0:
                pad_block = torch.zeros(needed, feature_dim)
                dyn_pad = torch.cat([s['dynamic_features'], pad_block], dim=0)
            else:
                dyn_pad = s['dynamic_features']
            batch_dynamic_feats.append(dyn_pad)

            # dynamic mask
            dyn_m = torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(needed, dtype=torch.long)
            ])
            batch_dynamic_mask.append(dyn_m)

    # Handle static cat
    has_static = any(s['static_cat_features'] is not None for s in samples)
    if has_static:
        # We'll just stack them. 
        for s in samples:
            if s['static_cat_features'] is None:
                # Create a dummy 
                batch_static_cat.append(torch.tensor([], dtype=torch.long))
            else:
                # Ensure it's a 1D or 0D
                cat_feat = s['static_cat_features']
                if cat_feat.dim() == 0:
                    cat_feat = cat_feat.unsqueeze(0)
                batch_static_cat.append(cat_feat)

    # Stack final
    batch_dict = {
        'input_ids': torch.stack(batch_input_ids, dim=0),           # [B, max_len]
        'attention_mask': torch.stack(batch_attention_mask, dim=0), # [B, max_len]
    }
    if has_dyn:
        batch_dict['dynamic_features'] = torch.stack(batch_dynamic_feats, dim=0)       # [B, max_len, feat_dim]
        batch_dict['dynamic_feature_mask'] = torch.stack(batch_dynamic_mask, dim=0)    # [B, max_len]

    if has_static:
        # Might have variable shapes if they're not consistent
        # For now, assume they share shape [something]
        # We'll just stack on dim=0
        # If the shape is different, you'd store them in a list or handle carefully.
        batch_static_cat = torch.nn.utils.rnn.pad_sequence(batch_static_cat, batch_first=True)
        batch_dict['static_cat_features'] = batch_static_cat

    return batch_dict

