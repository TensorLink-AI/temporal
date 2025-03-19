import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, List, Dict, Any, Union, Tuple
import math
from datetime import datetime

class TimeSeriesIterableDataset(IterableDataset):
    """
    A streaming dataset for time series data. 
    Expects `dataset` to be an indexable collection of dicts with:
        {
          'target': array-like,
          'feat_dynamic_real': optional array-like,
          'feat_static_cat': optional array-like,
          'freq': optional string,
          'start_date': optional,
          ...
        }
    The `config` should contain context_length, prediction_length, etc.
    """
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        config: Any,
        use_log_diff: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.use_log_diff = use_log_diff
        
        # Attempt to cache dataset length
        self._length = len(dataset)

    def __len__(self):
        return self._length
    
    def __iter__(self):
        """Iterate over dataset items, convert to model-ready format."""
        # In a more advanced scenario, you'd handle multi-worker sharding here.
        for idx, item in enumerate(self.dataset):
            yield self._process_item(idx, item)

    def _process_item(self, idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single raw item into a model-ready sample."""
        try:
            # Extract target
            target = item.get('target', None)
            if target is None:
                return {}
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)
            
            # Optionally apply log-diff
            if self.use_log_diff:
                target = compute_log_diff(target)

            # Possibly skip sequences that are too short
            if len(target) < getattr(self.config, 'context_length', 48):
                # skip
                return {}

            # Prepare dynamic features
            dynamic_features = None
            if 'feat_dynamic_real' in item:
                dyn = item['feat_dynamic_real']
                if not isinstance(dyn, torch.Tensor):
                    dyn = torch.tensor(dyn, dtype=torch.float32)
                # same length check or trimming
                if dyn.shape[0] > target.shape[0]:
                    dyn = dyn[: target.shape[0]]
                dynamic_features = dyn
            
            # Prepare static categorical features
            static_cat_features = None
            if 'feat_static_cat' in item:
                cat_vals = item['feat_static_cat']
                if not isinstance(cat_vals, torch.Tensor):
                    cat_vals = torch.tensor(cat_vals, dtype=torch.long)
                static_cat_features = cat_vals

            # Possibly generate time-based features
            if 'freq' in item:
                freq = item['freq']
                start_date = item.get('start_date', datetime.now())
                # Build timestamps for the length of target
                timestamps = pd.date_range(start=start_date, periods=target.size(0), freq=freq)
                time_feats = generate_time_features(timestamps, freq)
                # If we have dynamic_features, we can concat time feats
                if time_feats is not None:
                    if dynamic_features is not None:
                        # Expand time_feats if needed
                        dynamic_features = torch.cat([dynamic_features, time_feats], dim=1)
                    else:
                        dynamic_features = time_feats

            # Return sample
            sample = {
                'input_ids': target,  # Let the model handle context vs. label shifting
                'dynamic_features': dynamic_features,
                'static_cat_features': static_cat_features,
                'item_id': item.get('item_id', f"item_{idx}"),
                'freq': item.get('freq', '1H'),
            }
            return sample
        except Exception as e:
            print(f"Error in _process_item({idx}): {str(e)}")
            return {}


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


def create_dataloader(config, dataset, batch_size=32, num_workers=0, shuffle=False, use_log_diff=False):
    """
    Creates a streaming DataLoader using TimeSeriesIterableDataset.
    
    Args:
        config: Must have context_length, etc. 
        dataset: A list (or other indexable) of data samples (dict).
        batch_size: batch size
        num_workers: how many processes to load data
        shuffle: doesn't work for iterable, but you could implement your own random sampling
        use_log_diff: whether to apply a log-diff transform
    
    Returns:
        A DataLoader that yields padded batches.
    """
    # Build the iterable dataset
    iterable_dataset = TimeSeriesIterableDataset(
        dataset=dataset,
        config=config,
        use_log_diff=use_log_diff
    )
    # For actual streaming from disk, you'd open a file inside `TimeSeriesIterableDataset`
    # or pass a generator object, etc.

    # We pass timeseries_collate_fn so each mini-batch is properly padded
    loader = DataLoader(
        iterable_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=timeseries_collate_fn,
        shuffle=False,        # Typically not used with IterableDataset
        drop_last=False
    )
    return loader
