import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import math
from datetime import datetime

# Placeholder
def generate_time_features(timestamps, freq):
    return None

class TimeSeriesIterableDataset(IterableDataset):
    """
    A streaming dataset that:
      1. Converts each row into multiple AR examples by sliding windows.
      2. Optionally applies user-provided transforms to target or dynamic features.
      3. Optional Reversible Instance Normalization (ReVIN).
      4. Produces BOTH `input_ids` (context window) and `labels` (future window).
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
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.transform = transform
        self.transform_dynamic = transform_dynamic
        self.revin = revin
        self.stride = stride

        # Cache dataset length for convenience
        self._length = len(dataset)

        if not hasattr(self.config, "context_length"):
            raise ValueError("config must have 'context_length' attribute.")
        if not hasattr(self.config, "prediction_length"):
            raise ValueError("config must have 'prediction_length' attribute.")

    def __len__(self):
        return self._length

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            windowed_samples = self._process_item(idx, item)
            for sample in windowed_samples:
                yield sample

    def _process_item(self, idx: int, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Produce multiple (input_ids, labels) windows from a single item:
          input window = length `config.context_length`
          label window = length `config.prediction_length`
        """
        try:
            # 1) Extract target
            target = item.get("target", None)
            if target is None:
                return []
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)

            # 2) Transform if needed
            if self.transform is not None:
                target = self.transform(target)

            context_len = getattr(self.config, "context_length", 48)
            pred_len = getattr(self.config, "prediction_length", 1)
            full_length = len(target)

            # Must have enough data for context + prediction
            if full_length < context_len + pred_len:
                return []

            # 3) Dynamic features
            dynamic_features = None
            if "feat_dynamic_real" in item and item["feat_dynamic_real"] is not None:
                dyn = item["feat_dynamic_real"]
                if not isinstance(dyn, torch.Tensor):
                    dyn = torch.tensor(dyn, dtype=torch.float32)
                # Truncate if longer
                if dyn.shape[0] > full_length:
                    dyn = dyn[:full_length]
                # Transform if given
                if self.transform_dynamic is not None:
                    dyn = self.transform_dynamic(dyn)
                dynamic_features = dyn

            # 4) Static categorical features
            static_cat_features = None
            if "feat_static_cat" in item and item["feat_static_cat"] is not None:
                cat_vals = item["feat_static_cat"]
                if not isinstance(cat_vals, torch.Tensor):
                    cat_vals = torch.tensor(cat_vals, dtype=torch.long)
                static_cat_features = cat_vals

            # 5) Time-based features if freq & start_date exist
            if "freq" in item and item["freq"] is not None:
                freq = item["freq"]
                start_date = item.get("start_date", datetime.now())
                timestamps = pd.date_range(start=start_date, periods=full_length, freq=freq)
                time_feats = generate_time_features(timestamps, freq)
                if time_feats is not None:
                    if dynamic_features is not None:
                        dynamic_features = torch.cat([dynamic_features, time_feats], dim=1)
                    else:
                        dynamic_features = time_feats

            # 6) Slide windows
            # We'll generate windows so we have at least context_len + pred_len
            # from start_idx => start_idx + context_len (input)
            # and start_idx + context_len => start_idx + context_len + pred_len (labels)
            windows = []
            max_start = full_length - (context_len + pred_len) + 1
            for start_idx in range(0, max_start, self.stride):
                in_start = start_idx
                in_end = start_idx + context_len
                out_end = in_end + pred_len

                window_input = target[in_start:in_end]   # shape [context_len]
                window_label = target[in_end:out_end]    # shape [pred_len]

                if window_input.ndim == 1:
                    # for univariate, expand last dim so shape => [context_len, 1]
                    window_input = window_input.unsqueeze(-1)
                    window_label= window_label.unsqueeze(-1)

                # dynamic feats if present
                window_dyn_in = None
                window_dyn_out = None
                if dynamic_features is not None:
                    # slice same ranges for dynamic?
                    # typical approach: "input dynamic feats" => context_len, "label dynamic feats" => pred_len
                    # or you might store them combined
                    window_dyn_in = dynamic_features[in_start:in_end, :]
                    window_dyn_out = dynamic_features[in_end:out_end, :]

                # Reversible Instance Normalization
                revin_stats = None
                if self.revin:
                    mean = window_input.mean(dim=0, keepdim=True)
                    std = window_input.std(dim=0, keepdim=True)
                    std = torch.where(std == 0, torch.tensor(1.0, device=std.device), std)
                    window_input = (window_input - mean) / std
                    # optionally also transform window_label?
                    # depends on how you define ReVIN logic for forecast
                    revin_stats = {"mean": mean, "std": std}

                sample = {
                    "input_ids": window_input,             # shape [context_len]
                    "labels": window_label,                # shape [pred_len]
                    "dynamic_features_input": window_dyn_in,   # shape [context_len, feat_dim]
                    "dynamic_features_label": window_dyn_out,  # shape [pred_len, feat_dim]
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
    """Collate streamed samples into a single batch with padding for both input_ids and labels."""
    # Filter out empty or invalid samples
    samples = [s for s in samples if s and 'input_ids' in s]
    if not samples:
        return {}

    # 1) Gather input lengths
    input_lengths = [s['input_ids'].shape[0] for s in samples]
    max_input_len = max(input_lengths)

    # 2) Gather label lengths
    label_lengths = []
    for s in samples:
        if 'labels' in s and s['labels'] is not None:
            label_lengths.append(s['labels'].shape[0])
        else:
            label_lengths.append(0)
    max_label_len = max(label_lengths)

    # Prepare storages
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_label_mask = []

    # ---- Pad input_ids ----
    for s in samples:
        seq_len = s['input_ids'].shape[0]
        padded_input = torch.cat([
            s['input_ids'],
            torch.zeros(max_input_len - seq_len, dtype=torch.float32)
        ])
        batch_input_ids.append(padded_input)

        attn_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(max_input_len - seq_len, dtype=torch.long)
        ])
        batch_attention_mask.append(attn_mask)

    # ---- Pad labels ----
    has_labels = any('labels' in s for s in samples)
    if has_labels:
        for s in samples:
            if 'labels' not in s or s['labels'] is None:
                # no labels => zeros
                batch_labels.append(torch.zeros(max_label_len, dtype=torch.float32))
                batch_label_mask.append(torch.zeros(max_label_len, dtype=torch.long))
            else:
                lab_len = s['labels'].shape[0]
                padded_label = torch.cat([
                    s['labels'],
                    torch.zeros(max_label_len - lab_len, dtype=torch.float32)
                ])
                batch_labels.append(padded_label)
                lab_mask = torch.cat([
                    torch.ones(lab_len, dtype=torch.long),
                    torch.zeros(max_label_len - lab_len, dtype=torch.long)
                ])
                batch_label_mask.append(lab_mask)

    # ---- Handle dynamic features if present ----
    has_dyn_in = any(s.get('dynamic_features_input') is not None for s in samples)
    has_dyn_out = any(s.get('dynamic_features_label') is not None for s in samples)
    batch_dyn_in, batch_dyn_out = [], []
    batch_dyn_mask_in, batch_dyn_mask_out = [], []

    if has_dyn_in:
        # We assume shape [seq_len, feat_dim] for dynamic_features_input
        # pad each sample to [max_input_len, feat_dim]
        # similarly for dynamic_features_label -> [max_label_len, feat_dim]
        # or you might prefer to store them combined in a single tensor.
        first_in = None
        for s in samples:
            if s.get('dynamic_features_input') is not None:
                first_in = s['dynamic_features_input']
                break
        feat_dim_in = first_in.shape[1] if first_in is not None else 0

        for s in samples:
            dyn_in = s.get('dynamic_features_input')
            if dyn_in is None:
                dyn_in_pad = torch.zeros(max_input_len, feat_dim_in)
                mask_in = torch.zeros(max_input_len, dtype=torch.long)
            else:
                in_len = dyn_in.shape[0]
                needed_in = max_input_len - in_len
                if needed_in > 0:
                    pad_block_in = torch.zeros(needed_in, feat_dim_in)
                    dyn_in_pad = torch.cat([dyn_in, pad_block_in], dim=0)
                else:
                    dyn_in_pad = dyn_in
                mask_in = torch.cat([
                    torch.ones(in_len, dtype=torch.long),
                    torch.zeros(needed_in, dtype=torch.long)
                ])
            batch_dyn_in.append(dyn_in_pad)
            batch_dyn_mask_in.append(mask_in)

    if has_dyn_out:
        # dynamic_features_label
        first_out = None
        for s in samples:
            if s.get('dynamic_features_label') is not None:
                first_out = s['dynamic_features_label']
                break
        feat_dim_out = first_out.shape[1] if first_out is not None else 0

        for s in samples:
            dyn_out = s.get('dynamic_features_label')
            if dyn_out is None:
                dyn_out_pad = torch.zeros(max_label_len, feat_dim_out)
                mask_out = torch.zeros(max_label_len, dtype=torch.long)
            else:
                out_len = dyn_out.shape[0]
                needed_out = max_label_len - out_len
                if needed_out > 0:
                    pad_block_out = torch.zeros(needed_out, feat_dim_out)
                    dyn_out_pad = torch.cat([dyn_out, pad_block_out], dim=0)
                else:
                    dyn_out_pad = dyn_out
                mask_out = torch.cat([
                    torch.ones(out_len, dtype=torch.long),
                    torch.zeros(needed_out, dtype=torch.long)
                ])
            batch_dyn_out.append(dyn_out_pad)
            batch_dyn_mask_out.append(mask_out)

    # ---- Handle static cat if present ----
    has_static = any(s.get('static_cat_features') is not None for s in samples)
    batch_static_cat = []
    if has_static:
        for s in samples:
            cat_feat = s.get('static_cat_features')
            if cat_feat is None:
                batch_static_cat.append(torch.tensor([], dtype=torch.long))
            else:
                if cat_feat.dim() == 0:
                    cat_feat = cat_feat.unsqueeze(0)
                batch_static_cat.append(cat_feat)

    # ---- Stack up final batch dict ----
    batch_dict = {
        'input_ids': torch.stack(batch_input_ids, dim=0),           # [B, max_input_len]
        'attention_mask': torch.stack(batch_attention_mask, dim=0), # [B, max_input_len]
    }
    # after you stack/pad batch_input_ids



    if has_labels:
        batch_dict['labels'] = torch.stack(batch_labels, dim=0)             # [B, max_label_len]
        batch_dict['labels_mask'] = torch.stack(batch_label_mask, dim=0)    # [B, max_label_len]

    if has_dyn_in:
        batch_dict['dynamic_features_input'] = torch.stack(batch_dyn_in, dim=0)   # [B, max_input_len, feat_dim_in]
        batch_dict['dynamic_features_input_mask'] = torch.stack(batch_dyn_mask_in, dim=0)

    if has_dyn_out:
        batch_dict['dynamic_features_label'] = torch.stack(batch_dyn_out, dim=0)  # [B, max_label_len, feat_dim_out]
        batch_dict['dynamic_features_label_mask'] = torch.stack(batch_dyn_mask_out, dim=0)

    if has_static:
        batch_static_cat = torch.nn.utils.rnn.pad_sequence(batch_static_cat, batch_first=True)
        batch_dict['static_cat_features'] = batch_static_cat

    return batch_dict
