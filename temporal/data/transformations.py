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
        self.decoder_labels = None  # Placeholder for decoder labels

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
                    window_label = (window_label - mean) / std
                    # optionally also transform window_label?
                    # depends on how you define ReVIN logic for forecast
                    revin_stats = {"mean": mean, "std": std}


                decoder_input_ids = window_label  # or a shifted variant

                sample = {
                    "input_ids": window_input,             # shape [context_len]
                    "labels": window_label,                # shape [pred_len]
                    "dynamic_features_input": window_dyn_in,   # shape [context_len, feat_dim]
                    "dynamic_features_label": window_dyn_out,  # shape [pred_len, feat_dim]
                    "decoder_input_ids": decoder_input_ids, # multi-step decoder input

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
    """
    Collate streamed samples into a single batch with padding for:
      - input_ids
      - labels
      - (optional) decoder_input_ids (for multi-step teacher forcing)
      - dynamic features in/out
      - static cat features
      etc.
    """

    # 1) Filter out empty or invalid samples
    samples = [s for s in samples if s and 'input_ids' in s]
    if not samples:
        return {}

    # 2) Gather input lengths (for input_ids)
    input_lengths = [s['input_ids'].shape[0] for s in samples]
    max_input_len = max(input_lengths)

    # 3) Gather label lengths
    label_lengths = []
    for s in samples:
        if 'labels' in s and s['labels'] is not None:
            label_lengths.append(s['labels'].shape[0])
        else:
            label_lengths.append(0)
    max_label_len = max(label_lengths)

    # 4) Gather decoder_input_ids lengths if present
    has_decoder_inputs = any('decoder_input_ids' in s for s in samples)
    if has_decoder_inputs:
        decoder_lengths = []
        for s in samples:
            dec_ids = s.get('decoder_input_ids')
            if dec_ids is not None:
                decoder_lengths.append(dec_ids.shape[0])
            else:
                decoder_lengths.append(0)
        max_decoder_len = max(decoder_lengths)
    else:
        max_decoder_len = 0

    # Prepare storages
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_label_mask = []
    batch_decoder_inputs = []

    # ---- Pad input_ids ----
    for s in samples:
        seq_len = s['input_ids'].shape[0]
        to_pad = max_input_len - seq_len

        if s['input_ids'].dim() == 1:
            # shape => [seq_len]
            padded_input = torch.cat([
                s['input_ids'],
                torch.zeros(to_pad, dtype=torch.float32)
            ])
        else:
            # shape => [seq_len, feat_dim]
            feat_dim = s['input_ids'].shape[1]
            padded_input = torch.cat([
                s['input_ids'],
                torch.zeros(to_pad, feat_dim, dtype=torch.float32)
            ], dim=0)

        batch_input_ids.append(padded_input)

        # attention_mask => 1 => real tokens, 0 => padding
        attn_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(to_pad, dtype=torch.long)
        ])
        batch_attention_mask.append(attn_mask)

    # ---- Pad labels ----
    has_labels = any('labels' in s for s in samples)
    if has_labels:
        for s in samples:
            lab = s.get('labels', None)
            if lab is None:
                # no labels => fill zeros
                batch_labels.append(torch.zeros(max_label_len, dtype=torch.float32))
                batch_label_mask.append(torch.zeros(max_label_len, dtype=torch.long))
            else:
                lab_len = lab.shape[0]
                to_pad = max_label_len - lab_len

                if lab.dim() == 1:
                    # shape => [lab_len]
                    padded_label = torch.cat([
                        lab,
                        torch.zeros(to_pad, dtype=torch.float32)
                    ])
                else:
                    # shape => [lab_len, feat_dim]
                    feat_dim = lab.shape[1]
                    padded_label = torch.cat([
                        lab,
                        torch.zeros(to_pad, feat_dim, dtype=torch.float32)
                    ], dim=0)

                batch_labels.append(padded_label)

                lab_mask = torch.cat([
                    torch.ones(lab_len, dtype=torch.long),
                    torch.zeros(to_pad, dtype=torch.long)
                ])
                batch_label_mask.append(lab_mask)

    # ---- Pad decoder_input_ids if present (for multi-step teacher forcing) ----
    if has_decoder_inputs:
        for s in samples:
            dec_in = s.get('decoder_input_ids', None)
            if dec_in is None:
                # no multi-step => fill zeros if we have a max_decoder_len > 0
                if max_decoder_len > 0:
                    # univariate => shape [max_decoder_len]
                    # or multivariate => shape [max_decoder_len, feat_dim]
                    # we'll assume univariate for example. If multi, adapt similarly.
                    batch_decoder_inputs.append(torch.zeros(max_decoder_len, dtype=torch.float32))
                else:
                    # no decoder steps at all
                    batch_decoder_inputs.append(torch.tensor([], dtype=torch.float32))
            else:
                dec_len = dec_in.shape[0]
                to_pad = max_decoder_len - dec_len

                if dec_in.dim() == 1:
                    # shape => [dec_len]
                    padded_dec_in = torch.cat([
                        dec_in,
                        torch.zeros(to_pad, dtype=torch.float32)
                    ])
                else:
                    # shape => [dec_len, feat_dim]
                    feat_dim = dec_in.shape[1]
                    padded_dec_in = torch.cat([
                        dec_in,
                        torch.zeros(to_pad, feat_dim, dtype=torch.float32)
                    ], dim=0)

                batch_decoder_inputs.append(padded_dec_in)

    # ---- Handle dynamic features if present ----
    has_dyn_in = any(s.get('dynamic_features_input') is not None for s in samples)
    has_dyn_out = any(s.get('dynamic_features_label') is not None for s in samples)
    batch_dyn_in, batch_dyn_out = [], []
    batch_dyn_mask_in, batch_dyn_mask_out = [], []

    if has_dyn_in:
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

    # ---- Stack final batch dict ----
    batch_dict = {
        'input_ids': torch.stack(batch_input_ids, dim=0),           # [B, max_input_len, ...]
        'attention_mask': torch.stack(batch_attention_mask, dim=0), # [B, max_input_len]
    }

    if has_labels:
        batch_dict['labels'] = torch.stack(batch_labels, dim=0)          # [B, max_label_len, ...]
        batch_dict['labels_mask'] = torch.stack(batch_label_mask, dim=0) # [B, max_label_len]

    if has_decoder_inputs:
        batch_dict['decoder_input_ids'] = torch.stack(batch_decoder_inputs, dim=0)
        # shape => [B, max_decoder_len, ...features?]

    if has_dyn_in:
        batch_dict['dynamic_features_input'] = torch.stack(batch_dyn_in, dim=0)
        batch_dict['dynamic_features_input_mask'] = torch.stack(batch_dyn_mask_in, dim=0)

    if has_dyn_out:
        batch_dict['dynamic_features_label'] = torch.stack(batch_dyn_out, dim=0)
        batch_dict['dynamic_features_label_mask'] = torch.stack(batch_dyn_mask_out, dim=0)

    if has_static:
        batch_static_cat = torch.nn.utils.rnn.pad_sequence(batch_static_cat, batch_first=True)
        batch_dict['static_cat_features'] = batch_static_cat

    return batch_dict
