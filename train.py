
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import os
import random
from datetime import datetime, timedelta
from collections import deque
from itertools import islice
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

# Local project imports - assuming this script is run from the project root
from temporal.configs.transformer_config import TransformerTimeSeriesConfig, TransformerBlockConfig, AttentionConfig, FeedForwardConfig, EmbeddingConfig, OutputHeadConfig, TransformerArchitectureConfig
from temporal.models.builder import build_time_series_transformer

# --- Synthetic Data Generation ---
# Note: This section is adapted to make the script self-contained.
# In a real scenario, you would load your own data.

def generate_synthetic_data_list(num_series=500, min_len=150, max_len=300, noise_level=0.1):
    """Generates a list of synthetic time series data points."""
    data_list = []
    base_date = datetime(2023, 1, 1)
    for i in range(num_series):
    does     seq_length = np.random.randint(min_len, max_len + 1)
        t = np.linspace(0, 4 * np.pi, seq_length)
        amp = np.random.rand() * 2 + 0.5
        freq = np.random.rand() * 0.5 + 0.5
        phase = np.random.rand() * 2 * np.pi
        series = amp * np.sin(freq * t + phase) + np.random.randn(seq_length) * noise_level
        target_tensor = torch.tensor(series, dtype=torch.float32)
        data_list.append({"target": target_tensor, "item_id": f"series_{i}"})
    return data_list

class TimeSeriesIterableDataset(IterableDataset):
    """An iterable dataset for time series data."""
    def __init__(self, data_list, config, stride=1):
        self.data_list = data_list
        self.config = config
        self.total_length = config.context_length + config.prediction_length
        self.stride = stride

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            iter_start = 0
            iter_end = len(self.data_list)
        else:
            # Multi-process data loading
            per_worker = int(np.ceil(len(self.data_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data_list))

        for i in range(iter_start, iter_end):
            series_data = self.data_list[i]["target"]
            for start_idx in range(0, len(series_data) - self.total_length + 1, self.stride):
                end_idx = start_idx + self.total_length
                chunk = series_data[start_idx:end_idx]

                input_chunk = chunk[:self.config.context_length]
                label_chunk = chunk[self.config.context_length:self.total_length]

                yield {
                    "input_ids": input_chunk,
                    "labels": label_chunk,
                }

def timeseries_collate_fn(batch):
    """
    Custom collate function for an encoder-decoder architecture.
    """
    if not batch:
        return None

    # Encoder gets the historical context
    encoder_inputs = torch.stack([item['input_ids'] for item in batch]).unsqueeze(-1) # Shape: [B, T_ctx, 1]
    
    # Labels are the future values we want to predict
    labels = torch.stack([item['labels'] for item in batch]) # Shape: [B, T_pred]

    # Decoder input is the "shifted" version of the labels.
    # A start-of-sequence token (zeros) is prepended, and the last label is dropped.
    decoder_input_start = torch.zeros((labels.shape[0], 1), dtype=labels.dtype)
    decoder_inputs = torch.cat([decoder_input_start, labels[:, :-1]], dim=1).unsqueeze(-1) # Shape: [B, T_pred, 1]

    # Masks
    encoder_attention_mask = torch.ones(encoder_inputs.shape[:-1], dtype=torch.long) # Shape: [B, T_ctx]
    decoder_attention_mask = torch.ones(decoder_inputs.shape[:-1], dtype=torch.long) # Shape: [B, T_pred]
    loss_mask = torch.ones_like(labels, dtype=torch.float) # Shape: [B, T_pred]

    return {
        "encoder_inputs": encoder_inputs,
        "decoder_inputs": decoder_inputs,
        "labels": labels,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "loss_mask": loss_mask
    }

# --- Model Configuration ---
print("Defining model configuration...")
HIDDEN_SIZE = 512
NUM_HEADS = 8
CONTEXT_LENGTH = 128
PREDICTION_LENGTH = 64
FEATURE_SIZE = 1 # Univariate
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

# Switched to a more powerful GLU-based feedforward network
ffn_config = FeedForwardConfig(
    type="glu", 
    intermediate_size=HIDDEN_SIZE * 4, 
    activation="gelu", 
    dropout=0.1
)

encoder_blocks = [
    TransformerBlockConfig(
        block_type="default_encoder",
        attention_config=AttentionConfig(attention_type="full", num_heads=NUM_HEADS, dropout=0.1, use_rope=True),
        ffn_config=ffn_config,
    ) for _ in range(NUM_ENCODER_LAYERS)
]

decoder_blocks = [
    TransformerBlockConfig(
        block_type="default_decoder",
        attention_config=AttentionConfig(attention_type="full", num_heads=NUM_HEADS, dropout=0.1, use_rope=True),
        ffn_config=ffn_config,
    ) for _ in range(NUM_DECODER_LAYERS)
]

config = TransformerTimeSeriesConfig(
    model_type='transformer',
    input_dim=FEATURE_SIZE,
    output_dim=FEATURE_SIZE,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    d_model=HIDDEN_SIZE,
    # Switched to encoder-decoder architecture
    architecture=TransformerArchitectureConfig(
        layout="encoder-decoder", 
        num_encoder_layers=len(encoder_blocks),
        num_decoder_layers=len(decoder_blocks)
    ),
    encoder_blocks=encoder_blocks,
    decoder_blocks=decoder_blocks,
    value_embedding_config=EmbeddingConfig(type="value", kwargs={"feature_size": FEATURE_SIZE, "d_model": HIDDEN_SIZE}),
    positional_embedding_config=EmbeddingConfig(type="sinusoidal"),
    output_head_config=OutputHeadConfig(type="linear", output_size=FEATURE_SIZE),
    loss_config={"type": "mse"},
    use_cache=True
)

print("Building model...")
model = build_time_series_transformer(config)
print(f"Model built: {type(model).__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- Training Setup ---
# Hyperparameters
DEEP_VAL_INTERVAL  = 1001
VAL_SMOOTH_WINDOW  = 20
MAX_BATCHES        = 5000
DEEP_VAL_BATCHES   = 200
ETA_MIN            = 1e-6
CLIP_GRAD_NORM     = 1.0
LEARNING_RATE      = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)

# --- Main Training Loop ---
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler(enabled=(device.type == 'cuda'))
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_BATCHES, eta_min=ETA_MIN)

print("Generating data and creating DataLoader...")
raw_data = generate_synthetic_data_list(num_series=2000, min_len=config.context_length + config.prediction_length + 1)
dataset = TimeSeriesIterableDataset(raw_data, config, stride=16)
loader = DataLoader(dataset, batch_size=32, collate_fn=timeseries_collate_fn, num_workers=2)

data_iter = iter(loader)
val_loss_queue = deque(maxlen=VAL_SMOOTH_WINDOW)
batch_counter = 0
token_counter = 0

print("🔁 Starting training loop with new Encoder-Decoder architecture…")

while batch_counter < MAX_BATCHES:
    # --- TRAINING PHASE ---
    model.train()
    for _ in range(100): # Train for 100 batches before a quick validation
        if batch_counter >= MAX_BATCHES: break
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)

        if not batch: continue
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k] = v.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == 'cuda')):
            out = model(
                encoder_inputs=batch["encoder_inputs"],
                decoder_inputs=batch["decoder_inputs"],
                attention_mask=batch["encoder_attention_mask"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                targets=batch["labels"],
                loss_mask=batch["loss_mask"],
            )
            loss = out.loss

        if loss is None or not torch.isfinite(loss):
            print(f"Warning: Skipping batch {batch_counter} due to invalid loss: {loss}")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        B, T, _ = batch["encoder_inputs"].shape
        token_counter += B * T
        batch_counter += 1

        if batch_counter % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Train {batch_counter}/{MAX_BATCHES}] Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Tokens: {token_counter:,}")

    # --- QUICK VALIDATION ---
    model.eval()
    val_loss_total = 0.0
    val_seen = 0
    with torch.no_grad():
        for _ in range(10): # Quick validation on 10 batches
            try:
                val_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader); val_batch = next(data_iter)
            if not val_batch: continue
            for k, v in val_batch.items():
                if isinstance(v, torch.Tensor): val_batch[k] = v.to(device)

            with autocast(enabled=(device.type == 'cuda')):
                vout = model(
                    encoder_inputs=val_batch["encoder_inputs"],
                    decoder_inputs=val_batch["decoder_inputs"],
                    attention_mask=val_batch["encoder_attention_mask"],
                    decoder_attention_mask=val_batch["decoder_attention_mask"],
                    targets=val_batch["labels"],
                    loss_mask=val_batch["loss_mask"],
                )
                if vout.loss is not None and torch.isfinite(vout.loss):
                    val_loss_total += vout.loss.item()
                    val_seen += 1

    if val_seen > 0:
        avg_val = val_loss_total / val_seen
        val_loss_queue.append(avg_val)
        smooth_val = sum(val_loss_queue) / len(val_loss_queue)
        print(f"🔵 [Val @ {batch_counter}] Raw: {avg_val:.4f} | Smoothed: {smooth_val:.4f}")

print("✅ Training complete.")
