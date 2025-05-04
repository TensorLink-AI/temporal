# train_example.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd # Needed for date range generation

# --- Configuration Import ---
from temporal.configs.transformer_config import (
    TransformerTimeSeriesConfig,
    TransformerArchitectureConfig,
    EmbeddingConfig,
    TransformerBlockConfig,
    AttentionConfig,
    FeedForwardConfig,
    OutputHeadConfig,
)
from temporal.models.builder import build_time_series_transformer
from temporal.utils.hf_accessors import save_hf
from temporal.data.transformations import TimeSeriesIterableDataset, timeseries_collate_fn

# --- 1. New Configuration Definition ---
print("Defining model configuration...")
HIDDEN_SIZE = 32
NUM_HEADS = 2
MAX_SEQ_LEN = 4096
CONTEXT_LENGTH = 16
PREDICTION_LENGTH = 4
QUANTILES = [0.1, 0.5, 0.9]
FEATURE_SIZE = 1

encoder_blocks = [
    TransformerBlockConfig(
        block_type="default_encoder",
        attention_config=AttentionConfig(attention_type="full", num_heads=NUM_HEADS, dropout=0.1),
        ffn_config=FeedForwardConfig(type="standard", intermediate_size=64, activation="gelu", dropout=0.1),
    ) for _ in range(2)
]
decoder_blocks = [
    TransformerBlockConfig(
        block_type="default_decoder",
        attention_config=AttentionConfig(attention_type="full", num_heads=NUM_HEADS, dropout=0.1),
        ffn_config=FeedForwardConfig(type="standard", intermediate_size=64, activation="gelu", dropout=0.1),
    ) for _ in range(2)
]
stacked_pos_embed_config = EmbeddingConfig(
    type="stacked_embedding",
    kwargs={
        "embedding_configs": [
            {"type": "sinusoidal", "args": {"max_seq_len": MAX_SEQ_LEN}},
            {"type": "timedelta", "args": {"hidden_dim": 16}},
            {"type": "learned_abs", "args": {"max_seq_len": MAX_SEQ_LEN}}
        ], "max_seq_len": MAX_SEQ_LEN
    }
)
config = TransformerTimeSeriesConfig(
    model_type='transformer', input_dim=FEATURE_SIZE, output_dim=FEATURE_SIZE,
    context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH,
    num_quantiles=len(QUANTILES), loss_config={"type": "quantile_loss"},
    d_model=HIDDEN_SIZE, quantiles=QUANTILES,
    architecture=TransformerArchitectureConfig(
        layout="encoder-decoder", num_encoder_layers=len(encoder_blocks),
        num_decoder_layers=len(decoder_blocks), hidden_dropout_prob=0.1,
    ),
    output_head_config=OutputHeadConfig(
        type="quantile_regression", output_dims=[FEATURE_SIZE] * len(QUANTILES)
    ),
    encoder_blocks=encoder_blocks, decoder_blocks=decoder_blocks,
    positional_embedding_config=stacked_pos_embed_config,
    value_embedding_config=EmbeddingConfig(
         type="linear", embedding_dim=HIDDEN_SIZE, input_dim=FEATURE_SIZE
    ),
    decoder_start_token_value=3.0, use_teacher_forcing=True,
)
print("Configuration defined.")

# --- 2. Model ---
print("Building model...")
model = build_time_series_transformer(config)
print(f"Model built: {type(model).__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- 3. Synthetic Data Generation ---
print("Generating synthetic data for TimeSeriesIterableDataset...")
def generate_synthetic_data_list(num_series=100, min_len=60, max_len=120, noise_level=0.1):
    data_list = []
    base_date = datetime(2023, 1, 1)
    for i in range(num_series):
        # Use variable length again now that mask handling should work
        seq_length = np.random.randint(min_len, max_len + 1)
        t = np.linspace(0, 4 * np.pi, seq_length)
        amp, freq, phase = np.random.rand()*2+0.5, np.random.rand()*0.5+0.5, np.random.rand()*2*np.pi
        series = amp * np.sin(freq * t + phase) + np.random.randn(seq_length) * noise_level
        target_tensor = torch.tensor(series, dtype=torch.float32)
        start_date, freq = base_date + timedelta(days=i), "1D"
        data_list.append({"target": target_tensor, "start_date": start_date, "freq": freq, "item_id": f"series_{i}"})
    return data_list

raw_data_list = generate_synthetic_data_list(
    num_series=500,
    min_len=config.context_length+config.prediction_length+5,
    max_len=config.context_length+config.prediction_length+50
)
print(f"Generated {len(raw_data_list)} time series items.")

# --- 4. Dataset and DataLoader ---
print("Creating DataLoader with TimeSeriesIterableDataset...")
train_size = int(0.8 * len(raw_data_list))
train_data_list = raw_data_list[:train_size]
train_dataset = TimeSeriesIterableDataset(dataset=train_data_list, config=config, revin=False, stride=1)
train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=timeseries_collate_fn)
print(f"DataLoader created.")

# --- 5. Training Loop (Reverted to passing attention_mask) ---
print("Starting training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5
batches_per_epoch = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    for i, batch in enumerate(train_dataloader):
        if not batch: continue
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # --- Model Forward Pass (Passing 2D Masks) ---
        try:
             outputs = model(
                 encoder_inputs=batch['input_ids'],
                 # *** Pass the 2D attention mask from the batch ***
                 attention_mask=batch.get('attention_mask'),
                 decoder_inputs=batch.get('decoder_input_ids'),
                 # Pass the 2D mask for decoder inputs if available (collate_fn might use 'labels_mask')
                 decoder_attention_mask=batch.get('labels_mask'), # Assuming collate_fn uses 'labels_mask' for decoder padding
                 targets=batch.get('labels'),
                 # Other flags if needed for debugging/generation
                 # output_attentions=False,
                 # output_hidden_states=False,
                 # use_cache=False,
             )
             loss = outputs.loss
        except KeyError as e:
            print(f"KeyError during model forward pass: {e}. Batch keys: {list(batch.keys())}")
            print("Ensure model's forward signature arguments map to batch keys from collate_fn ('input_ids', 'attention_mask', 'decoder_input_ids', 'labels_mask', 'labels').")
            raise e
        except TypeError as e:
             print(f"TypeError during model forward pass: {e}. Review model forward signature and passed arguments.")
             print(f"Provided keys: {list(batch.keys())}")
             raise e
        except ValueError as e:
             print(f"ValueError during model forward pass: {e}. Often related to mask processing or shape mismatches.")
             raise e
        except Exception as e:
             print(f"Error during model forward pass: {e}")
             raise e
        # --- End Model Forward Pass ---

        # Check for non-finite loss
        if loss is None:
            print("Warning: Loss is None. Check targets and loss function configuration.")
            continue # Skip backpropagation if loss is None
        if not torch.isfinite(loss):
             print(f"Warning: Non-finite loss detected: {loss.item()}. Skipping backward pass.")
             continue # Skip backpropagation if loss is NaN or Inf

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}/{batches_per_epoch}], Loss: {loss.item():.4f}")
        if batch_count >= batches_per_epoch: break

    if batch_count == 0:
        print(f"Epoch {epoch+1} finished, but no batches were processed.")
        continue
    avg_epoch_loss = epoch_loss / batch_count
    print(f"--- Epoch {epoch+1} Finished --- Avg Loss: {avg_epoch_loss:.4f} ---")

print("Training finished.")

# --- 6. Save Model ---
print("Saving model...")
local_save_dir = "./trained_transformer_new_config_robust"
hub_repo_id = "your_username/temporal_advanced_config_test_robust" # <<< CHANGE THIS
save_to_hub = False # Set to True to upload

try:
    save_hf(
        model=model, config=config, save_directory=local_save_dir, safe=True,
        push_to_hub=save_to_hub, repo_id=hub_repo_id, private=False,
        commit_message="Upload trained model with robust mask handling"
    )
    print(f"Model saved locally to {local_save_dir}")
    if save_to_hub: print(f"Model potentially pushed to Hugging Face Hub: {hub_repo_id}")
except Exception as e: print(f"Error saving model: {e}")

