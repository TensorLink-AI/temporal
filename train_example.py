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
    OutputHeadConfig, # Keep this import
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
FEATURE_SIZE = 1 # Univariate

# Define Quantiles (Ensemble Size K for DistPred)
NUM_QUANTILES = 100 # K = 100
QUANTILES = np.linspace(0.5 / NUM_QUANTILES, 1 - 0.5 / NUM_QUANTILES, NUM_QUANTILES).tolist()
print(f"Using {len(QUANTILES)} quantiles/ensemble members for DistPred.")

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

# Assemble the top-level config
config = TransformerTimeSeriesConfig(
    model_type='transformer',
    input_dim=FEATURE_SIZE, # input_dim usually same as feature_size for value embedding
    output_dim=FEATURE_SIZE, # Base output dimension before head projection
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_quantiles=len(QUANTILES), # K = 100
    quantiles=QUANTILES, # Pass the list of 100 quantiles (might be optional depending on head/loss)

    loss_config={"type": "crps"}, # Use CRPS loss

    d_model=HIDDEN_SIZE, # Renamed from hidden_size for consistency with HF

    architecture=TransformerArchitectureConfig(
        layout="encoder-decoder",
        num_encoder_layers=len(encoder_blocks),
        num_decoder_layers=len(decoder_blocks),
        hidden_dropout_prob=0.1,
    ),

    # --- Use the new DistPredHead --- Specify type and required args
    output_head_config=OutputHeadConfig(
        type="distpred", # Use the registered DistPredHead
        # Provide args needed by DistPredHead.__init__:
        num_outputs=len(QUANTILES), # K = 100
        feature_size=FEATURE_SIZE   # 1 for univariate
    ),
    # --- End Output Head Config ---

    encoder_blocks=encoder_blocks,
    decoder_blocks=decoder_blocks,

    # Positional embedding config (ensure max_seq_len is sufficient)
    positional_embedding_config=stacked_pos_embed_config,

    # Value embedding config
    value_embedding_config=EmbeddingConfig(
         type="linear", embedding_dim=HIDDEN_SIZE, input_dim=FEATURE_SIZE
    ),

    # Other transformer settings
    decoder_start_token_value=0.0, # Often 0 or mean/median of data
    use_teacher_forcing=True, # Common during training
)

print(f"Configuration defined with loss type: {config.loss_config['type']} and output head: {config.output_head_config.type}")

# --- 2. Model ---
print("Building model...")
# The build function should now use CRPSLoss based on the config and DistPredHead
model = build_time_series_transformer(config)
print(f"Model built: {type(model).__name__}")
print(f"Output Head: {type(model.output_head).__name__}")
print(f"Loss Function: {type(model.loss_fn).__name__}") # Check if loss_fn attribute exists and is CRPSLoss
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# --- 3. Synthetic Data Generation ---
print("Generating synthetic data for TimeSeriesIterableDataset...")
def generate_synthetic_data_list(num_series=100, min_len=60, max_len=120, noise_level=0.1):
    data_list = []
    base_date = datetime(2023, 1, 1)
    for i in range(num_series):
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

# --- 5. Training Loop ---
print("Starting training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5 # Reduced for quick testing
batches_per_epoch = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    for i, batch in enumerate(train_dataloader):
        if not batch: continue # Skip empty batches
        optimizer.zero_grad()

        # Move relevant tensors to device
        batch_on_device = {}
        relevant_keys = ['input_ids', 'attention_mask', 'decoder_input_ids', 'labels_mask', 'labels', 'loss_mask']
        for k, v in batch.items():
            if k in relevant_keys and isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(device)
            # Note: Time features might also be needed if using time embeddings

        # Ensure required inputs are present
        if 'input_ids' not in batch_on_device or 'labels' not in batch_on_device:
             print(f"Skipping batch {i}, missing required keys.")
             continue

        # --- Model Forward Pass ---
        try:
             outputs = model(
                 encoder_inputs=batch_on_device['input_ids'],
                 attention_mask=batch_on_device.get('attention_mask'), # Encoder mask
                 decoder_inputs=batch_on_device.get('decoder_input_ids'), # May be optional depending on model
                 decoder_attention_mask=batch_on_device.get('labels_mask'), # Decoder mask (for padding)
                 targets=batch_on_device.get('labels'),
                 loss_mask=batch_on_device.get('loss_mask') # Pass the loss mask if available
             )
             loss = outputs.loss
        except Exception as e:
             print(f"Error during model forward pass in epoch {epoch+1}, batch {i}: {e}")
             # Optionally print shapes for debugging
             print("--- Batch Shapes ---")
             for k, v in batch_on_device.items():
                 if isinstance(v, torch.Tensor):
                     print(f"  {k}: {v.shape}")
             print("--------------------");
             raise e # Re-raise after printing info
        # --- End Model Forward Pass ---

        # Check for non-finite loss
        if loss is None:
            print(f"Warning: Loss is None in epoch {epoch+1}, batch {i}. Check model output and loss calculation.")
            # Example: Check if outputs.logits exists
            # print("Model output keys:", outputs.keys() if hasattr(outputs, 'keys') else "N/A")
            continue
        if not torch.isfinite(loss):
             print(f"Warning: Non-finite loss detected in epoch {epoch+1}, batch {i}: {loss.item()}. Skipping backward pass.")
             continue

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1

        if batch_count % 10 == 0: # Print loss every 10 batches
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
local_save_dir = "./trained_transformer_distpred_crps"
hub_repo_id = "your_username/temporal_distpred_crps_k100" # <<< CHANGE THIS
save_to_hub = False # Set to True to upload

try:
    save_hf(
        model=model, config=config, save_directory=local_save_dir, safe=True,
        push_to_hub=save_to_hub, repo_id=hub_repo_id, private=False,
        commit_message=f"Upload trained model with DistPredHead and CRPS loss (K={len(QUANTILES)})"
    )
    print(f"Model saved locally to {local_save_dir}")
    if save_to_hub: print(f"Model potentially pushed to Hugging Face Hub: {hub_repo_id}")
except Exception as e: print(f"Error saving model: {e}")

