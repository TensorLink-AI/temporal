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
    NormalizationConfig,
)
from temporal.models.builder import build_time_series_transformer
from temporal.utils.hf_accessors import save_hf
from temporal.data.transformations import TimeSeriesIterableDataset, timeseries_collate_fn

# --- 1. New Configuration Definition ---
print("Defining model configuration...")
HIDDEN_SIZE = 32 # Corresponds to d_model
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
        normalization_config=NormalizationConfig(type="layer"),
    ) for _ in range(2)
]
decoder_blocks = [
    TransformerBlockConfig(
        block_type="default_decoder",
        attention_config=AttentionConfig(attention_type="full", num_heads=NUM_HEADS, dropout=0.1),
        ffn_config=FeedForwardConfig(type="standard", intermediate_size=64, activation="gelu", dropout=0.1),
        normalization_config=NormalizationConfig(type="layer"),
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

# Calculate the required output_size for the head
# output_size = num_quantiles * feature_size
head_output_size = len(QUANTILES) * FEATURE_SIZE

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
    feature_size=FEATURE_SIZE, # Make sure feature_size is available in main config
    hidden_dropout_prob=0.1, # Moved dropout prob here

    architecture=TransformerArchitectureConfig(
        layout="encoder-decoder",
        num_encoder_layers=len(encoder_blocks),
        num_decoder_layers=len(decoder_blocks),
        # Removed hidden_dropout_prob from here
    ),

    # --- UPDATED DistPredHead Config ---
    output_head_config=OutputHeadConfig(
        type="distpred", # Use the registered DistPredHead
        output_size=head_output_size, # Explicitly set total output size
        # Pass required args for DistPredHead inside nested kwargs
        kwargs={
            "num_outputs": len(QUANTILES), # K = 100
            "feature_size": FEATURE_SIZE   # 1 for univariate
        }
    ),
    # --- End Output Head Config ---

    encoder_blocks=encoder_blocks,
    decoder_blocks=decoder_blocks,

    # Positional embedding config (ensure max_seq_len is sufficient)
    positional_embedding_config=stacked_pos_embed_config,

    # --- REVERTED & CORRECTED Value embedding config --- Use type "value" and kwargs
    value_embedding_config=EmbeddingConfig(
         type="value", # Use the registered name "value"
         kwargs={ # Arguments for TimeSeriesValueEmbedding go here
             "feature_size": FEATURE_SIZE,
             "d_model": HIDDEN_SIZE
         }
    ),
    # --- End Value embedding config ---

    # Other transformer settings
    decoder_start_token_value=0.0, # Often 0 or mean/median of data
    use_teacher_forcing=True, # Common during training
)

print(f"Configuration defined with loss type: {config.loss_config['type']} and output head: {config.output_head_config.type}")

# --- 2. Model ---
print("Building model...")
# The build function should now correctly build DistPredHead and use CRPSLoss
model = build_time_series_transformer(config)
print(f"Model built: {type(model).__name__}")
# --- CORRECTED attribute access --- #
print(f"Output Head: {type(model.output_heads).__name__}") # Use plural 'output_heads'
# Check the actual loss function attached to the model (might need access to model internals)
if hasattr(model, 'loss_fn'):
    print(f"Loss Function: {type(model.loss_fn).__name__}")
elif hasattr(model, 'loss'): # Some HF models store loss internally
     print(f"Loss Function (internal): {type(model.loss).__name__}")
else:
     print("Loss function attribute not found (check model implementation).")
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

# --- 5. Training Loop (Reverted to simpler, working version) ---
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
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # --- Model Forward Pass --- 
        try:
             outputs = model(
                 encoder_inputs=batch['input_ids'],
                 # Pass the 2D attention mask from the batch
                 attention_mask=batch.get('attention_mask'),
                 decoder_inputs=batch.get('decoder_input_ids'),
                 # Pass the 2D mask for decoder inputs if available
                 decoder_attention_mask=batch.get('labels_mask'),
                 targets=batch.get('labels'),
                 # loss_mask=batch.get('loss_mask') # Add this back if your model/loss handles it
             )
             loss = outputs.loss
        except KeyError as e:
            print(f"KeyError during model forward pass: {e}. Batch keys: {list(batch.keys())}")
            print("Ensure model's forward signature arguments (e.g., 'encoder_inputs') correctly map to batch keys (e.g., 'input_ids').")
            raise e
        except Exception as e:
             print(f"Error during model forward pass: {e}")
             raise e
        # --- End Model Forward Pass ---

        # Check for non-finite loss (moved after forward pass)
        if loss is None:
            print(f"Warning: Loss is None in epoch {epoch+1}, batch {i}. Check model output and loss calculation.")
            continue 
        if not torch.isfinite(loss):
             print(f"Warning: Non-finite loss detected in epoch {epoch+1}, batch {i}: {loss.item()}. Skipping backward pass.")
             continue

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1

        # Print loss less frequently
        if batch_count % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}/{batches_per_epoch}], Loss: {loss.item():.4f}")
        if batch_count >= batches_per_epoch: break

    if batch_count == 0:
        print(f"Epoch {epoch+1} finished, but no batches were processed.")
        continue
    # Avoid division by zero if batch_count is 0
    avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0 
    print(f"--- Epoch {epoch+1} Finished --- Avg Loss: {avg_epoch_loss:.4f} ---")

print("Training finished.")
# --- End Training Loop ---


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
