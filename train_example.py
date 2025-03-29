import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import  DataLoader
from temporal.models.vanilla_transformer_ar import VanillaTransformerAR
from temporal.configs.transformertimeseriesconfig import TransformerTimeSeriesConfig
from temporal.data.dataloaders import TimeSeriesIterableDataset

# Model configuration
seq_len = 20  # Example sequence length
input_dim = 5  # Example input dimension
output_dim = 1 # example output_dim
batch_size = 16
config = TransformerTimeSeriesConfig(
    input_size=input_dim,
    output_size=output_dim,
    d_model=32,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=64,
    dropout=0.1,
    activation='relu',
    max_len=seq_len
)

# Create a dummy dataset using a generator
def dummy_data_generator(num_samples, seq_len, input_dim, output_dim):
    for _ in range(num_samples):
        yield {
            'features': torch.randn(seq_len, input_dim),
            'target': torch.randn(seq_len, output_dim)
        }


num_samples = 100

# Data Loading using the generator
target_key = 'target'

full_dataset = TimeSeriesIterableDataset(data_generator=dummy_data_generator(num_samples, seq_len, input_dim, output_dim), target_key=target_key)
train_size = int(num_samples * 0.7)
val_size = int(num_samples * 0.15)
test_size = num_samples - train_size - val_size
train_dataloader, val_dataloader, test_dataloader = full_dataset.split([train_size, val_size, test_size], batch_size=batch_size)



# Model, loss, and optimizer
model = VanillaTransformerAR(config)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# Training, Validation, and Test loops
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()        
        loss.backward()   
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Train: Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

def val_loop(dataloader, model, loss_fn):
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        if batch_idx % 10 == 0:
            print(f"Val: Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        if batch_idx % 10 == 0:
            print(f"Test: Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")


# training
num_epochs = 5
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, criterion, optimizer)
    val_loop(val_dataloader, model, criterion)
    test_loop(test_dataloader, model, criterion)
print("Done!")