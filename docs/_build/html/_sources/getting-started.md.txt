# Getting Started with `temporal`

Welcome to the `temporal` tutorial! In this guide, you'll learn how to build and train your first time series forecasting model with `temporal`. We'll cover:

1.  **Defining a Configuration:** How to specify your model's architecture using a simple and readable configuration object.
2.  **Building the Model:** How to use the `build_time_series_transformer` function to instantiate your model.
3.  **Making a Forecast:** How to use the `.generate()` method to make predictions.

## 1. Define a Configuration

In `temporal`, you define your model's architecture using a `TransformerTimeSeriesConfig` object. This dataclass allows you to specify everything from the model's dimensions to the types of layers you want to use.

Let's start with a simple example: a single-layer encoder-only Transformer with a linear output head for point forecasting.

```python
from temporal.configs import TransformerTimeSeriesConfig

config = TransformerTimeSeriesConfig(
    feature_size=1,  # The number of features in your time series
    context_length=128,  # The length of the input sequence
    prediction_length=24,  # The number of steps to forecast
    d_model=64,  # The hidden dimension of the model
    encoder_blocks=[{"type": "default_encoder"}],  # A single encoder layer
    output_head_config={"type": "linear", "output_size": 1},  # A linear output head
)
```
That's it! You've just defined a complete Transformer model.

2. Build the Model
Now that you have your configuration, you can use the build_time_series_transformer function to instantiate your model.

```python

from temporal.models import build_time_series_transformer

model = build_time_series_transformer(config)
``` 

This function takes your configuration and uses the temporal builder and registry to assemble all the necessary components into a complete PyTorch model.

3. Make a Forecast
The resulting model is a standard PyTorch module with .forward() and .generate() methods. You can use the .forward() method for training and the .generate() method for making predictions.

Let's create some dummy data and make a forecast.

```python

import torch

# Create a dummy input tensor of shape (batch_size, context_length, feature_size)
context = torch.randn(1, 128, 1)

# Generate a forecast for the next 24 time steps
forecast = model.generate(context, prediction_length=24)

print(forecast.shape)  # torch.Size([1, 24, 1])
```
Congratulations! You've just built and used your first temporal model.

What's Next?
Now that you've seen the basics, you're ready to explore the more advanced features of temporal. Check out the Core Concepts guide to learn about the underlying design of the framework, or dive into the Transformer Capabilities guide to see all the powerful components that temporal has to offer.
