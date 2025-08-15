# temporal: The Modern Toolkit for Time Series Forecasting

**Build, train, and deploy state-of-the-art deep learning models for time series forecasting with unparalleled flexibility and ease.**

`temporal` is a powerful and extensible Python framework designed to make cutting-edge time series forecasting accessible. It launches with a powerful, state-of-the-art Transformer toolkit, with plans to incorporate other model architectures in the near future. Whether you're a researcher experimenting with novel architectures or a practitioner building robust forecasting solutions, `temporal` provides the tools you need to get the job done.

## 🚀 Getting Started

It's easy to get started with `temporal`.

### Installation

```bash
pip install temporal
```
Your First Forecast in 60 Seconds
```Python

import torch
from temporal.models import build_time_series_transformer
from temporal.configs import TransformerTimeSeriesConfig

# 1. Define your model with a simple configuration
config = TransformerTimeSeriesConfig(
    feature_size=1,
    context_length=128,
    prediction_length=24,
    d_model=64,
    encoder_blocks=[{"type": "default_encoder"}],
    output_head_config={"type": "linear", "output_size": 1},
)

# 2. Build your model
model = build_time_series_transformer(config)

# 3. Make a forecast!
# (B, T, F) -> (1, 128, 1)
context = torch.randn(1, 128, 1)
forecast = model.generate(context, prediction_length=24)

print(forecast.shape)  # torch.Size([1, 24, 1])
```
✨ Key Features
Configuration-Driven: Design complex models with simple, readable configurations. No more boilerplate code.

Modular and Extensible: Swap out components like attention mechanisms, normalization layers, and output heads with ease. Add your own custom components with a single decorator.

State-of-the-Art Components: temporal comes with a rich set of pre-built components, including:

Advanced Attention Mechanisms: FlashAttention, LSEAttention, DifferentialAttention, and more.

Probabilistic Forecasting: A variety of output heads for modeling uncertainty, including GaussianHead, QuantileRegressionOutputHead, and MixtureOutputHead.

Patch-Based Modeling: First-class support for patch-based time series modeling for improved efficiency and performance.

Hugging Face Compatible: Seamlessly integrate your models with the Hugging Face ecosystem for training, sharing, and deployment.

📚 Learn More
Tutorial: A detailed guide to building and training your first model.

Core Concepts: Understand the "magic" behind temporal.

Transformer Capabilities: A deep dive into the advanced features of the Transformer module.

Extending temporal: Learn how to add your own custom components.