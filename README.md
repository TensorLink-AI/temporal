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
## ✨ Key Features
* **Configuration-Driven:** Design complex models with simple, readable configurations. No more boilerplate code.
* **Modular and Extensible:** Swap out components like attention mechanisms, normalization layers, and output heads with ease. Add your own custom components with a single decorator.
* **State-of-the-Art Components:** `temporal` comes with a rich set of pre-built components, including:
    * **Advanced Attention Mechanisms:** `FlashAttention`, `LSEAttention`, `DifferentialAttention`, and more.
    * **Probabilistic Forecasting:** A variety of output heads for modeling uncertainty, including `GaussianHead`, `QuantileRegressionOutputHead`, and `MixtureOutputHead`.
    * **Patch-Based Modeling:** First-class support for patch-based time series modeling for improved efficiency and performance.
* **Hugging Face Compatible:** Seamlessly integrate your models with the Hugging Face ecosystem for training, sharing, and deployment.

## 📚 Learn More
* **[Tutorial](./docs/getting-started.md):** A detailed guide to building and training your first model.
* **[Core Concepts](./docs/core-concepts.md):** Understand the "magic" behind `temporal`.
* **[Transformer Capabilities](./docs/transformer-capabilities.md):** A deep dive into the advanced features of the Transformer module.
* **[Extending `temporal`](./docs/extending-temporal.md):** Learn how to add your own custom components.


# `temporal`: A Modular and Configurable Library for Time Series Transformers

`temporal` is a powerful and flexible library for building and experimenting with transformer-based models for time series forecasting. It is designed for researchers and practitioners who need to go beyond off-the-shelf models and build custom solutions for their specific needs.

## Why `temporal`?

In a world of many time series libraries, `temporal` stands out by offering:

* **Unparalleled Flexibility:** `temporal`'s modular architecture allows you to mix and match components to create novel transformer architectures with ease.
* **Deep Customization:** Go beyond simple hyperparameter tuning and control every aspect of your model, from the attention mechanism to the normalization layers.
* **Research-Ready:** `temporal` is built for experimentation, with a focus on making it easy to implement and test new ideas.
* **Hugging Face Integration:** Seamlessly share and use your `temporal` models within the Hugging Face ecosystem.

## `temporal` vs. The World: A Comparative Look

| Library | Core Philosophy | Key Features & Strengths | Probabilistic Support | Primary Target Audience |
| :--- | :--- | :--- | :--- | :--- |
| **`temporal`** | High modularity and configurability for research and experimentation with novel transformer architectures. | Registry for dynamic component registration, configuration-driven model building, rich library of modules, Hugging Face integration. | Extensive, with various loss functions and output heads for distributional prediction. | Researchers and practitioners building custom transformer models for time series. |
| **Neural Forecast** | Scalable and user-friendly neural forecasting algorithms with a focus on performance and usability. | Large collection of state-of-the-art models, familiar `sklearn` syntax, support for exogenous variables, automatic hyperparameter tuning. | Yes, through quantile losses and parametric distributions. | Data scientists and ML engineers looking for a user-friendly and scalable library. |
| **Darts** | User-friendly forecasting and anomaly detection, aiming to be the "scikit-learn for time series." | Unified `fit()`/`predict()` API, wide range of models (classical to deep learning), backtesting, anomaly detection. | Yes, supports estimating parametric distributions or quantiles. | Data scientists and practitioners who want a simple and unified interface for a variety of models. |
| **GluonTS** | Probabilistic time series modeling with a focus on deep learning-based models. | Built on PyTorch and MXNet, strong emphasis on probabilistic forecasting, includes models like DeepAR. | Core focus of the library. | Researchers and practitioners who require robust probabilistic forecasts. |
| **Merlion** | An end-to-end machine learning framework for time series intelligence (forecasting, anomaly detection, change point detection). | Unified interface for various models, AutoML, post-processing rules for anomaly detection, GUI dashboard. | Yes, provides forecasts with confidence intervals. | Engineers and researchers looking for a one-stop solution for various time series tasks, with a focus on production deployment. |
| **PyTorch Forecasting** | Ease state-of-the-art time series forecasting with neural networks for both real-world cases and research. | Built on PyTorch Lightning, includes models like TFT and N-BEATS, built-in interpretation capabilities. | Yes, with models like DeepAR and support for quantile losses. | Professionals and beginners who want to use state-of-the-art models with a high-level API. |

## `temporal` vs. Hugging Face `transformers`: Why a Specialized Library?

While the Hugging Face `transformers` library is an incredible tool for NLP, time series data has unique characteristics that demand a specialized approach. `temporal` is designed from the ground up for time series, offering:

* **Time-Series-Native Components:** `temporal` provides a rich set of components specifically designed for time series data, including:
    * **Normalization Layers:** Reversible Instance Normalization (RevIN) for handling distribution shifts.
    * **Embeddings:** A variety of positional and value embeddings for temporal features.
    * **Attention Mechanisms:** Attention mechanisms designed to capture temporal dependencies.
* **Architectural Flexibility for Time Series:** `temporal`'s modular and configuration-driven design is optimized for experimenting with different transformer architectures for time series forecasting.
* **A Strong Focus on Probabilistic Forecasting:** `temporal` has a strong focus on probabilistic forecasting, a critical requirement for many real-world time series applications.

## Key Features

* **Registry System:** A powerful registry that allows for the dynamic registration and resolution of various modules.
* **Configuration Dataclasses:** A structured and type-safe way to define model architectures.
* **Model Builder:** Automates the model creation process and ensures that the resulting model is consistent with the specified configuration.
* **Rich Module Library:** A comprehensive library of pre-built modules, including a wide variety of attention mechanisms, embeddings, normalization layers, and output heads.
* **Hugging Face Integration:** An adapter that allows `temporal` models to be easily shared and used within the `transformers` framework.
