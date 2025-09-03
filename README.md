# Temporal: The Modern Toolkit for Time Series Forecasting

[![Tests](https://github.com/TensorLink-AI/temporal/actions/workflows/tests.yml/badge.svg)](https://github.com/TensorLink-AI/temporal/actions/workflows/tests.yml)
[![Coverage](https://github.com/TensorLink-AI/temporal/actions/workflows/coverage.yml/badge.svg)](https://github.com/TensorLink-AI/temporal/actions/workflows/coverage.yml)
[![Docs](https://github.com/TensorLink-AI/temporal/actions/workflows/docs.yml/badge.svg)](https://tensorlink-ai.github.io/temporal/)


**Build, train, and deploy state-of-the-art deep learning models for time series forecasting with unparalleled flexibility and ease.**

`temporal` is a powerful and extensible Python framework designed to make cutting-edge time series forecasting accessible. It launches with a powerful, state-of-the-art Transformer toolkit, with plans to incorporate other model architectures in the near future. Whether you're a researcher experimenting with novel architectures or a practitioner building robust forecasting solutions, `temporal` provides the tools you need to get the job done.

## 🚀 Getting Started

It's easy to get started with `temporal`.

### Installation

```bash
pip install uv
pip install temporal
```

## 🚀 Quick API (`temporal.quick`)

The `temporal.quick` module is the **fastest way** to build, train, and predict with Temporal models — while still mapping 1:1 to the underlying typed configs and builders.

### Hello, Temporal
```python
from temporal.quick import Temporal

model = (
    Temporal()
      .as_encoder(n_layers=6)
      .with_dims(d_model=512, n_heads=8)   # d_head = 64 (even → RoPE-safe)
      .with_context(192)
      .with_horizon(288)
      .with_attention("full")
      .with_rope()
      .head_quantile([0.05, 0.5, 0.95])
      .with_mc_samples(128)
      .build()
)

# Train
model.fit_arrays(inputs={"input_values": x_train}, targets=y_train, epochs=4, lr=2e-4)

# Predict
q = model.predict_quantiles({"input_values": x_future})  # [Q, T, F]
```

### One-liner helper
```python
from temporal.quick import forecast

q = forecast({"input_values": x}, horizon=96, context=192, targets=y,
             arch="decoder", attention="full", rope=True)
```

### Common recipes

#### Decoder-only with RoPE
```python
Temporal().as_decoder(6).with_dims(512, 8).with_context(192).with_horizon(288).with_rope() \
    .head_quantile([0.05, 0.5, 0.95]).build()
```

#### Enc–Dec with d_head = 64
```python
Temporal().as_encdec(8, 4).with_dims(512, 8).with_context(336).with_horizon(96).with_rope() \
    .head_quantile([0.05, 0.5, 0.95]).build()
```

#### Patterned (local) attention
```python
Temporal().as_encoder(6).with_dims(512, 8).with_context(256).with_horizon(96) \
    .with_attention("patterned").with_attention_pattern(kind="local", window_size=128, stride=64) \
    .head_quantile([0.1, 0.5, 0.9]).build()
```

#### ALiBi & qk-LayerNorm
```python
Temporal().as_decoder(6).with_dims(512, 8).with_context(192).with_horizon(96) \
    .with_alibi().with_qk_layernorm().head_gaussian().build()
```

#### Value patch embedding
```python
Temporal().as_encoder(4).with_dims(512, 8).with_context(256).with_horizon(96) \
    .value_embedding("patch", patch_size=4, feature_size=512, stride=2, use_mlp=True, mlp_hidden_size=1024) \
    .positional_embedding("sinusoidal") \
    .head_quantile([0.05, 0.5, 0.95]).build()
```

#### RevIN (instance normalization)
```python
Temporal().as_encoder(6).with_dims(512, 8).with_context(192).with_horizon(96) \
    .with_instance_norm(type="revin", num_features=512, affine=True, subtract_last=False) \
    .build()
```

### API cheat sheet
| Category      | Method                                                                        | Notes                                                              |
| :------------ | :---------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| Topology      | `.as_encoder(n), .as_decoder(n), .as_encdec(n_enc, n_dec)`                      | Build stack type & depth                                           |
| Dims          | `.with_dims(d_model, n_heads, dropout?, max_pos?)`                             | Validates `d_model % n_heads == 0` and RoPE’s even `d_head`          |
| Windows       | `.with_context(L), .with_horizon(H)`                                          | Look-back and forecast length                                      |
| Attention     | `.with_attention(kind, **cfg)`                                                | `full`, `patterned`, `lse`, `hybrid`, `diffwist`                   |
| RoPE/ALiBi    | `.with_rope(base=10000), .with_alibi()`                                        | Set inside attention config                                        |
| QK Norm/Bias  | `.with_qk_layernorm(), .with_attention_bias(True | False)`                      |                                                                    |
| Patterns      | `.with_attention_pattern(kind, window_size, stride?, dilation?, global_indices?)` | For `patterned`                                                    |
| Embeddings    | `.value_embedding("value" | "patch", **cfg)`                                   |                                                                    |
| Positional    | `.positional_embedding("sinusoidal" | "learned_abs")`                           |                                                                    |
| Norms (model) | `.with_layer_norm(kind="layer" | "rms")`                                       |                                                                    |
| Instance Norm | `.with_instance_norm(type="revin" | "dynamic_revin")`                          |                                                                    |
| Block Norm    | `.with_block_norm(kind="layer" | "rms")`                                       |                                                                    |
| Heads/Loss    | `.head_quantile(qs), .head_gaussian(), .head_mixture(kind, components)`       | CRPS or NLL                                                        |
| Aggregation   | `.head_aggregator(kind="mean" | "gated")`                                     |                                                                    |
| Train         | `.fit_arrays(...), .fit_dataloader(...), .train_config(...)`                   | AMP, grad clip, etc.                                               |
| Inference     | `.predict_samples(...), .predict_quantiles(...), .predict_point(...)`          | Uses MC-dropout                                                    |

### Config mapping (trustworthy)

All methods compile into the same typed configs your core builder expects:

*   `TransformerTimeSeriesConfig`
*   `transformer_block_config_from_dict`
*   `attention_config` (with `use_rope`, `use_alibi`, `qk_layernorm`, `bias`, `pattern`)
*   `embedding_config` (value/patch + positional)
*   `normalization_config` (`layer`/`rms`/`scale` + RevIN variants)
*   `output_head_config`, `loss_config`, `head_aggregation_config`

This guarantees clean parity between `temporal.quick` and the low-level modules.

### Tests

A matching test suite (`tests/test_quick_api.py`) validates:

*   Attention kind mapping (incl. RoPE/ALiBi flags)
*   Embeddings (value patch) & positional choices
*   Norms (Layer/RMS/Scale), RevIN variants, block norms
*   Heads/loss/aggregation config propagation
*   Fit/predict shape sanity with stubbed model & sampler

Run:
```bash
pytest -q
```

## 🚀 Getting Started (low-level builder)

If you prefer configs + builder directly:

```python
import torch
from temporal.models import build_time_series_transformer
from temporal.configs import TransformerTimeSeriesConfig

config = TransformerTimeSeriesConfig(
    feature_size=1,
    context_length=128,
    prediction_length=24,
    d_model=64,
    encoder_blocks=[{"type": "default_encoder"}],
    output_head_config={"type": "linear", "output_size": 1},
)

model = build_time_series_transformer(config)
forecast = model.generate(torch.randn(1, 128, 1), prediction_length=24)
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
* **Production-Grade Code:** While built for research, the library is engineered with robust, high-quality components ready for production deployment.
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
