⏳ Temporal: A Modern Time Series Foundation Model Toolkit
Temporal is a modular, extensible deep learning framework for building next-generation foundation models for time series data. Inspired by Hugging Face’s transformers, Temporal offers a clean, registry-based interface for designing, training, and deploying powerful time series models — with full support for distributional outputs, autoregressive decoding, Monte Carlo dropout ensembles, and custom hybrid attention mechanisms.

Whether you're working in finance, energy, healthcare, sensors, or any other temporal domain, Temporal is built to be your go-to foundation for general-purpose time series modeling.

✨ Key Features
🧱 Block-based Transformer Architecture
Build encoder-decoder or decoder-only models with modular blocks composed of attention, FFN, normalization, and hybrid structures.

🧠 Registry-First Modular Design
Every component — attention, block, head, embedding, loss — is registered and dynamically resolved from config.

📐 Hybrid Multi-Head Attention with Custom Fusion
Assign different attention types (dot, sparse, wavelet) to each head and fuse with SE, attention, mean, or gated mechanisms.

🔁 Autoregressive & Multistep Decoding
Easily switch between .generate(), .generate_multistep(), and .forward() with full support for causal masks and caching.

🎯 Monte Carlo Dropout Ensembles
Perform stochastic inference at test time by enabling dropout for empirical ensemble sampling and uncertainty quantification.

📊 Distributional Output Heads
Swap in Linear, MultiQuantile, Gaussian, or TDistribution heads and automatically receive the matching loss (e.g., CRPS, NLL, quantile).

🔄 Flexible Output Head Aggregation
Fuse outputs from multiple output heads using registered aggregation strategies (e.g., mean, head2head, moe, low_rank).

🧪 Fully Configurable via JSON or Code
Drive your entire model via TransformerTimeSeriesConfig — with structured subconfigs for attention, heads, loss, and architecture.

💾 FromPretrained + Serialization
Save models, config, and training metadata with .save_pretrained() and reload them with .from_pretrained().

🔨 Example Use Cases
🔋 Forecasting electricity demand or load balancing

📈 Predicting asset prices or futures spreads

🩺 Modeling patient health data or hospital readmission risk

🛠️ Monitoring sensors in industrial IoT environments

💡 Learning residual signals or latent temporal regimes

🧪 Generating synthetic time series for augmentation

🚀 Get Started
bash
Copy
Edit
pip install temporal  # or your package name
python
Copy
Edit
from temporal.models import build_time_series_transformer
from temporal.configs import TransformerTimeSeriesConfig

config = TransformerTimeSeriesConfig(...)
model = build_time_series_transformer(config)
