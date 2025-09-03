# Transformer Capabilities in `temporal`

The `temporal` package provides a powerful and flexible implementation of the Transformer architecture, tailored specifically for time series forecasting. Here's a deep dive into its key capabilities.

## 🧱 Core Modules

At the heart of every `temporal` model are three core modules that work together to process your time series data.

* **Embedding Layer:** The embedding layer is responsible for converting the raw input time series into a high-dimensional representation that the Transformer can process. `temporal` provides a variety of embedding options, including:
    * **`TimeSeriesValueEmbedding`:** A simple linear projection.
    * **`TimeSeriesPatchEmbedding`:** For patch-based models.
    * A variety of positional embeddings, including `SinusoidalPositionalEmbedding`, `RotaryPositionalEmbedding`, and `LearnedAbsolutePositionalEmbedding`.
* **Encoder Layer:** The encoder is responsible for processing the input sequence and creating a rich contextual representation. The `TimeSeriesTransformerEncoder` is a stack of encoder layers, each of which contains a self-attention mechanism and a feed-forward network.
* **Decoder Layer:** The decoder is responsible for generating the output sequence. The `TimeSeriesTransformerDecoder` is a stack of decoder layers, each of which contains a self-attention mechanism, a cross-attention mechanism (for attending to the encoder's output), and a feed-forward network.

## 🏗️ Architectural Flexibility

`temporal` allows you to define a wide variety of Transformer architectures with ease.

* **Encoder-Decoder, Encoder-Only, and Decoder-Only Models:** You can specify the overall layout of your model using the `TransformerArchitectureConfig`.
* **Customizable Layers:** You can mix and match different types of layers within the same model. For example, you can have a model with standard attention in the early layers and a more efficient attention mechanism in the later layers.
* **Patch-Based Modeling:** The framework has first-class support for patch-based time series modeling.

## ✨ Advanced Attention Mechanisms

`temporal` provides a rich set of attention mechanisms beyond the standard self-attention.

* **`FullAttention`:** The standard, full self-attention mechanism.
* **`FlashAttention`:** A highly efficient implementation that uses the `flash-attn` library.
* **`LSEAttention` (Log-Sum-Exp Attention):** A memory-efficient and numerically stable attention mechanism.
* **`DifferentialAttention` (DiffWist):** A novel attention mechanism featuring a learnable gating mechanism and grouped-query attention.
* **`PatternedMultiHeadAttention`:** This allows you to apply fixed, predefined patterns to the attention matrix, such as local, sliding, or dilated attention.
* **`HybridAttention`:** A powerful feature that allows you to combine different attention mechanisms within a single layer.

## 🔮 Probabilistic Forecasting

`temporal` has extensive support for probabilistic forecasting, allowing you to model uncertainty in your predictions.

* **A Variety of Output Heads:**
    * **`LinearOutputHead`:** For simple point forecasts.
    * **`GaussianHead`:** For predicting the mean and standard deviation of a Gaussian distribution.
    * **`QuantileRegressionOutputHead`:** For directly predicting multiple quantiles.
    * **`DistPredHead`:** Designed for CRPS loss, this head outputs an ensemble of values to approximate the predictive distribution.
    * **`MixtureOutputHead`:** For Mixture Density Networks (MDNs).
    * **`TimeFlowHead`:** A specialized head for the TimeFlow model, a diffusion-based approach.
* **Flexible Loss Functions:** The framework provides a corresponding set of loss functions for each output head.

## 🧩 Extensibility

The `temporal` package is designed to be easily extensible.

* **Central Registry:** All components are registered in a central registry, making it easy to add your own custom components.
* **Configuration-Driven Builders:** The builder classes use the configuration objects to automatically instantiate and connect all the necessary components.
* **Hugging Face Compatibility:** The framework is designed to be compatible with the Hugging Face ecosystem.