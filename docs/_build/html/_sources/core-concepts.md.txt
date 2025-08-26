```markdown
# Core Concepts of `temporal`

The `temporal` package is designed to be both powerful and easy to use. This is made possible by a few core concepts that work together to provide a flexible and extensible framework for time series forecasting.

## 1. Configuration-Driven Design

In `temporal`, you don't build models with imperative code. Instead, you define your model's architecture in a configuration object. This has several advantages:

* **Readability:** Your model's architecture is defined in a single, easy-to-read object.
* **Reproducibility:** You can easily save and share your configurations to reproduce your results.
* **Experimentation:** It's easy to experiment with different architectures by simply changing the configuration.

The main configuration class is `TransformerTimeSeriesConfig`, which is a dataclass that allows you to specify every aspect of your model's architecture.

## 2. The Registry

The `temporal` registry is a global dictionary that keeps track of all the available components, such as attention mechanisms, normalization layers, and output heads. This is what makes `temporal` so extensible.

To add a new component, you simply need to define your new class and add the `@register_module` decorator. For example, to add a new attention mechanism, you would do the following:

```python
from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention

@register_module("attention", "my_custom_attention")
class MyCustomAttention(BaseMultiHeadAttention):
    # ... your implementation ...
-```
Once registered, your new component is available to the builder and can be used in your configurations.

3. The Builder
The temporal builder is responsible for taking your configuration and instantiating the correct components from the registry. It then assembles them into a complete model.

The main builder function is build_time_series_transformer, which takes your TransformerTimeSeriesConfig object and returns a fully instantiated PyTorch model. The builder handles all the details of creating the model, so you can focus on defining your architecture.

These three core concepts—configuration-driven design, the registry, and the builder—work together to create a powerful and flexible framework for time series forecasting. They allow you to easily experiment with different architectures, reproduce your results, and extend the framework with your own custom components.