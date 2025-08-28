# Extending `temporal`

One of the most powerful features of `temporal` is its extensibility. You can easily add your own custom components, such as attention mechanisms, normalization layers, and output heads, with just a few lines of code.

## The `@register_module` Decorator

The key to extending `temporal` is the `@register_module` decorator. This decorator adds your custom component to the central registry, making it available to the builder.

The decorator takes two arguments:

1.  **`kind`:** The type of component you're adding (e.g., `"attention"`, `"normalization"`, `"output_head"`).
2.  **`name`:** A unique name for your component.

## Example: Adding a Custom Attention Mechanism

Let's say you've developed a new attention mechanism called `MyAwesomeAttention`. To add it to `temporal`, you would do the following:

1.  **Define Your Class:** First, you would define your new attention mechanism as a PyTorch module. It's a good practice to inherit from `BaseMultiHeadAttention` to get the standard query, key, and value projections.

    ```python
    from temporal.modules.attentions.base_attention import BaseMultiHeadAttention

    class MyAwesomeAttention(BaseMultiHeadAttention):
        # ... your implementation ...
    ```

2.  **Register Your Class:** Next, you would add the `@register_module` decorator to your class.

    ```python
    from temporal.registry.core import register_module
    from temporal.modules.attentions.base_attention import BaseMultiHeadAttention

    @register_module("attention", "my_awesome_attention")
    class MyAwesomeAttention(BaseMultiHeadAttention):
        # ... your implementation ...
    ```

That's it! Your new attention mechanism is now registered and can be used in your configurations.

## Using Your Custom Component

To use your new component, you simply need to specify its name in your `TransformerTimeSeriesConfig` object.

```python
from temporal.configs import TransformerTimeSeriesConfig

config = TransformerTimeSeriesConfig(
    # ... other parameters ...
    encoder_blocks=[{
        "type": "default_encoder",
        "attention_config": {
            "type": "my_awesome_attention",
            # ... any other parameters for your attention mechanism ...
        }
    }],
    # ... other parameters ...
)
```
The temporal builder will automatically find your registered component and instantiate it with the specified parameters.

This simple and powerful extension mechanism makes temporal a highly flexible and customizable framework for time series forecasting. You can easily experiment with new ideas and tailor the framework to your specific needs.