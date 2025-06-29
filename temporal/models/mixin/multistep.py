import torch
import torch.nn as nn
from typing import Optional

class MultiStepMixin:
    """
    Mixin for models that generate forecasts in a single forward pass (non-autoregressive).
    """

    @torch.no_grad()
    def generate_multistep(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generates a multi-step forecast in a single pass.

        This method supports both raw feature inputs and pre-processed embeddings.
        - If `input_ids` has 3 dimensions and the last dimension matches `config.feature_size`,
          it is treated as raw features and passed through the `preprocessor`.
        - Otherwise, it is assumed to be pre-processed embeddings.
        """
        self.eval()

        # 1) Preprocess inputs if they are raw features
        if hasattr(self, "preprocessor") and input_ids.ndim == 3 and input_ids.shape[-1] == self.config.feature_size:
            proc = self.preprocessor.process(
                input_values=input_ids,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=False,
                verbose=False,
            )
            processed_inputs = proc["hidden_states"]
            attention_mask = proc["attention_mask"]
        else:
            processed_inputs = input_ids  # Assumed to be embeddings

        # 2) Pass inputs through the main model forward pass
        # This is simpler than the autoregressive case as we don't need a decoder loop.
        # The forward pass will handle the encoder and any subsequent processing.
        outputs = self.forward(
            encoder_inputs=processed_inputs,
            attention_mask=attention_mask,
            **kwargs,
        )

        # 3) Project the final hidden states to the prediction space
        last_hidden_state = outputs.last_hidden_state

        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model is missing output_heads required to project hidden states to features.")

        if isinstance(self.output_heads, nn.ModuleList):
            head_outputs = [head(last_hidden_state) for head in self.output_heads]
            if hasattr(self, 'head_aggregator') and self.head_aggregator:
                predictions = self.head_aggregator(head_outputs)
            else:
                # Default to the first head if no aggregator
                predictions = head_outputs[0]
        else:
            predictions = self.output_heads(last_hidden_state)

        return predictions
