# file: temporal/modules/quantizers.py
import torch
import torch.nn as nn
from temporal.registry.core import register_module
from temporal.configs.quantizer_config import QuantizerConfig

class BaseQuantizer(nn.Module):
    """
    Abstract base class for quantizers.
    """
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")

@register_module("quantizer", "mean_std_bins")
class MeanSTDBinsQuantizer(BaseQuantizer):
    """
    Quantizes a time series based on bins derived from its mean and standard
    deviation. Each feature is quantized independently.
    """
    def __init__(self, config: QuantizerConfig):
        super().__init__(config)
        
        # Create the bin boundaries. These are not trainable.
        # We'll create self.config.vocab_size bins.
        # The bins are centered around 0 and scaled by a factor.
        # Let's assume a range of [-3, 3] standard deviations is sufficient.
        bin_edges = torch.linspace(-3.0, 3.0, self.config.vocab_size - 1)
        self.register_buffer("bin_edges", bin_edges)

        # Embedding layer to project quantized values back to the model's dimension
        self.embedding = nn.Embedding(self.config.vocab_size, config.d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape `[B, L, F]`.

        Returns:
            torch.Tensor: The quantized and embedded tensor of shape `[B, L, D_model]`.
        """
        # 1. Normalize the input tensor per feature
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + 1e-6  # Add epsilon for stability
        normalized_x = (x - mean) / std

        # 2. Digitize the normalized values into bins
        # The shape of quantized_indices will be the same as normalized_x
        quantized_indices = torch.bucketize(normalized_x, self.bin_edges)

        # 3. Project the quantized indices to the model's dimension
        quantized_embedding = self.embedding(quantized_indices)
        
        # 4. Sum the embeddings across the feature dimension
        # This will result in a tensor of shape [B, L, D_model]
        summed_embedding = torch.sum(quantized_embedding, dim=-2)

        return summed_embedding


@register_module("quantizer", "vq_vae")
class VQVAEQuantizer(BaseQuantizer):
    """
    Vector Quantizer module from VQ-VAE.

    Args:
        config (QuantizerConfig): Configuration object containing vocab_size,
                                   d_model, and commitment_cost.
    """
    def __init__(self, config: QuantizerConfig):
        super().__init__(config)
        self.codebook_size = config.vocab_size
        self.embedding_dim = config.d_model
        self.commitment_cost = config.kwargs.get("commitment_cost", 0.25)

        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)

    def get_indices(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Gets the indices of the closest codebook vectors.
        Args:
            latents (torch.Tensor): The continuous latent vectors, shape [B, L, D_model].
        Returns:
            torch.Tensor: The indices of the closest embeddings, shape [B, L].
        """
        B, L, D = latents.shape
        flat_latents = latents.view(-1, self.embedding_dim) # [B*L, D]

        # Calculate L2 distance between latents and codebook
        distances = (torch.sum(flat_latents**2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight**2, dim=1)
                     - 2 * torch.matmul(flat_latents, self.codebook.weight.t()))

        # Find the closest codebook vector for each latent vector
        encoding_indices = torch.argmin(distances, dim=1) # [B*L]
        return encoding_indices.view(B, L)


    def forward(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VQ-VAE quantizer.

        Args:
            latents (torch.Tensor): The continuous latent vectors from the encoder,
                                   shape [B, L, D_model].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'quantized': The quantized latent vectors.
                - 'loss': The VQ commitment loss.
                - 'indices': The indices of the chosen codebook vectors.
        """
        encoding_indices = self.get_indices(latents) # [B, L]
        quantized = self.codebook(encoding_indices) # [B, L, D]

        # VQ Loss Calculation
        # 1. Commitment Loss: encourage encoder outputs to be close to chosen codebook vectors.
        e_latent_loss = F.mse_loss(quantized.detach(), latents)
        # 2. Codebook Loss: encourage codebook vectors to be close to encoder outputs.
        q_latent_loss = F.mse_loss(quantized, latents.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = latents + (quantized - latents).detach()

        return {
            "quantized": quantized,
            "loss": loss,
            "indices": encoding_indices
        }
