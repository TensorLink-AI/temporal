# file: temporal/modules/quantizers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal.registry.core import register_module
from temporal.configs.quantizer_config import QuantizerConfig

from typing import Dict, Optional

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
    Vector Quantizer (VQ-VAE) with optional EMA codebook updates and metrics.

    Expects latents of shape [B, L, D]. Returns:
      - quantized: [B, L, D]
      - loss: scalar
      - indices: [B, L] (torch.long)
      - perplexity: scalar
      - usage: [K] normalized usage of each code
    """

    def __init__(self, config: QuantizerConfig):
        super().__init__(config)
        self.codebook_size = int(config.vocab_size)
        self.embedding_dim = int(config.d_model)
        kw = getattr(config, "kwargs", {}) or {}

        # Loss weights
        self.commitment_cost = float(kw.get("commitment_cost", 0.25))

        # EMA options (recommended)
        self.use_ema = bool(kw.get("use_ema", True))
        self.ema_decay = float(kw.get("ema_decay", 0.99))
        self.ema_eps = float(kw.get("ema_eps", 1e-5))

        # Optional cosine distance (unit-norm latents + codes)
        self.use_cosine = bool(kw.get("use_cosine", False))

        # Optional dead-code reinit threshold (fraction of batch assignments)
        self.reinit_threshold = float(kw.get("reinit_threshold", 0.0))  # 0.0 = disabled

        # Codebook
        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

        # EMA state (buffers so they move with .to(device))
        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
            self.register_buffer("ema_codebook", torch.zeros(self.codebook_size, self.embedding_dim))

        # Bookkeeping for lazy data-driven init (optional)
        self._initialized = False
        self._init_from_data = bool(kw.get("init_from_data", False))

    @torch.no_grad()
    def _maybe_data_init(self, latents: torch.Tensor) -> None:
        if self._initialized or not self._init_from_data:
            return
        B, L, D = latents.shape
        flat = latents.reshape(-1, D)
        num = min(self.codebook_size, flat.size(0))
        # Random sample latents to seed the codebook (cheap alternative to k-means).
        idx = torch.randperm(flat.size(0), device=flat.device)[:num]
        self.codebook.weight.data[:num].copy_(flat[idx])
        if num < self.codebook_size:
            rem = self.codebook_size - num
            self.codebook.weight.data[num:].uniform_(-flat.std().item(), flat.std().item())
        if self.use_ema:
            self.ema_codebook.copy_(self.codebook.weight.data)
            self.ema_cluster_size.zero_()
        self._initialized = True

    def _normalize_if_cosine(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cosine:
            return F.normalize(x, dim=-1)
        return x

    def get_indices(self, latents: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        latents: [B, L, D], mask: [B, L] (1=valid, 0=pad) or None
        returns: [B, L] long
        """
        B, L, D = latents.shape
        codes = self._normalize_if_cosine(self.codebook.weight)      # [K, D]
        x = self._normalize_if_cosine(latents).reshape(-1, D)        # [B*L, D]

        # Distances/similarities
        if self.use_cosine:
            # Maximize dot product ~ minimize (−x·e)
            sim = x @ codes.t()                                      # [B*L, K]
            encoding_indices = torch.argmax(sim, dim=1)
        else:
            # Squared L2: ||x||^2 + ||e||^2 − 2x·e
            x_sq = (x ** 2).sum(dim=1, keepdim=True)                 # [B*L, 1]
            e_sq = (codes ** 2).sum(dim=1).unsqueeze(0)              # [1, K]
            xe = x @ codes.t()                                       # [B*L, K]
            distances = x_sq + e_sq - 2 * xe
            encoding_indices = torch.argmin(distances, dim=1)

        encoding_indices = encoding_indices.view(B, L)

        if mask is not None:
            # For padded positions, set a benign index (0) to avoid accidental updates.
            encoding_indices = torch.where(mask.bool(), encoding_indices, torch.zeros_like(encoding_indices))

        return encoding_indices.long()

    def quantize(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [B, L] -> quantized: [B, L, D]"""
        return self.codebook(indices)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Alias for quantize (useful for clarity)."""
        return self.quantize(indices)

    def _ema_update(self, flat_latents: torch.Tensor, enc_onehot: torch.Tensor) -> None:
        """
        EMA updates per VQ-VAE v2.
        flat_latents: [N, D]
        enc_onehot:   [N, K] (float)
        """
        # N_i and latents sum per code
        cluster_size = enc_onehot.sum(0)                               # [K]
        embed_sum = enc_onehot.t() @ flat_latents                      # [K, D]

        # Decay
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_codebook.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        # Laplace smoothing to avoid div-by-zero
        n = self.ema_cluster_size + self.ema_eps
        self.codebook.weight.data.copy_(self.ema_codebook / n.unsqueeze(1))

        # Optional dead-code reinit
        if self.reinit_threshold > 0.0:
            total = self.ema_cluster_size.sum().clamp_min(self.ema_eps)
            usage = self.ema_cluster_size / total
            dead = usage < self.reinit_threshold
            if dead.any():
                # Reinit dead codes from current latents
                dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
                num_dead = dead_idx.numel()
                rand_idx = torch.randint(0, flat_latents.size(0), (num_dead,), device=flat_latents.device)
                self.codebook.weight.data[dead_idx] = flat_latents[rand_idx]
                # Reset EMA stats for those codes
                self.ema_cluster_size[dead_idx] = flat_latents.new_full((num_dead,), flat_latents.size(0) / self.codebook_size)
                self.ema_codebook[dead_idx] = self.codebook.weight.data[dead_idx]

    def _metrics(self, enc_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Perplexity & usage
        probs = enc_onehot.mean(dim=0)  # [K]
        perp = torch.exp(-(probs * (probs.clamp_min(1e-12)).log()).sum())
        return {"perplexity": perp, "usage": probs}

    def forward(
        self,
        latents: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        latents: [B, L, D], mask: [B, L]=1 for valid, 0 for pad (optional)
        """
        assert latents.dim() == 3 and latents.size(-1) == self.embedding_dim, "Expected [B, L, D_model] latents"

        # Lazy data-driven init if desired
        self._maybe_data_init(latents.detach())

        B, L, D = latents.shape
        if mask is None:
            mask = latents.new_ones(B, L, dtype=torch.bool)

        # Indices & quantized codes
        indices = self.get_indices(latents, mask=mask)                 # [B, L]
        quantized = self.quantize(indices)                             # [B, L, D]

        # Losses
        # Encourage encoder outputs to match chosen codes (commitment) and vice versa.
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            e_latent_loss = F.mse_loss(quantized.detach() * m, latents * m)
            q_latent_loss = F.mse_loss(quantized * m, (latents * m).detach())
        else:
            e_latent_loss = F.mse_loss(quantized.detach(), latents)
            q_latent_loss = F.mse_loss(quantized, latents.detach())

        loss = self.commitment_cost * e_latent_loss
        if not self.use_ema:
            # Gradient to codebook via codebook loss (original VQ-VAE)
            loss = loss + q_latent_loss

        # Straight-through estimator
        quantized_st = latents + (quantized - latents).detach()

        # EMA update of codebook (no gradient through codes)
        metrics = {}
        if self.training:
            with torch.no_grad():
                flat_latents = latents.reshape(-1, D)
                flat_mask = mask.reshape(-1)                            # [B*L]
                flat_idx = indices.reshape(-1)                          # [B*L]

                # One-hot encodings only for valid positions
                enc_onehot = F.one_hot(flat_idx, num_classes=self.codebook_size).float()
                if flat_mask is not None:
                    enc_onehot = enc_onehot * flat_mask.unsqueeze(-1).float()

                if self.use_ema:
                    self._ema_update(flat_latents, enc_onehot)

                metrics = self._metrics(enc_onehot)

        else:
            # In eval, still compute perplexity/usage for logging (cheap)
            with torch.no_grad():
                flat_idx = indices.reshape(-1)
                enc_onehot = F.one_hot(flat_idx, num_classes=self.codebook_size).float()
                metrics = self._metrics(enc_onehot)

        out = {
            "quantized": quantized_st,
            "loss": loss,
            "indices": indices,
            "perplexity": metrics.get("perplexity", torch.tensor(0.0, device=latents.device)),
            "usage": metrics.get("usage", torch.zeros(self.codebook_size, device=latents.device)),
        }
        return out
