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

    def _ema_update(self, flat_latents: torch.Tensor, flat_indices: torch.Tensor, mask: torch.Tensor) -> None:
            """
            Memory-efficient EMA updates using index_add_ instead of a one-hot matrix.

            Args:
                flat_latents (torch.Tensor): Flattened latent vectors, shape [N, D].
                flat_indices (torch.Tensor): Flattened codebook indices, shape [N].
                mask (torch.Tensor): Flattened boolean mask, shape [N], where True indicates a valid position.
            """
            # --- Step 1: Filter to only include valid (unmasked) data ---
            active_latents = flat_latents[mask]
            active_indices = flat_indices[mask]

            if active_indices.numel() == 0:
                # Skip update if the batch contained only padding
                return

            # --- Step 2: Calculate cluster sizes and embedding sums efficiently ---
            # Count occurrences of each code index
            cluster_size = torch.zeros(
                self.codebook_size,
                device=active_latents.device,
                dtype=torch.long
            )
            cluster_size.index_add_(0, active_indices, torch.ones_like(active_indices, dtype=torch.long))

            # Sum the latent vectors corresponding to each code index
            embed_sum = torch.zeros(
                self.codebook_size, self.embedding_dim,
                device=active_latents.device,
                dtype=active_latents.dtype
            )
            embed_sum.index_add_(0, active_indices, active_latents)

            # --- Step 3: Apply EMA decay ---
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            self.ema_codebook.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

            # --- Step 4: Update codebook with Laplace smoothing ---
            n = self.ema_cluster_size.sum()
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.ema_eps) / (n + self.codebook_size * self.ema_eps) * n
            )
            self.codebook.weight.data.copy_(self.ema_codebook / smoothed_cluster_size.unsqueeze(1))

            # --- Step 5: Optional dead-code re-initialization (now more robust) ---
            if self.reinit_threshold > 0.0 and self.training:
                usage = self.ema_cluster_size / n
                dead_codes_mask = usage < self.reinit_threshold
                if dead_codes_mask.any():
                    dead_indices = torch.where(dead_codes_mask)[0]
                    num_dead = len(dead_indices)

                    # Resample from the CURRENT BATCH's active latents
                    # This is more effective than resampling from random noise or old values
                    random_latent_indices = torch.randint(0, active_latents.size(0), (num_dead,), device=active_latents.device)
                    new_codes = active_latents[random_latent_indices]

                    self.codebook.weight.data[dead_indices] = new_codes
                    # Reset the EMA statistics for the re-initialized codes
                    self.ema_cluster_size[dead_indices] = smoothed_cluster_size.mean() # A reasonable starting count
                    self.ema_codebook[dead_indices] = new_codes * self.ema_cluster_size[dead_indices].unsqueeze(1)

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
            Performs the forward pass for the VQ-VAE quantizer.

            Args:
                latents (torch.Tensor): The continuous latent vectors from the encoder.
                    Expected shape: [Batch, SequenceLength, d_model].
                mask (Optional[torch.Tensor]): A boolean mask for handling padding.
                    Shape: [Batch, SequenceLength], where True indicates a valid token.

            Returns:
                Dict[str, torch.Tensor]: A dictionary containing:
                    - "quantized": The quantized vectors with a straight-through gradient.
                    - "loss": The commitment loss for training the encoder.
                    - "indices": The discrete codebook indices for each position.
                    - "perplexity": A scalar metric indicating codebook usage.
                    - "usage": A vector indicating the usage frequency of each code.
            """
            # --- 1. Input Validation and Initialization ---
            if latents.dim() != 3 or latents.size(-1) != self.embedding_dim:
                raise AssertionError(f"Expected [B, L, D_model] latents, but got shape {latents.shape}")

            self._maybe_data_init(latents.detach()) # Lazy data-driven init if configured

            B, L, D = latents.shape
            if mask is None:
                mask = latents.new_ones(B, L, dtype=torch.bool)

            # --- 2. Find Closest Codes and Quantize ---
            # Find the index of the nearest codebook vector for each latent vector
            indices = self.get_indices(latents, mask=mask)  # Shape: [B, L]
            # Retrieve the corresponding quantized vectors from the codebook
            quantized = self.quantize(indices)              # Shape: [B, L, D]

            # --- 3. Calculate Loss ---
            # The VQ-VAE loss has two components:
            #  a) q_latent_loss: Pushes the codebook vectors towards the encoder outputs.
            #  b) e_latent_loss (commitment loss): Pushes the encoder to commit to a code.
            m = mask.unsqueeze(-1).float()
            q_latent_loss = F.mse_loss(quantized, latents.detach() * m, reduction='none').mean()
            e_latent_loss = F.mse_loss(quantized.detach(), latents * m, reduction='none').mean()

            # The final loss depends on whether EMA is used for codebook updates
            loss = self.commitment_cost * e_latent_loss
            if not self.use_ema:
                # In the original VQ-VAE, gradients flow to the codebook via this loss term.
                loss = loss + q_latent_loss

            # --- 4. Straight-Through Estimator ---
            # For the backward pass, we copy the gradients from the quantized output
            # directly to the continuous encoder output, "bypassing" the non-differentiable
            # argmin operation.
            quantized_st = latents + (quantized - latents).detach()

            # --- 5. EMA Codebook Update and Metrics (Memory-Optimized) ---
            metrics = {}
            if self.training:
                with torch.no_grad():
                    # Flatten tensors for efficient processing
                    flat_latents = latents.reshape(-1, D)
                    flat_mask = mask.reshape(-1)
                    flat_indices = indices.reshape(-1)

                    if self.use_ema:
                        # Call the optimized EMA update function that avoids one-hot encoding
                        self._ema_update(flat_latents, flat_indices, flat_mask)

                    # Calculate metrics directly from indices for the valid (unmasked) part of the batch
                    if flat_mask.sum() > 0:
                        active_indices = flat_indices[flat_mask]
                        # Use bincount for a highly efficient histogram of code usage
                        probs = torch.bincount(active_indices, minlength=self.codebook_size).float() / active_indices.numel()
                        perplexity = torch.exp(-(probs * (probs.clamp_min(1e-12)).log()).sum())
                        metrics = {"perplexity": perplexity, "usage": probs}

            # For eval mode, we can still compute perplexity for logging without updating the codebook
            elif not self.training:
                with torch.no_grad():
                    flat_indices = indices.reshape(-1)
                    probs = torch.bincount(flat_indices, minlength=self.codebook_size).float() / flat_indices.numel()
                    perplexity = torch.exp(-(probs * (probs.clamp_min(1e-12)).log()).sum())
                    metrics = {"perplexity": perplexity, "usage": probs}


            # --- 6. Assemble Final Output ---
            return {
                "quantized": quantized_st,
                "loss": loss,
                "indices": indices,
                "perplexity": metrics.get("perplexity", torch.tensor(0.0, device=latents.device)),
                "usage": metrics.get("usage", torch.zeros(self.codebook_size, device=latents.device)),
            }
