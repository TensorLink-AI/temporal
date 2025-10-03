from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any, Sequence

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    # Avoid circular import while allowing type hints
    from temporal.models.transformer_model import TransformerTemporalModel


class TrainingStrategy(ABC):
    """Abstract base class for defining training strategies."""

    @abstractmethod
    def __call__(
        self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Determines the input for the decoder at each training step.
        """
        raise NotImplementedError()

    def get_loss(self) -> torch.Tensor | None:
        """Optional: return a custom scalar loss computed by the strategy."""
        return None


class TeacherForcingStrategy(TrainingStrategy):
    """Always uses ground-truth inputs."""
    def __call__(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return decoder_inputs

class ScheduledSamplingStrategy(TrainingStrategy):
    """
    Scheduled Sampling in native space (pragmatic for your patch-AR design).
    Replaces decoder inputs with a *point* forecast sometimes.
    """

    def __init__(
        self,
        total_steps: int,
        sampling_probability: float = 0.5,
        prediction_method: str = "median",   # 'median' or 'mean'
        return_raw: bool = True,
    ):
        self.total_steps = int(total_steps)
        self.sampling_probability = float(sampling_probability)
        self.prediction_method = prediction_method
        self.return_raw = bool(return_raw)

    def __call__(self, model: "TransformerTemporalModel", decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        current_step = kwargs.get("current_step", 0)
        p = self._get_sampling_probability(int(current_step))

        # no replacement → teacher forcing
        if torch.rand(1, device=decoder_inputs.device).item() >= p:
            return decoder_inputs

        targets = kwargs.get("targets")
        if targets is None:
            raise ValueError("ScheduledSamplingStrategy requires 'targets' to infer prediction_length.")

        # Preserve original training mode because generate() sets eval()
        was_training = model.training
        try:
            with torch.no_grad():  # hard no-grad guard
                # NOTE: generate() currently ignores `prediction_strategy`.
                # If you want 'mean' vs 'median', plumb it through in your generate() implementation.
                gen = model.generate(
                    encoder_inputs=kwargs.get("encoder_inputs"),
                    decoder_inputs=decoder_inputs,
                    prediction_length=targets.shape[1],
                    attention_mask=kwargs.get("attention_mask"),
                    differentiable=False,        # belt-and-braces with torch.no_grad
                    return_samples=False,        # point, not paths
                    sampling=False,              # deterministic via predict()
                    prediction_strategy=self.prediction_method,  # only effective if wired in generate()
                    return_raw=self.return_raw,
                    return_bundle=False,         # just the tensor
                    quantile_levels=None,        # don't request quantiles
                )
        finally:
            # Restore the original mode so the main forward stays in train()
            model.train(was_training)

        # Shape fixups to match decoder_inputs [B, T_in, F_in]
        if gen.ndim == 2:
            gen = gen.unsqueeze(-1)  # [B,T] -> [B,T,1]
        num_in_features = decoder_inputs.shape[-1]
        gen = gen[..., :num_in_features]

        return gen

    def _get_sampling_probability(self, current_step: int) -> float:
        if self.total_steps <= 0:
            return 0.0
        return self.sampling_probability * min(1.0, current_step / self.total_steps)



if TYPE_CHECKING:
    from temporal.models.transformer_model import TransformerTemporalModel


class RLTrainingStrategy:
    """
    Direct-reward optimization on an ensemble [B, T, F, S].

    1) Try to obtain S "paths" by asking the model to compute Q=S quantiles
       and reading ForecastBundle.quantiles  -> [B, T, F, S].
    2) If the head cannot yield quantiles, fall back to step-wise sampling that
       respects heads which require T==1 sampling (e.g., DistPred).
    """

    def __init__(
        self,
        reward_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        num_samples: int = 100,
        quantile_levels: Optional[Sequence[float]] = None,  # e.g., [0.1, ..., 0.9]
        prefer_quantiles: bool = True,
        return_raw: bool = True,
    ):
        """
        Args:
            reward_loss_fn: callable(paths, targets) -> scalar tensor.
                            paths: [B, T, F, S]; targets: [B, T] or [B, T, F]
            num_samples: default S when quantile_levels is None.
            quantile_levels: explicit quantiles to request (S = len(levels)).
            prefer_quantiles: try ForecastBundle.quantiles first.
            return_raw: keep normalized values if your loss expects it.
        """
        self.reward_loss_fn = reward_loss_fn
        self.return_raw = bool(return_raw)
        self.prefer_quantiles = bool(prefer_quantiles)

        if quantile_levels is not None:
            # validate (0,1)
            qs = [float(q) for q in quantile_levels]
            if not all(0.0 < q < 1.0 for q in qs):
                raise ValueError(f"All quantiles must be in (0,1). Got {quantile_levels}")
            self.quantile_levels = qs
            self.num_samples = len(qs)
        else:
            self.quantile_levels = None
            self.num_samples = int(num_samples)

        self.latest_loss: Optional[torch.Tensor] = None

    # ---------------- Public API ----------------

    def __call__(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes self.latest_loss and returns decoder_inputs unchanged (so your outer
        loop can still call model(...) if needed).
        """
        self._rl_forward(model, decoder_inputs, **kwargs)
        return decoder_inputs

    def get_loss(self) -> torch.Tensor | None:
        return self.latest_loss

    # ---------------- Internals -----------------

    def _rl_forward(self, model: TransformerTemporalModel, decoder_inputs: torch.Tensor, **kwargs) -> None:
        targets = kwargs.get("targets")
        if targets is None:
            raise ValueError("RLTrainingStrategy requires 'targets' for reward computation.")

        device = decoder_inputs.device
        dtype = decoder_inputs.dtype
        targets = self._to_f32(targets, device)

        horizon = targets.shape[1]
        paths: Optional[torch.Tensor] = None

        # A) Prefer ForecastBundle.quantiles (fast, vectorized)
        if self.prefer_quantiles:
            qs = self.quantile_levels
            if qs is None:
                # Build mid-point quantiles in (0,1), length = num_samples
                qs = self._midpoint_quantiles(self.num_samples, device, dtype).tolist()

            bundle = model.generate(
                encoder_inputs=kwargs.get("encoder_inputs"),
                decoder_inputs=decoder_inputs,
                attention_mask=kwargs.get("attention_mask"),
                prediction_length=horizon,
                # critical:
                differentiable=True,
                return_bundle=True,
                quantile_levels=qs,
                post_quantiles=True,
                return_raw=self.return_raw,
            )
            paths = getattr(bundle, "quantiles", None)  # [B, T, F, S] or None

        # B) Fallback: step-wise sampling (T==1) that honors heads with T==1 sample()
        if paths is None:
            paths = self._sample_paths_stepwise(
                model=model,
                decoder_inputs=decoder_inputs,
                attention_mask=kwargs.get("attention_mask"),
                encoder_inputs=kwargs.get("encoder_inputs"),
                horizon=horizon,
                num_paths=self.num_samples,
            )  # [B, T, F, S]

        # C) Compute final reward
        self.latest_loss = self.reward_loss_fn(paths, targets)

    @torch.enable_grad()
    def _sample_paths_stepwise(
        self,
        *,
        model: TransformerTemporalModel,
        decoder_inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_inputs: Optional[torch.Tensor],
        horizon: int,
        num_paths: int,
    ) -> torch.Tensor:
        """
        Produces [B, T, F, S] by:
          1) getting full-horizon head params with grads (return_params=True),
          2) calling head.sample(...) per time step with T==1 slices,
             stacking across time into paths.
        """
        # (1) Get head params once (differentiable)
        params = model.generate(
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            attention_mask=attention_mask,
            prediction_length=horizon,
            differentiable=True,
            return_params=True,      # tensor or dict of tensors [B, T, ...]
            return_raw=self.return_raw,
        )
        head = model._get_primary_head()  # type: ignore[attr-defined]

        B = decoder_inputs.shape[0]
        F = int(getattr(model.config, "feature_size", 1))

        def _slice_t(p: Any, t: int) -> Any:
            if isinstance(p, dict):
                out: Dict[str, Any] = {}
                for k, v in p.items():
                    if torch.is_tensor(v) and v.dim() >= 2 and v.size(1) > t:
                        out[k] = v[:, t:t+1, ...]  # keep T==1
                    else:
                        out[k] = v
                return out
            return p[:, t:t+1, ...]

        def _btf(x: torch.Tensor) -> torch.Tensor:
            # ensure [B, 1, F] for per-step samples
            if x.ndim == 2:  # [B, F] -> [B, 1, F]
                x = x.unsqueeze(1)
            if x.ndim != 3:
                raise ValueError(f"Expected [B,1,F] or [B,T,F], got {tuple(x.shape)}")
            if x.size(-1) == 1 and F > 1:
                x = x.expand(x.size(0), x.size(1), F)
            return x

        cols = []
        state = None  # thread state if the head returns/consumes it

        for t in range(horizon):
            p_t = _slice_t(params, t)  # keep T==1

            # (2) Try modern signature with num_paths; fallback to legacy.
            try:
                s_t = head.sample(p_t, state=state, num_paths=num_paths)  # ideally [B,1,F,S]
            except TypeError:
                s_t = head.sample(p_t, state=state)  # might be [B,1,F] or [B,1,F,S]

            new_state = None
            if isinstance(s_t, tuple):
                s_t, new_state = s_t[0], s_t[1]
            elif isinstance(s_t, dict):
                new_state = s_t.get("state", None)
                s_t = s_t.get("samples", s_t.get("paths", None))
                if s_t is None:
                    raise TypeError("head.sample must return Tensor or dict with 'samples'/'paths'.")

            s_t = _btf(s_t)  # [B,1,F] or [B,1,F,S]

            # If head returned no S dimension, tile deterministically across S
            if s_t.ndim == 3:
                s_t = s_t.unsqueeze(-1).expand(-1, -1, -1, num_paths)

            cols.append(s_t)
            state = new_state if new_state is not None else state

        paths = torch.cat(cols, dim=1)  # [B, T, F, S]
        return paths

    # ---------------- Utils ----------------

    @staticmethod
    def _to_f32(x: Any, device: torch.device) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    @staticmethod
    def _midpoint_quantiles(S: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 0<S quantiles strictly in (0,1)
        return (torch.arange(S, device=device, dtype=dtype) + 0.5) / S
