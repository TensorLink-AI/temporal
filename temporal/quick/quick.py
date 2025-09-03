# temporal/quick/api.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

# ---- Wire to your core (all exist in your repo) ----
from temporal.models import build_time_series_transformer
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.configs.transformer_block_config import transformer_block_config_from_dict
from temporal.configs.embedding_config import embedding_config_from_dict, EmbeddingConfig
from temporal.configs.normalization_config import normalization_config_from_dict, NormalizationConfig
from temporal.configs.output_head_config import output_head_config_from_dict, OutputHeadConfig
from temporal.configs.loss_config import loss_config_from_dict, LossConfig
from temporal.configs.head_aggregation_config import head_aggregation_config_from_dict, HeadAggregationConfig
from temporal.utils.ensemble import EnsembleSampler


# ---------- helpers & guards ----------

ATTN_KIND_MAP = {
    "full": "full_attention",
    "patterned": "patterned_attention",
    "flash": "flash_attention",
    "lse": "lse_attention",
    "hybrid": "hybrid_attention",
    "diff": "diffwist_attention",
    "diffwist": "diffwist_attention",
}

POS_EMB_TYPES = {"sinusoidal", "learned_abs", "conv_pos", "fourier", "none"}
VAL_EMB_TYPES = {"value", "patch"}

def _validate_head_dim(d_model: int, n_heads: int):
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

def _validate_rope(d_model: int, n_heads: int):
    _validate_head_dim(d_model, n_heads)
    d_head = d_model // n_heads
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires even d_head; got d_model={d_model}, n_heads={n_heads} → d_head={d_head}.")

def _to_device(x, device):
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, dict): return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_to_device(v, device) for v in x)
    return x

def _forward_kwargs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return dict(inputs)

def _l2_loss(out, target):
    if isinstance(out, dict) and "preds" in out:
        pred = out["preds"]
    else:
        pred = out
    if target is None:
        raise RuntimeError("Targets required to compute fallback loss.")
    return torch.mean((pred - target) ** 2)


# ---------- BuildSpec captures “quick” knobs ----------

@dataclass(frozen=True)
class BuildSpec:
    # Topology
    arch: str = "encoder"                   # "encoder" | "decoder" | "encdec"
    n_encoder_layers: int = 6
    n_decoder_layers: int = 0               # used if arch in {"decoder","encdec"}

    # Sizes
    prediction_length: int = 24             # horizon
    context_length: int = 96
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    max_pos: int = 4096

    # Attention
    attention_kind: str = "full"            # maps via ATTN_KIND_MAP
    attention_cfg: Dict[str, Any] = field(default_factory=dict)
    use_rope: bool = False
    rope_base: int = 10000
    use_alibi: bool = False
    qk_layernorm: bool = False
    attn_bias: bool = True
    # Patterned attention windowing
    pattern: Optional[Dict[str, Any]] = None  # {"type":"local|sliding|dilated|global", "window_size":..., ...}

    # Embeddings
    value_embedding: Dict[str, Any] = field(default_factory=lambda: {
        "type": "value",                 # "value" | "patch"
        "d_model": 256,                  # for "value"
    })
    positional_embedding: Dict[str, Any] = field(default_factory=lambda: {
        "type": "sinusoidal"             # "sinusoidal" | "learned_abs" | "conv_pos" | "fourier" | "none"
    })

    # Norms (model-level around embeddings) + optional instance/RevIN
    layer_norm: Dict[str, Any] = field(default_factory=lambda: {"type": "layer", "eps": 1e-5, "elementwise_affine": True})
    instance_norm: Optional[Dict[str, Any]] = None     # e.g., {"type":"revin","num_features":256, ...}

    # Block-level normalization inside each transformer block
    block_norm: Dict[str, Any] = field(default_factory=lambda: {"type": "layer", "eps": 1e-5, "elementwise_affine": True})

    # Head / loss / aggregation
    head: Dict[str, Any] = field(default_factory=lambda: {"type": "quantile", "quantiles": [0.05, 0.5, 0.95]})
    loss: Dict[str, Any] = field(default_factory=lambda: {"type": "crps"})
    head_agg: Dict[str, Any] = field(default_factory=lambda: {"type": "mean"})

    # Inference
    mc_dropout_samples: int = 100

    # Extras (escape hatch)
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------- High-level class ----------

class Temporal:
    """
    Quick, expressive front-door to Temporal’s model builder.
    Covers attention variants (RoPE/ALiBi/qk-LN), norms (LayerNorm/RMS/RevIN),
    value patch embeddings, positional embeddings, heads/losses, train/predict.
    """

    def __init__(self, spec: Optional[BuildSpec] = None):
        self.spec = spec or BuildSpec()
        self.model = None
        self.config: Optional[TransformerTimeSeriesConfig] = None

    # --- topology ---
    def as_encoder(self, n_layers: int) -> "Temporal":
        return self._r(spec=replace(self.spec, arch="encoder", n_encoder_layers=n_layers, n_decoder_layers=0))

    def as_decoder(self, n_layers: int) -> "Temporal":
        return self._r(spec=replace(self.spec, arch="decoder", n_encoder_layers=0, n_decoder_layers=n_layers))

    def as_encdec(self, n_enc: int, n_dec: int) -> "Temporal":
        return self._r(spec=replace(self.spec, arch="encdec", n_encoder_layers=n_enc, n_decoder_layers=n_dec))

    def with_dims(self, d_model: int, n_heads: int, *, dropout: Optional[float] = None, max_pos: Optional[int] = None) -> "Temporal":
        sp = replace(self.spec, d_model=d_model, n_heads=n_heads)
        if dropout is not None: sp = replace(sp, dropout=dropout)
        if max_pos is not None: sp = replace(sp, max_pos=max_pos)
        return self._r(spec=sp)

    def with_context(self, L: int) -> "Temporal":
        return self._r(spec=replace(self.spec, context_length=L))

    def with_horizon(self, H: int) -> "Temporal":
        return self._r(spec=replace(self.spec, prediction_length=H))

    # --- attention ---
    def with_attention(self, kind: str = "full", **cfg) -> "Temporal":
        kind = kind.lower()
        if kind not in ATTN_KIND_MAP:
            raise ValueError(f"Unknown attention kind '{kind}'. Options: {sorted(ATTN_KIND_MAP)}")
        sp = replace(self.spec, attention_kind=kind, attention_cfg=cfg)
        return self._r(spec=sp)

    def with_rope(self, *, base: int = 10000) -> "Temporal":
        return self._r(spec=replace(self.spec, use_rope=True, rope_base=base))

    def with_alibi(self, enabled: bool = True) -> "Temporal":
        return self._r(spec=replace(self.spec, use_alibi=bool(enabled)))

    def with_qk_layernorm(self, enabled: bool = True) -> "Temporal":
        return self._r(spec=replace(self.spec, qk_layernorm=bool(enabled)))

    def with_attention_bias(self, enabled: bool = True) -> "Temporal":
        return self._r(spec=replace(self.spec, attn_bias=bool(enabled)))

    def with_attention_pattern(self, *, kind: str, window_size: int = 0,
                               stride: Optional[int] = None, dilation: int = 1,
                               global_indices: Optional[Sequence[int]] = None) -> "Temporal":
        """
        For patterned attention kinds ("local", "sliding", "dilated", "global").
        Takes effect if attention_kind == "patterned".
        """
        pat = {
            "type": kind,
            "window_size": int(window_size),
            "stride": None if stride is None else int(stride),
            "dilation": int(dilation),
            "global_indices": list(global_indices) if global_indices is not None else [],
        }
        return self._r(spec=replace(self.spec, pattern=pat))

    # --- embeddings ---
    def value_embedding(self, emb_type: str = "value", **cfg) -> "Temporal":
        emb_type = emb_type.lower()
        if emb_type not in VAL_EMB_TYPES:
            raise ValueError(f"Unknown value embedding '{emb_type}'. Options: {sorted(VAL_EMB_TYPES)}")
        spec_cfg = {"type": emb_type, **cfg}
        return self._r(spec=replace(self.spec, value_embedding=spec_cfg))

    def positional_embedding(self, emb_type: str = "sinusoidal", **cfg) -> "Temporal":
        emb_type = emb_type.lower()
        if emb_type not in POS_EMB_TYPES:
            raise ValueError(f"Unknown positional embedding '{emb_type}'. Options: {sorted(POS_EMB_TYPES)}")
        spec_cfg = {"type": emb_type, **cfg}
        return self._r(spec=replace(self.spec, positional_embedding=spec_cfg))

    # --- norms ---
    def with_layer_norm(self, *, kind: str = "layer", eps: float = 1e-5, elementwise_affine: bool = True) -> "Temporal":
        """
        Model/embedding-level norm around embeddings: kind in {"layer","rms","scale"}.
        """
        ln = {"type": kind, "eps": float(eps), "elementwise_affine": bool(elementwise_affine)}
        return self._r(spec=replace(self.spec, layer_norm=ln))

    def with_instance_norm(self, **cfg) -> "Temporal":
        """
        Instance/RevIN at model level. Examples:
          - with_instance_norm(type="revin", num_features=256, affine=True, subtract_last=False)
          - with_instance_norm(type="dynamic_revin", num_features=256, affine=True, mapper="mlp", hidden_dim=16)
          - with_instance_norm(type="revin2d", num_features=256, affine=True)
        """
        t = cfg.get("type")
        if t not in {"revin", "dynamic_revin", "revin2d"}:
            raise ValueError("instance_norm.type must be one of {'revin','dynamic_revin','revin2d'}")
        if "num_features" not in cfg:
            raise ValueError("instance_norm requires 'num_features' (usually d_model or feature size).")
        return self._r(spec=replace(self.spec, instance_norm=dict(cfg)))

    def with_block_norm(self, *, kind: str = "layer", eps: float = 1e-5, elementwise_affine: bool = True) -> "Temporal":
        """
        Per-block normalization inside transformer blocks.
        """
        bn = {"type": kind, "eps": float(eps), "elementwise_affine": bool(elementwise_affine)}
        return self._r(spec=replace(self.spec, block_norm=bn))

    # --- heads & loss & aggregation ---
    def head_quantile(self, quantiles: Sequence[float]) -> "Temporal":
        return self._r(spec=replace(self.spec,
                                    head={"type": "quantile", "quantiles": list(quantiles)},
                                    loss={"type": "crps"}))

    def head_gaussian(self) -> "Temporal":
        return self._r(spec=replace(self.spec,
                                    head={"type": "gaussian"},
                                    loss={"type": "nll", "kwargs": {"distribution_type": "gaussian"}}))

    def head_mixture(self, kind: str = "student_t", components: int = 3, **kw) -> "Temporal":
        head = {"type": "mixture", "kind": kind, "components": int(components), **kw}
        loss = {"type": "nll", "kwargs": {"distribution_type": kind}}
        return self._r(spec=replace(self.spec, head=head, loss=loss))

    def head_aggregator(self, *, kind: str = "mean", **kw) -> "Temporal":
        return self._r(spec=replace(self.spec, head_agg={"type": kind, "kwargs": dict(kw)}))

    def with_mc_samples(self, n: int) -> "Temporal":
        return self._r(spec=replace(self.spec, mc_dropout_samples=int(n)))

    # --- build / train / infer ---
    def build(self) -> "Temporal":
        # Validate dims + RoPE constraints
        _validate_head_dim(self.spec.d_model, self.spec.n_heads)
        if self.spec.use_rope:
            _validate_rope(self.spec.d_model, self.spec.n_heads)

        cfg = _compile_config(self.spec)
        self.model = build_time_series_transformer(cfg)
        self.config = cfg
        return self

    def fit_dataloader(
        self,
        train_loader: DataLoader,
        *,
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        amp: Optional[bool] = None,
        optimizer_ctor = torch.optim.AdamW,
        grad_clip: Optional[float] = None,
    ) -> "Temporal":
        assert self.model is not None, "Call .build() first."
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(device).train()
        opt = optimizer_ctor(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(True if amp is None else amp))

        for _ in range(epochs):
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None

                x = _to_device(x, device)
                y = _to_device(y, device) if y is not None else None

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = self.model(**_forward_kwargs(x))
                    loss = out["loss"] if isinstance(out, dict) and "loss" in out else _l2_loss(out, y)

                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip))
                scaler.step(opt)
                scaler.update()

        return self

    def fit_arrays(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor, *,
                   batch_size: int = 64, shuffle: bool = True, **fit_kw) -> "Temporal":
        ds = _DictDataset(inputs, targets)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        return self.fit_dataloader(dl, **fit_kw)

    @torch.no_grad()
    def predict_samples(self, inputs: Dict[str, torch.Tensor], *,
                        samples: Optional[int] = None,
                        device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        assert self.model is not None and self.config is not None, "Call .build() first."
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval().to(device)

        sampler = EnsembleSampler(self.model)
        n = int(samples or self.spec.mc_dropout_samples)
        return sampler.generate(
            input_values=_to_device(inputs, device),
            prediction_length=self.config.prediction_length,
            samples=n,
        )

    @torch.no_grad()
    def predict_quantiles(self, inputs: Dict[str, torch.Tensor], q: Optional[Sequence[float]] = None,
                          *, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        samples = self.predict_samples(inputs, samples=self.spec.mc_dropout_samples, device=device)
        qs = torch.tensor(list(q) if q is not None else _get_quantiles(self.config),
                          device=samples.device, dtype=samples.dtype)
        return torch.quantile(samples, qs, dim=0)  # [Q, T, F]

    @torch.no_grad()
    def predict_point(self, inputs: Dict[str, torch.Tensor], agg: str = "median",
                      *, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        if agg == "mean":
            s = self.predict_samples(inputs, device=device)
            return s.mean(dim=0)
        q = self.predict_quantiles(inputs, q=[0.5], device=device)
        return q[0]

    # utils
    def to(self, device: Union[str, torch.device]) -> "Temporal":
        if self.model is not None:
            self.model.to(device)
        return self

    def set_seed(self, seed: int) -> "Temporal":
        import random, numpy as np
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        return self

    def _r(self, **kw) -> "Temporal":
        new = Temporal(self.spec)
        new.__dict__.update(self.__dict__)
        for k, v in kw.items():
            setattr(new, k, v)
        return new


# ---------- compile BuildSpec → TransformerTimeSeriesConfig ----------

def _compile_config(spec: BuildSpec) -> TransformerTimeSeriesConfig:
    # attention config (self-attn for enc/dec)
    attn_type = ATTN_KIND_MAP[spec.attention_kind]
    base_attn = {
        "type": attn_type,
        "num_heads": spec.n_heads,
        "dropout": spec.dropout,
        "bias": spec.attn_bias,
        "qk_layernorm": spec.qk_layernorm,
        **(spec.attention_cfg or {}),
    }
    # RoPE / ALiBi (Full & Patterned need max_position_embeddings)
    if spec.use_rope:
        base_attn.update({"use_rope": True, "rope_base": spec.rope_base, "max_position_embeddings": spec.max_pos})
    if spec.use_alibi:
        base_attn.update({"use_alibi": True, "max_position_embeddings": spec.max_pos})

    # Pattern block (if chosen)
    if spec.attention_kind == "patterned":
        pattern = spec.pattern or {"type": "local", "window_size": 128}
        base_attn["pattern"] = pattern

    # block normalization
    blk_norm = normalization_config_from_dict(spec.block_norm).to_dict()

    # feed-forward
    ffn = {"type": "standard", "hidden_size": 4 * spec.d_model, "dropout": spec.dropout}

    # blocks
    enc_blocks, dec_blocks = [], []
    if spec.arch in {"encoder", "encdec"} and spec.n_encoder_layers > 0:
        for _ in range(spec.n_encoder_layers):
            enc_blocks.append(
                transformer_block_config_from_dict({
                    "type": "encoder",
                    "attention_config": dict(base_attn),
                    "ffn_config": dict(ffn),
                    "normalization_config": dict(blk_norm),
                }).to_dict()
            )

    if spec.arch in {"decoder", "encdec"} and spec.n_decoder_layers > 0:
        for _ in range(spec.n_decoder_layers):
            dec_blocks.append(
                transformer_block_config_from_dict({
                    "type": "decoder",
                    "attention_config": dict(base_attn),
                    # cross-attn: keep simple (no RoPE/ALiBi by default)
                    "cross_attention_config": {"type": "full_attention", "num_heads": spec.n_heads, "dropout": spec.dropout, "bias": spec.attn_bias},
                    "ffn_config": dict(ffn),
                    "normalization_config": dict(blk_norm),
                }).to_dict()
            )

    # embeddings (value + positional)
    ve_cfg = embedding_config_from_dict(dict(spec.value_embedding), feature_size=spec.d_model).to_dict()
    pe_cfg = embedding_config_from_dict(dict(spec.positional_embedding), embedding_dim=spec.d_model).to_dict()

    # layer norm (embedding-level) + optional instance/RevIN
    ln_cfg = normalization_config_from_dict(dict(spec.layer_norm)).to_dict()
    in_cfg = normalization_config_from_dict(dict(spec.instance_norm)).to_dict() if spec.instance_norm else None

    # head / loss / head aggregation
    head_cfg = output_head_config_from_dict(dict(spec.head)).to_dict()
    loss_cfg = loss_config_from_dict(dict(spec.loss)).to_dict()
    agg_cfg = head_aggregation_config_from_dict(dict(spec.head_agg)).to_dict()

    # final model config
    cfg = TransformerTimeSeriesConfig.from_dict({
        "type": "transformer_model",
        "d_model": spec.d_model,
        "num_attention_heads": spec.n_heads,
        "hidden_dropout_prob": spec.dropout,
        "max_position_embeddings": spec.max_pos,
        "prediction_length": spec.prediction_length,
        "context_length": spec.context_length,

        "value_embedding_config": ve_cfg,
        "positional_embedding_config": pe_cfg,

        "layer_norm_config": ln_cfg,
        "instance_norm_config": in_cfg,

        "encoder_blocks": enc_blocks or None,
        "decoder_blocks": dec_blocks or None,

        "output_head_config": head_cfg,
        "loss_config": loss_cfg,
        "head_agg_config": agg_cfg,

        **(spec.extras or {}),
    })
    return cfg


def _get_quantiles(cfg: TransformerTimeSeriesConfig) -> List[float]:
    head = getattr(cfg, "output_head_config", None)
    if isinstance(head, dict):
        qs = head.get("kwargs", {}).get("quantiles") or head.get("quantiles")
        if qs is not None: return list(qs)
    return [0.05, 0.5, 0.95]


class _DictDataset(torch.utils.data.Dataset):
    def __init__(self, X: Dict[str, torch.Tensor], y: torch.Tensor):
        assert all(isinstance(v, torch.Tensor) for v in X.values())
        self.X = X; self.y = y
        self.N = next(iter(X.values())).shape[0]
    def __len__(self): return self.N
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.X.items()}, self.y[i]


# ---------- “quickest path” helper ----------

def forecast(
    inputs: Dict[str, torch.Tensor],
    *,
    horizon: int,
    context: int,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    arch: str = "encoder",
    attention: str = "full",
    rope: bool = True,
    alibi: bool = False,
    mc_samples: int = 100,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 64,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    tm = Temporal() \
        .with_dims(d_model, n_heads) \
        .with_context(context) \
        .with_horizon(horizon) \
        .with_attention(attention)

    if arch == "encoder":
        tm = tm.as_encoder(n_layers)
    elif arch == "decoder":
        tm = tm.as_decoder(n_layers)
    elif arch == "encdec":
        tm = tm.as_encdec(n_layers, max(1, n_layers//2))
    else:
        raise ValueError("arch must be 'encoder'|'decoder'|'encdec'")

    if rope: tm = tm.with_rope()
    if alibi: tm = tm.with_alibi()
    tm = tm.with_mc_samples(mc_samples).head_quantile([0.05, 0.5, 0.95]).build()

    if targets is not None:
        tm.fit_arrays(inputs, targets, epochs=epochs, lr=lr, batch_size=batch_size)

    return tm.predict_quantiles(inputs)
