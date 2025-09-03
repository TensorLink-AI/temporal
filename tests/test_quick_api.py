# tests/test_quick_api.py
import math
import types
import pytest
import torch

from temporal.quick import Temporal

# -----------------------
# Stubs via monkeypatch
# -----------------------

class _DummyModel(torch.nn.Module):
    def __init__(self, horizon: int, feat_out: int = 1):
        super().__init__()
        self._horizon = int(horizon)
        self._feat_out = int(feat_out)

    def forward(self, **kwargs):
        """
        Return zeros shaped like targets: [B, H, F]. We infer B,F from inputs.
        """
        x = kwargs.get("input_values")
        assert isinstance(x, torch.Tensor), "Expected input_values tensor in forward()"
        B, _, F = x.shape
        return torch.zeros(B, self._horizon, self._feat_out if self._feat_out else F, device=x.device, dtype=x.dtype)


@pytest.fixture(autouse=True)
def patch_builder_and_sampler(monkeypatch):
    """
    Patch:
      - temporal.models.build_time_series_transformer → returns _DummyModel with horizon from config
      - temporal.utils.ensemble.EnsembleSampler.generate → returns normal samples [N, T, F]
    """
    import temporal.models
    import temporal.utils.ensemble

    def fake_builder(cfg):
        # cfg is a TransformerTimeSeriesConfig; we can read prediction_length
        H = getattr(cfg, "prediction_length", None) or cfg.to_dict()["prediction_length"]
        return _DummyModel(horizon=H, feat_out=1)

    class _FakeSampler:
        def __init__(self, model): self.model = model
        def generate(self, *, input_values, prediction_length, samples):
            B, _, F = input_values.shape
            # Return [samples, T, F]
            return torch.randn(samples, prediction_length, F, device=input_values.device, dtype=input_values.dtype)

    monkeypatch.setattr(temporal.models, "build_time_series_transformer", fake_builder, raising=True)
    monkeypatch.setattr(temporal.utils.ensemble, "EnsembleSampler", _FakeSampler, raising=True)


# -----------------------
# Utilities
# -----------------------

def _cfg_dict(tm: Temporal):
    assert tm.config is not None
    # config is a dataclass with to_dict()
    return tm.config.to_dict()


def _attn_cfgs(cfg_dict):
    enc = cfg_dict.get("encoder_blocks") or []
    dec = cfg_dict.get("decoder_blocks") or []
    out = []
    for blk in enc + dec:
        a = blk.get("attention_config")
        if a: out.append(a)
    return out


# -----------------------
# Tests: attention flags
# -----------------------

@pytest.mark.parametrize("kind, expected_type", [
    ("full", "full_attention"),
    ("patterned", "patterned_attention"),
    ("lse", "lse_attention"),
    ("hybrid", "hybrid_attention"),
    ("diff", "diffwist_attention"),
    ("diffwist", "diffwist_attention"),
])
def test_attention_kinds_mapping(kind, expected_type):
    tm = (Temporal()
          .as_encoder(2)
          .with_dims(256, 8)
          .with_context(64).with_horizon(16)
          .with_attention(kind)
          .build())
    cd = _cfg_dict(tm)
    attn_types = [a["type"] for a in _attn_cfgs(cd)]
    assert all(t == expected_type for t in attn_types)

def test_rope_flag_and_validation_even_dhead():
    # Valid: d_model=512, n_heads=8 -> d_head=64 (even)
    tm = (Temporal()
          .as_encoder(2)
          .with_dims(512, 8)
          .with_context(64).with_horizon(16)
          .with_rope()
          .build())
    cd = _cfg_dict(tm)
    for a in _attn_cfgs(cd):
        assert a.get("use_rope") is True
        assert a.get("rope_base") == 10000
        assert a.get("max_position_embeddings") == cd["max_position_embeddings"]

    # Invalid: RoPE with odd d_head should raise
    with pytest.raises(ValueError):
        (Temporal()
         .as_encoder(2)
         .with_dims(320, 5)   # d_head=64 (even) -> OK
         .with_context(64).with_horizon(16)
         .with_rope()
         .build())

    with pytest.raises(ValueError):
        (Temporal()
         .as_encoder(2)
         .with_dims(300, 7)   # not divisible
         .with_context(64).with_horizon(16)
         .with_rope()
         .build())

def test_alibi_and_qk_layernorm_and_bias():
    tm = (Temporal()
          .as_decoder(3)
          .with_dims(256, 8)
          .with_context(64).with_horizon(32)
          .with_attention_bias(False)
          .with_qk_layernorm(True)
          .with_alibi(True)
          .build())
    cd = _cfg_dict(tm)
    for a in _attn_cfgs(cd):
        assert a.get("bias") is False
        assert a.get("qk_layernorm") is True
        assert a.get("use_alibi") is True
        assert a.get("max_position_embeddings") == cd["max_position_embeddings"]

def test_patterned_attention_with_local_pattern():
    tm = (Temporal()
          .as_encoder(2)
          .with_dims(256, 8)
          .with_context(128).with_horizon(16)
          .with_attention("patterned")
          .with_attention_pattern(kind="local", window_size=64, stride=32)
          .build())
    cd = _cfg_dict(tm)
    a = _attn_cfgs(cd)[0]
    assert a["type"] == "patterned_attention"
    assert "pattern" in a
    assert a["pattern"]["type"] == "local"
    assert a["pattern"]["window_size"] == 64
    assert a["pattern"]["stride"] == 32


# -----------------------
# Tests: embeddings & norms
# -----------------------

def test_value_embedding_patch_and_positional_choice():
    tm = (Temporal()
          .as_encoder(1)
          .with_dims(512, 8)
          .with_context(64).with_horizon(16)
          .value_embedding("patch", patch_size=4, feature_size=512, stride=2, use_mlp=True, mlp_hidden_size=1024)
          .positional_embedding("sinusoidal")
          .build())
    cd = _cfg_dict(tm)
    ve = cd["value_embedding_config"]
    pe = cd["positional_embedding_config"]
    assert ve["type"] == "patch"
    assert ve["kwargs"]["patch_size"] == 4
    assert ve["kwargs"]["stride"] == 2
    assert ve["kwargs"]["use_mlp"] is True
    assert ve["kwargs"]["mlp_hidden_size"] == 1024
    assert pe["type"] == "sinusoidal"

@pytest.mark.parametrize("kind", ["layer", "rms", "scale"])
def test_model_layer_norm_variants(kind):
    tm = (Temporal()
          .as_encoder(1)
          .with_dims(256, 8)
          .with_context(64).with_horizon(16)
          .with_layer_norm(kind=kind, eps=1e-6, elementwise_affine=False)
          .build())
    cd = _cfg_dict(tm)
    ln = cd["layer_norm_config"]
    assert ln["type"] == kind
    assert ln["kwargs"]["eps"] == pytest.approx(1e-6)
    assert ln["kwargs"]["elementwise_affine"] is False

def test_instance_norm_revin_and_block_norm_layer():
    tm = (Temporal()
          .as_encoder(1)
          .with_dims(512, 8)
          .with_context(64).with_horizon(16)
          .with_instance_norm(type="revin", num_features=512, affine=True, subtract_last=False)
          .with_block_norm(kind="layer", eps=1e-5)
          .build())
    cd = _cfg_dict(tm)
    rin = cd["instance_norm_config"]
    bn = None
    # Check one encoder block for normalization_config
    enc = cd["encoder_blocks"][0]
    bn = enc["normalization_config"]
    assert rin["type"] == "revin"
    assert rin["kwargs"]["num_features"] == 512
    assert bn["type"] == "layer"
    assert bn["kwargs"]["eps"] == pytest.approx(1e-5)


# -----------------------
# Tests: heads / loss / aggregation
# -----------------------

def test_quantile_head_defaults_to_crps():
    tm = (Temporal()
          .as_encoder(1)
          .with_dims(256, 8)
          .with_context(64).with_horizon(16)
          .head_quantile([0.1, 0.5, 0.9])
          .build())
    cd = _cfg_dict(tm)
    head = cd["output_head_config"]
    loss = cd["loss_config"]
    assert head["type"] == "quantile"
    assert head["kwargs"]["quantiles"] == [0.1, 0.5, 0.9]
    assert loss["type"] == "crps"

def test_mixture_head_sets_nll_and_aggregator():
    tm = (Temporal()
          .as_encoder(1)
          .with_dims(256, 8)
          .with_context(64).with_horizon(16)
          .head_mixture(kind="student_t", components=3)
          .head_aggregator(kind="gated", hidden_dim=64)
          .build())
    cd = _cfg_dict(tm)
    head = cd["output_head_config"]
    loss = cd["loss_config"]
    agg  = cd["head_agg_config"]
    assert head["type"] == "mixture"
    assert head["kwargs"]["kind"] == "student_t"
    assert head["kwargs"]["components"] == 3
    assert loss["type"] == "nll"
    assert loss["kwargs"]["distribution_type"] == "student_t"
    assert agg["type"] == "gated"
    assert agg["kwargs"]["hidden_dim"] == 64


# -----------------------
# Tests: fit & predict (end-to-end with stubs)
# -----------------------

@pytest.mark.parametrize("arch", ["encoder", "decoder", "encdec"])
def test_fit_and_predict_shapes(arch):
    B, L, H, F = 8, 48, 24, 1
    x = torch.randn(B, L, F)
    y = torch.randn(B, H, F)

    tm = Temporal().with_dims(256, 8).with_context(L).with_horizon(H).head_quantile([0.05, 0.5, 0.95])
    if arch == "encoder":
        tm = tm.as_encoder(2)
    elif arch == "decoder":
        tm = tm.as_decoder(2)
    else:
        tm = tm.as_encdec(2, 1)

    tm = tm.build()
    tm.fit_arrays({"input_values": x}, y, epochs=1, lr=1e-3, batch_size=4)
    q = tm.predict_quantiles({"input_values": x})
    assert q.shape == (3, H, F)  # [Q, T, F]

def test_attention_bias_default_true():
    tm = (Temporal()
          .as_encoder(1)
          .with_dims(256, 8)
          .with_context(32).with_horizon(8)
          .build())
    cd = _cfg_dict(tm)
    for a in _attn_cfgs(cd):
        assert a.get("bias") is True
