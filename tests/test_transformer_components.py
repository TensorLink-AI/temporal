
import pytest
import torch
from temporal.models.builder import build_time_series_transformer
from temporal.configs.transformer_config import TransformerTimeSeriesConfig

@pytest.fixture
def basic_config():
    return TransformerTimeSeriesConfig(
        feature_size=1,
        d_model=16,
        num_heads=2,
        context_length=10,
        prediction_length=5,
        architecture={"layout": "encoder-decoder"},
        loss_config={"type": "mse"}
    )

def test_encoder_only_forward_pass(basic_config):
    basic_config.architecture = {"layout": "encoder-only"}
    model = build_time_series_transformer(basic_config)
    
    past_values = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    
    output = model(past_values=past_values)
    
    assert output.logits.shape == (2, basic_config.context_length, basic_config.feature_size)

def test_decoder_only_forward_pass(basic_config):
    basic_config.architecture = {"layout": "decoder-only"}
    model = build_time_series_transformer(basic_config)
    
    future_values = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    
    output = model(future_values=future_values)
    
    assert output.logits.shape == (2, basic_config.prediction_length, basic_config.feature_size)

def test_encoder_decoder_forward_pass(basic_config):
    model = build_time_series_transformer(basic_config)
    
    past_values = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    future_values = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    
    output = model(past_values=past_values, future_values=future_values)
    
    assert output.logits.shape == (2, basic_config.prediction_length, basic_config.feature_size)

def test_multi_feature_forward_pass(basic_config):
    basic_config.feature_size = 3
    model = build_time_series_transformer(basic_config)
    
    past_values = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    future_values = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    
    output = model(past_values=past_values, future_values=future_values)
    
    assert output.logits.shape == (2, basic_config.prediction_length, basic_config.feature_size)

def test_different_context_prediction_lengths(basic_config):
    basic_config.context_length = 20
    basic_config.prediction_length = 10
    model = build_time_series_transformer(basic_config)
    
    past_values = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    future_values = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    
    output = model(past_values=past_values, future_values=future_values)
    
    assert output.logits.shape == (2, basic_config.prediction_length, basic_config.feature_size)

def test_backward_pass(basic_config):
    model = build_time_series_transformer(basic_config)
    
    past_values = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    future_values = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    labels = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    
    output = model(past_values=past_values, future_values=future_values, labels=labels)
    
    output.loss.backward()
    
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

def test_reproducibility(basic_config):
    torch.manual_seed(0)
    model1 = build_time_series_transformer(basic_config)
    past_values = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    future_values = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    output1 = model1(past_values=past_values, future_values=future_values)

    torch.manual_seed(0)
    model2 = build_time_series_transformer(basic_config)
    output2 = model2(past_values=past_values, future_values=future_values)

    assert torch.allclose(output1.logits, output2.logits)

def test_inference_mode(basic_config):
    model = build_time_series_transformer(basic_config)
    model.eval()
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert not module.training

def test_batch_invariance(basic_config):
    model = build_time_series_transformer(basic_config)
    
    past_values_batch1 = torch.randn(1, basic_config.context_length, basic_config.feature_size)
    future_values_batch1 = torch.randn(1, basic_config.prediction_length, basic_config.feature_size)
    
    past_values_batch4 = torch.randn(4, basic_config.context_length, basic_config.feature_size)
    future_values_batch4 = torch.randn(4, basic_config.prediction_length, basic_config.feature_size)
    
    try:
        model(past_values=past_values_batch1, future_values=future_values_batch1)
        model(past_values=past_values_batch4, future_values=future_values_batch4)
    except Exception as e:
        pytest.fail(f"Model failed with different batch sizes: {e}")

def test_data_types(basic_config):
    model = build_time_series_transformer(basic_config)
    
    past_values = torch.randn(2, basic_config.context_length, basic_config.feature_size)
    future_values = torch.randn(2, basic_config.prediction_length, basic_config.feature_size)
    
    try:
        model(past_values=past_values.to(torch.float16), future_values=future_values.to(torch.float16))
        model(past_values=past_values.to(torch.float32), future_values=future_values.to(torch.float32))
    except Exception as e:
        pytest.fail(f"Model failed with different data types: {e}")
