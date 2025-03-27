import torch
from temporal.models import TimeSeriesTransformerModel, TimeSeriesConfig

def test_forward_pass_shapes():
    config = TimeSeriesConfig(
        feature_size=1,
        context_length=32,
        prediction_length=8,
        output_token_lengths=3,
        head_aggregation_method="mean",
        loss_type="mse"
    )
    model = TimeSeriesTransformerModel(config)

    batch_size = 4
    input_ids = torch.randn(batch_size, config.context_length)
    decoder_input_ids = torch.randn(batch_size, 1)
    labels = torch.randn(batch_size, config.prediction_length, config.num_quantiles)

    output = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
        return_dict=True
    )

    assert output.last_hidden_state.shape == (batch_size, config.prediction_length, config.num_quantiles)
    assert output.loss is not None
