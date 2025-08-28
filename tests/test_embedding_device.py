import torch
import pytest
from temporal.modules.embedders.embedding import NoneEmbedding

@pytest.mark.parametrize("d_model", [16, 32])
def test_none_embedding_device_placement(d_model):
    """
    Test that NoneEmbedding correctly places its output tensor on the expected device.
    """
    embedding = NoneEmbedding(d_model=d_model)

    # Test on CPU
    output_cpu = embedding(batch_size=2, seq_len=10)
    assert output_cpu.device == torch.device("cpu")
    assert output_cpu.shape == (2, 10, d_model)
    assert torch.all(output_cpu == 0.0)

    if torch.cuda.is_available():
        # Test on GPU
        device = torch.device("cuda:0")
        embedding.to(device)
        output_gpu = embedding(batch_size=2, seq_len=10)
        assert output_gpu.device == device
        assert output_gpu.shape == (2, 10, d_model)
        assert torch.all(output_gpu == 0.0)
    else:
        print("CUDA not available, skipping GPU device placement test.")

