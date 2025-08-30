import torch
import unittest
from temporal.modules.heads.timeflow_head import (
    TimeFlowTimestepEmbedder,
    TimeFlowResBlock,
    TimeFlowFinalLayer,
    TimeFlowMLPAdaLN,
    TimeFlowHead,
)

class TestTimeFlowHead(unittest.TestCase):

    def test_timeflow_timestep_embedder(self):
        embedder = TimeFlowTimestepEmbedder(emb_dim=128)
        t = torch.randn(2)
        output = embedder(t)
        self.assertEqual(output.shape, (2, 128))

    def test_timeflow_res_block(self):
        block = TimeFlowResBlock(channels=128)
        x = torch.randn(2, 128)
        cond = torch.randn(2, 128)
        output = block(x, cond)
        self.assertEqual(output.shape, (2, 128))

    def test_timeflow_final_layer(self):
        layer = TimeFlowFinalLayer(model_channels=128, out_channels=64)
        x = torch.randn(2, 128)
        cond = torch.randn(2, 128)
        output = layer(x, cond)
        self.assertEqual(output.shape, (2, 64))

    def test_timeflow_mlp_adaln(self):
        mlp = TimeFlowMLPAdaLN(
            in_channels=64,
            model_channels=128,
            out_channels=64,
            cond_channels=32,
            num_blocks=2,
        )
        x = torch.randn(2, 64)
        t = torch.randn(2)
        cond = torch.randn(2, 32)
        output = mlp(x, t, cond)
        self.assertEqual(output.shape, (2, 64))

    def test_timeflow_head(self):
        head = TimeFlowHead(
            target_channels=64,
            cond_channels=32,
            num_blocks=2,
            model_channels=128,
        )
        cond = torch.randn(2, 32)
        targets = torch.randn(2, 64)
        output = head(cond, targets)
        self.assertEqual(output.shape, (2, 64))

    def test_timeflow_head_sample(self):
        head = TimeFlowHead(
            target_channels=64,
            cond_channels=32,
            num_blocks=2,
            model_channels=128,
        )
        cond = torch.randn(2, 32)
        samples = head.sample(cond, num_samples=5)
        self.assertEqual(samples.shape, (2, 5, 64))

    def test_timeflow_head_sample_quantiles(self):
        head = TimeFlowHead(
            target_channels=64,
            cond_channels=32,
            num_blocks=2,
            model_channels=128,
        )
        cond = torch.randn(2, 32)
        quantiles = head.sample_quantiles(cond, quantile_levels=[0.25, 0.5, 0.75])
        self.assertEqual(quantiles.shape, (2, 3, 64))

if __name__ == '__main__':
    unittest.main()
