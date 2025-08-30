import torch
import unittest
from temporal.modules.attentions.patterned_attention import PatternedMultiHeadAttention
from temporal.configs.attention_config import AttentionPatternConfig

class TestPatternedAttention(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 16
        self.num_heads = 4
        self.seq_len = 10
        self.batch_size = 2

    def test_sliding_pattern(self):
        pattern = AttentionPatternConfig(type="sliding", window_size=3)
        attention = PatternedMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            pattern=pattern.to_dict(),
        )
        mask = attention.compute_pattern_mask(self.seq_len, device="cpu")
        self.assertEqual(mask.shape, (self.seq_len, self.seq_len))
        self.assertTrue(torch.all(mask.diagonal()))

    def test_local_pattern(self):
        pattern = AttentionPatternConfig(type="local", window_size=3)
        attention = PatternedMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            pattern=pattern.to_dict(),
        )
        mask = attention.compute_pattern_mask(self.seq_len, device="cpu")
        self.assertEqual(mask.shape, (self.seq_len, self.seq_len))
        self.assertTrue(torch.all(torch.diag(mask, 0)))
        self.assertFalse(torch.any(torch.triu(mask, 1)))

    def test_dilated_pattern(self):
        pattern = AttentionPatternConfig(type="dilated", dilation=2)
        attention = PatternedMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            pattern=pattern.to_dict(),
        )
        mask = attention.compute_pattern_mask(self.seq_len, device="cpu")
        self.assertEqual(mask.shape, (self.seq_len, self.seq_len))
        self.assertTrue(torch.all(mask[0, ::2]))

    def test_forward_pass(self):
        pattern = AttentionPatternConfig(type="sliding", window_size=3)
        attention = PatternedMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            pattern=pattern.to_dict(),
        )
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output, _, _ = attention(hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

if __name__ == "__main__":
    unittest.main()
