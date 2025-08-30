import torch
import unittest
from temporal.modules.attentions.base_attention import (
    BaseMultiHeadAttention,
    FullAttention,
    FlashAttention,
)

class TestBaseAttention(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 16
        self.num_heads = 4
        self.seq_len = 10
        self.batch_size = 2

    def test_base_multi_head_attention(self):
        attention = BaseMultiHeadAttention(self.embed_dim, self.num_heads)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output, _, _ = attention(hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_full_attention_with_rope(self):
        attention = FullAttention(self.embed_dim, self.num_heads, use_rope=True)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output, _, _ = attention(hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_full_attention_with_alibi(self):
        attention = FullAttention(self.embed_dim, self.num_heads, use_alibi=True)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        output, _, _ = attention(hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_flash_attention(self):
        try:
            attention = FlashAttention(self.embed_dim, self.num_heads)
            hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
            output, _, _ = attention(hidden_states)
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        except ImportError:
            self.skipTest("FlashAttention not available")

if __name__ == '__main__':
    unittest.main()
