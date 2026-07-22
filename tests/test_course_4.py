import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.course_4_attention import scaled_dot_product_attention, softmax


class TestCourse4Attention(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.d_k = 8
        self.d_v = 8

        np.random.seed(42)
        self.Q = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        self.K = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        self.V = np.random.randn(self.batch_size, self.seq_len, self.d_v)

    def test_softmax_sums_to_one(self):
        """Softmax probabilities must sum to 1 along the specified axis."""
        prob = softmax(self.Q, axis=-1)
        sums = np.sum(prob, axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5)

    def test_attention_shapes(self):
        """Attention output should match (batch, seq_len, d_v) and weights match (batch, seq_len, seq_len)."""
        output, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_v))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))

    def test_attention_weights_valid_probability_distribution(self):
        """Attention weights must sum to 1 for each query position."""
        _, weights = scaled_dot_product_attention(self.Q, self.K, self.V)
        weight_sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(weight_sums, np.ones_like(weight_sums), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
