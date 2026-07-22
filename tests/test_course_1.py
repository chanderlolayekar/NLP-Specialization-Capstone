import sys
import os
import unittest
import numpy as np

# Ensure Python can import modules from the src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.course_1_classification import cosine_similarity, complete_analogy


class TestCourse1Classification(unittest.TestCase):

    def setUp(self):
        """Set up mock embeddings for geometric testing."""
        # Mock 3D embeddings: [Is_Royal, Is_Female, Is_Adult]
        self.mock_embeddings = {
            'king':  np.array([1.0, 0.0, 1.0]),
            'man':   np.array([0.0, 0.0, 1.0]),
            'queen': np.array([1.0, 1.0, 1.0]),
            'woman': np.array([0.0, 1.0, 1.0]),
            'apple': np.array([0.0, 0.0, 0.0])
        }

    def test_cosine_similarity_identical(self):
        """Identical vectors should have a cosine similarity of 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        """Perpendicular vectors should have a cosine similarity of 0.0."""
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        sim = cosine_similarity(u, v)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_complete_analogy_king_man_queen(self):
        """Test classic analogy: king : man :: queen : ? -> woman"""
        predicted_word, similarity = complete_analogy(
            'king', 'man', 'queen', self.mock_embeddings
        )
        self.assertEqual(predicted_word, 'woman')
        self.assertGreater(similarity, 0.8)

    def test_complete_analogy_missing_word(self):
        """Test that passing an out-of-vocabulary word raises a ValueError."""
        with self.assertRaises(ValueError):
            complete_analogy('king', 'man', 'unknown_word', self.mock_embeddings)


if __name__ == '__main__':
    unittest.main()
