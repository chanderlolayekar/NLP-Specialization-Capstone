import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.course_1_classification import (
    cosine_similarity, 
    complete_analogy, 
    align_word_vectors, 
    translate_word
)


class TestCourse1Classification(unittest.TestCase):

    def setUp(self):
        self.mock_embeddings = {
            'king':  np.array([1.0, 0.0, 1.0]),
            'man':   np.array([0.0, 0.0, 1.0]),
            'queen': np.array([1.0, 1.0, 1.0]),
            'woman': np.array([0.0, 1.0, 1.0]),
            'apple': np.array([0.0, 0.0, 0.0])
        }

    def test_complete_analogy_king_man_queen(self):
        predicted_word, similarity = complete_analogy(
            'king', 'man', 'queen', self.mock_embeddings
        )
        self.assertEqual(predicted_word, 'woman')
        self.assertGreater(similarity, 0.8)

    def test_word_translation(self):
        en_emb = {'cat': np.array([1.0, 0.0]), 'dog': np.array([0.0, 1.0])}
        es_emb = {'gato': np.array([0.0, 1.0]), 'perro': np.array([-1.0, 0.0])}

        X = np.vstack([en_emb['cat'], en_emb['dog']])
        Y = np.vstack([es_emb['gato'], es_emb['perro']])

        R = align_word_vectors(X, Y, learning_rate=0.1, num_iters=500)
        translated = translate_word('cat', en_emb, es_emb, R)
        self.assertEqual(translated, 'gato')


if __name__ == '__main__':
    unittest.main()
