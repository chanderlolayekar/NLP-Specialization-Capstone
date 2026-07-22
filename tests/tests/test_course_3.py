import sys
import os
import unittest
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.course_3_sequence import build_ner_model, build_siamese_network, SequenceModels


class TestCourse3Sequence(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 100
        self.num_tags = 4
        self.max_len = 10
        self.num_samples = 8

        self.X_dummy = np.random.randint(1, self.vocab_size, size=(self.num_samples, self.max_len))
        self.y_dummy = np.random.randint(0, self.num_tags, size=(self.num_samples, self.max_len))

    def test_lstm_ner_shape(self):
        model = build_ner_model(
            vocab_size=self.vocab_size, 
            num_tags=self.num_tags, 
            max_len=self.max_len,
            use_gru=False
        )
        predictions = model.predict(self.X_dummy, verbose=0)
        self.assertEqual(predictions.shape, (self.num_samples, self.max_len, self.num_tags))

    def test_siamese_network_similarity_shape(self):
        """Verify Siamese Network outputs a similarity score of shape (batch_size, 1) bounded between -1 and 1."""
        model = build_siamese_network(vocab_size=self.vocab_size, max_len=self.max_len)
        
        X1_dummy = np.random.randint(1, self.vocab_size, size=(self.num_samples, self.max_len))
        X2_dummy = np.random.randint(1, self.vocab_size, size=(self.num_samples, self.max_len))
        
        scores = model.predict([X1_dummy, X2_dummy], verbose=0)
        self.assertEqual(scores.shape, (self.num_samples, 1))
        self.assertTrue(np.all(scores >= -1.0) and np.all(scores <= 1.0))


if __name__ == '__main__':
    unittest.main()
