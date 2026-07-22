import sys
import os
import unittest
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.course_3_sequence import build_ner_model, SequenceModels


class TestCourse3Sequence(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 100
        self.num_tags = 4  # e.g., 0: O, 1: B-PER, 2: B-LOC, 3: B-ORG
        self.max_len = 10
        self.num_samples = 8

        # Dummy dataset: tokenized sequence batches
        self.X_dummy = np.random.randint(1, self.vocab_size, size=(self.num_samples, self.max_len))
        self.y_dummy = np.random.randint(0, self.num_tags, size=(self.num_samples, self.max_len))

    def test_lstm_ner_shape(self):
        """Verify Bi-LSTM NER model output shape equals (batch_size, sequence_length, num_tags)."""
        model = build_ner_model(
            vocab_size=self.vocab_size, 
            num_tags=self.num_tags, 
            max_len=self.max_len,
            use_gru=False
        )
        predictions = model.predict(self.X_dummy, verbose=0)
        self.assertEqual(predictions.shape, (self.num_samples, self.max_len, self.num_tags))

    def test_gru_ner_training(self):
        """Verify Bi-GRU model can run a single training step without raising errors."""
        model = build_ner_model(
            vocab_size=self.vocab_size, 
            num_tags=self.num_tags, 
            max_len=self.max_len,
            use_gru=True
        )
        history = SequenceModels.train_ner_step(model, self.X_dummy, self.y_dummy, epochs=1)
        self.assertIn("loss", history.history)


if __name__ == '__main__':
    unittest.main()
