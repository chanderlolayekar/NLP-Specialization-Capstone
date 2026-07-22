import unittest
import numpy as np
from src.course_3_sequence import build_ner_model, build_siamese_network


class TestCourse3Sequence(unittest.TestCase):

    def test_lstm_ner_shape(self):
        """Test NER model output tensor shape."""
        model = build_ner_model(vocab_size=100, num_tags=5, embedding_dim=16, lstm_units=32)
        dummy_input = np.random.randint(0, 100, size=(2, 10))  # Batch size 2, seq len 10
        output = model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (2, 10, 5))

    def test_siamese_network_similarity_shape(self):
        """Test Siamese Network output shape."""
        model = build_siamese_network(vocab_size=100, embedding_dim=16, lstm_units=32)
        sent1 = np.random.randint(0, 100, size=(2, 8))
        sent2 = np.random.randint(0, 100, size=(2, 8))
        output = model.predict([sent1, sent2], verbose=0)
        self.assertEqual(output.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
