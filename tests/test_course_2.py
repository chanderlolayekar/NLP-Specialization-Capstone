import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.course_2_probabilistic import (
    min_edit_distance, 
    autocorrect, 
    viterbi_pos_tagger
)


class TestCourse2Probabilistic(unittest.TestCase):

    def test_min_edit_distance_identical(self):
        """Identical strings should have an edit distance of 0."""
        self.assertEqual(min_edit_distance("cat", "cat"), 0)

    def test_min_edit_distance_operations(self):
        """Test edit distance for basic substitutions and insertions."""
        # 'play' -> 'stay' (substitution 'p'->'s' = cost 2)
        self.assertEqual(min_edit_distance("play", "stay"), 2)
        # 'cat' -> 'hats' (substitution 'c'->'h' = 2, insertion 's' = 1 -> total 3)
        self.assertEqual(min_edit_distance("cat", "hats"), 3)

    def test_autocorrect_candidates(self):
        """Test autocorrect suggestions for misspelled word."""
        vocab = ["apple", "apply", "ape", "banana", "cat"]
        suggestions = autocorrect("appl", vocab, max_distance=2)
        
        suggested_words = [word for word, dist in suggestions]
        self.assertIn("apple", suggested_words)
        self.assertIn("apply", suggested_words)
        self.assertNotIn("banana", suggested_words)

    def test_viterbi_pos_tagger(self):
        """Test POS tagger with simplified HMM transition and emission matrices."""
        words = ["the", "cat", "runs"]
        tags = ["NOUN", "VERB", "DET"] # Indices: 0=NOUN, 1=VERB, 2=DET

        # Initial probabilities: Sentence starts with DET
        initial_prob = np.array([0.1, 0.1, 0.8])

        # Transitions: DET -> NOUN -> VERB
        transition_matrix = np.array([
            [0.1, 0.8, 0.1],  # NOUN -> [NOUN, VERB, DET]
            [0.2, 0.1, 0.7],  # VERB -> [NOUN, VERB, DET]
            [0.9, 0.05, 0.05] # DET  -> [NOUN, VERB, DET]
        ])

        # Emission probabilities: P(word | tag)
        emission_matrix = {
            "NOUN": {"cat": 0.8, "runs": 0.1},
            "VERB": {"runs": 0.8, "cat": 0.1},
            "DET":  {"the": 0.95}
        }

        predicted_tags = viterbi_pos_tagger(words, tags, initial_prob, transition_matrix, emission_matrix)
        self.assertEqual(predicted_tags, ["DET", "NOUN", "VERB"])


if __name__ == '__main__':
    unittest.main()
