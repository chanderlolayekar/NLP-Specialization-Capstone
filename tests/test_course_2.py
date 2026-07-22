import unittest
import numpy as np
from src.course_2_probabilistic import (
    min_edit_distance,
    autocorrect,
    viterbi_pos_tagger
)


class TestCourse2Probabilistic(unittest.TestCase):

    def test_min_edit_distance_identical(self):
        """Identical strings should have an edit distance of 0."""
        self.assertEqual(min_edit_distance("cat", "cat"), 0)
        self.assertEqual(min_edit_distance("python", "python"), 0)

    def test_min_edit_distance_operations(self):
        """Test edit distance for basic substitutions, insertions, and deletions."""
        # "cat" -> "hats": 1 sub ('c' -> 'h'), 1 ins ('s') = 2
        self.assertEqual(min_edit_distance("cat", "hats"), 2)
        # "play" -> "stay": 2 subs ('p' -> 's', 'l' -> 't') = 2
        self.assertEqual(min_edit_distance("play", "stay"), 2)

    def test_autocorrect_candidates(self):
        """Test autocorrect suggestions for misspelled word."""
        vocab = ["cat", "hat", "bat", "cats", "dog", "elephant"]
        # Misspelled "cta" -> "cat" (2 edits), "cats" (2 edits), "hat" (2 edits), "bat" (2 edits)
        suggestions = autocorrect("cta", vocab, max_distance=2)
        candidate_words = [word for word, dist in suggestions]

        self.assertIn("cat", candidate_words)
        self.assertNotIn("elephant", candidate_words)

    def test_viterbi_pos_tagger(self):
        """Test POS tagger with simplified HMM transition and emission matrices."""
        words = ["the", "cat", "runs"]
        tags = ["DET", "NOUN", "VERB"]

        # Starting probabilities: 100% chance to start with DET
        initial_prob = np.array([1.0, 0.0, 0.0])

        # Transition matrix: DET -> NOUN (1.0), NOUN -> VERB (1.0), VERB -> END (1.0)
        transition_matrix = np.array([
            [0.0, 1.0, 0.0],  # From DET
            [0.0, 0.0, 1.0],  # From NOUN
            [0.0, 0.0, 0.0]   # From VERB
        ])

        emission_matrix = {
            "DET": {"the": 1.0},
            "NOUN": {"cat": 1.0},
            "VERB": {"runs": 1.0}
        }

        predicted_tags = viterbi_pos_tagger(
            words, tags, initial_prob, transition_matrix, emission_matrix
        )

        self.assertEqual(predicted_tags, ["DET", "NOUN", "VERB"])


if __name__ == "__main__":
    unittest.main()
