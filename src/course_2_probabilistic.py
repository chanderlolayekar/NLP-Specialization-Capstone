import numpy as np


def min_edit_distance(source: str, target: str, ins_cost: int = 1, del_cost: int = 1, rep_cost: int = 2) -> int:
    """
    Computes the Minimum Edit Distance between source and target strings 
    using Dynamic Programming (Levenshtein Distance variant).
    """
    m, n = len(source), len(target)
    # Initialize DP table (m+1 x n+1) with zeros
    D = np.zeros((m + 1, n + 1), dtype=int)

    # Base cases: empty source or empty target
    for i in range(1, m + 1):
        D[i][0] = D[i - 1][0] + del_cost
    for j in range(1, n + 1):
        D[0][j] = D[0][j - 1] + ins_cost

    # Fill DP table row by row
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_del = D[i - 1][j] + del_cost
            cost_ins = D[i][j - 1] + ins_cost
            # If characters match, replace cost is 0; otherwise rep_cost
            cost_rep = D[i - 1][j - 1] + (0 if source[i - 1] == target[j - 1] else rep_cost)
            
            D[i][j] = min(cost_del, cost_ins, cost_rep)

    return int(D[m][n])


def autocorrect(word: str, vocabulary: list[str], max_distance: int = 2) -> list[tuple[str, int]]:
    """
    Finds words in vocabulary with an edit distance <= max_distance from the misspelled word.
    Returns list of (candidate_word, edit_distance) sorted by closest match.
    """
    candidates = []
    for vocab_word in vocabulary:
        dist = min_edit_distance(word, vocab_word)
        if dist <= max_distance:
            candidates.append((vocab_word, dist))
            
    # Sort candidates by distance (ascending)
    return sorted(candidates, key=lambda x: x[1])


def viterbi_pos_tagger(
    words: list[str], 
    tags: list[str], 
    initial_prob: np.ndarray, 
    transition_matrix: np.ndarray, 
    emission_matrix: dict[str, dict[str, float]]
) -> list[str]:
    """
    Executes the Viterbi Algorithm to find the optimal POS tags for a sentence.
    
    Parameters:
        words: List of words in sentence (length T)
        tags: List of unique POS tags (length K)
        initial_prob: Array of starting probabilities for each tag (shape: K,)
        transition_matrix: Array where [i, j] is P(tag_j | tag_i) (shape: K x K)
        emission_matrix: Dict mapping tag -> {word: P(word | tag)}
    Returns:
        best_path_tags: Most likely sequence of POS tags for words
    """
    num_words = len(words)
    num_tags = len(tags)
    
    # viterbi_matrix[k, t] stores max prob of observing words[:t+1] ending in state tags[k]
    viterbi_matrix = np.zeros((num_tags, num_words))
    # backpointer[k, t] stores index of predecessor state that maximized probability
    backpointer = np.zeros((num_tags, num_words), dtype=int)

    # 1. Initialization Step
    for k in range(num_tags):
        tag = tags[k]
        word = words[0]
        emission_p = emission_matrix.get(tag, {}).get(word, 1e-6) # Smooth unseen words
        viterbi_matrix[k, 0] = initial_prob[k] * emission_p
        backpointer[k, 0] = 0

    # 2. Forward Step (Dynamic Programming)
    for t in range(1, num_words):
        word = words[t]
        for j in range(num_tags):
            tag_j = tags[j]
            emission_p = emission_matrix.get(tag_j, {}).get(word, 1e-6)
            
            # Vectorized computation across all possible predecessor states i
            prev_probs = viterbi_matrix[:, t - 1] * transition_matrix[:, j] * emission_p
            best_prev_state = np.argmax(prev_probs)
            
            viterbi_matrix[j, t] = prev_probs[best_prev_state]
            backpointer[j, t] = best_prev_state

    # 3. Backtracking Step
    best_last_state = np.argmax(viterbi_matrix[:, num_words - 1])
    best_path = [best_last_state]

    for t in range(num_words - 1, 0, -1):
        best_last_state = backpointer[best_last_state, t]
        best_path.insert(0, best_last_state)

    # Map state indices back to tag strings
    return [tags[idx] for idx in best_path]
