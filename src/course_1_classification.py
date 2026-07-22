import numpy as np

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes the cosine similarity between two 1D NumPy arrays.
    Formula: (u . v) / (||u|| * ||v||)
    """
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # Small epsilon addition prevents division by zero if a zero vector is passed
    return dot_product / ((norm_u * norm_v) + 1e-9)


def complete_analogy(
    word_a: str, 
    word_b: str, 
    word_c: str, 
    embeddings: dict[str, np.ndarray]
) -> tuple[str, float]:
    """
    Solves word analogies of the form: word_a : word_b :: word_c : ?
    Example: 'king' : 'man' :: 'queen' : 'woman'
    
    Parameters:
        word_a, word_b, word_c: Strings representing input words
        embeddings: Dictionary mapping words (str) to 1D vectors (np.ndarray)
        
    Returns:
        best_word: The predicted word completing the analogy
        best_similarity: The highest cosine similarity score
    """
    # 1. Ensure all input words exist in our vocabulary embeddings
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    for w in [word_a, word_b, word_c]:
        if w not in embeddings:
            raise ValueError(f"Word '{w}' not found in the embedding vocabulary.")

    # 2. Extract vectors for A, B, and C
    vec_a = embeddings[word_a]
    vec_b = embeddings[word_b]
    vec_c = embeddings[word_c]

    # 3. Compute target vector: B - A + C
    target_vec = vec_b - vec_a + vec_c

    # 4. Search vocabulary for the closest word (highest cosine similarity)
    best_word = None
    best_similarity = -1.0 # Cosine similarity ranges from -1.0 to 1.0
    input_words = {word_a, word_b, word_c}

    for word, vec in embeddings.items():
        # Do not allow the model to cheat by returning one of the input words!
        if word in input_words:
            continue
            
        sim = cosine_similarity(target_vec, vec)
        
        if sim > best_similarity:
            best_similarity = sim
            best_word = word

    return best_word, float(best_similarity)
