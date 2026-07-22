import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Computes the cosine similarity between two 1D NumPy arrays."""
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return float(dot_product / ((norm_u * norm_v) + 1e-9))


def complete_analogy(
    word_a: str, 
    word_b: str, 
    word_c: str, 
    embeddings: dict[str, np.ndarray]
) -> tuple[str, float]:
    """Solves word analogies of the form: word_a : word_b :: word_c : ?"""
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    for w in [word_a, word_b, word_c]:
        if w not in embeddings:
            raise ValueError(f"Word '{w}' not found in vocabulary.")

    vec_a, vec_b, vec_c = embeddings[word_a], embeddings[word_b], embeddings[word_c]
    target_vec = vec_b - vec_a + vec_c

    best_word = None
    best_similarity = -1.0
    input_words = {word_a, word_b, word_c}

    for word, vec in embeddings.items():
        if word in input_words:
            continue
        sim = cosine_similarity(target_vec, vec)
        if sim > best_similarity:
            best_similarity = sim
            best_word = word

    return best_word, float(best_similarity)


def align_word_vectors(
    X: np.ndarray, 
    Y: np.ndarray, 
    learning_rate: float = 0.01, 
    num_iters: int = 400
) -> np.ndarray:
    """Learns transformation matrix R mapping source embeddings X to target Y."""
    dim = X.shape[1]
    np.random.seed(42)
    R = np.random.rand(dim, dim)
    num_samples = X.shape[0]

    for _ in range(num_iters):
        gradient = (2 / num_samples) * np.dot(X.T, (np.dot(X, R) - Y))
        R -= learning_rate * gradient

    return R


def translate_word(
    source_word: str,
    source_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
    R: np.ndarray
) -> str:
    """Translates a source word into target language using transformation matrix R."""
    if source_word not in source_embeddings:
        raise ValueError(f"Word '{source_word}' not found in source vocabulary.")

    source_vec = source_embeddings[source_word]
    pred_target_vec = np.dot(source_vec, R)

    best_word = None
    best_sim = -1.0

    for word, target_vec in target_embeddings.items():
        sim = cosine_similarity(pred_target_vec, target_vec)
        if sim > best_sim:
            best_sim = sim
            best_word = word

    return best_word


class NLPClassifier:
    """Course 1 High-Level Classifier Class."""
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.log_model = LogisticRegression()
        self.nb_model = MultinomialNB()
        
    def train_logistic_regression(self, texts, labels):
        X_features = self.vectorizer.fit_transform(texts)
        self.log_model.fit(X_features, labels)
        return self.log_model
