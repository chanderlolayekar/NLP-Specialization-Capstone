import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes numerically stable softmax along a specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


from typing import Optional, Tuple

def scaled_dot_product_attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Scaled Dot-Product Attention using NumPy math.
    
    Formula: Softmax((Q @ K^T) / sqrt(d_k)) @ V

    Parameters:
        Q: Query matrix of shape (..., seq_len_q, d_k)
        K: Key matrix of shape (..., seq_len_k, d_k)
        V: Value matrix of shape (..., seq_len_v, d_v) (seq_len_k == seq_len_v)
        mask: Optional binary or additive mask tensor for casual decoding or padding
        
    Returns:
        output: Contextualized representations of shape (..., seq_len_q, d_v)
        attention_weights: Probability distributions of shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    
    # 1. Compute dot product similarity scores: (Q @ K^T) / sqrt(d_k)
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)

    # 2. Apply mask (if provided, e.g., setting padded/future positions to -1e9)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 3. Apply Softmax to get attention weights (probabilities)
    attention_weights = softmax(scores, axis=-1)

    # 4. Multiply attention weights by Values matrix V
    output = np.matmul(attention_weights, V)

    return output, attention_weights


class AttentionModels:
    """
    Course 4: Attention Models & Transformer Pipelines.
    Uses Hugging Face Transformers for end-to-end task inference.
    """
    def __init__(self):
        # Delayed import to avoid overhead if only using NumPy attention
        from transformers import pipeline
        self.translator = pipeline("translation_en_to_fr")
        self.summarizer = pipeline("summarization")
        self.qa_model = pipeline("question-answering")

    def machine_translate(self, text: str) -> str:
        """Translates complete sentences using an Encoder-Decoder transformer model."""
        res = self.translator(text)
        return res[0]["translation_text"]

    def summarize_text(self, text: str, max_len: int = 50) -> str:
        """Summarizes text using self-attention mechanisms."""
        res = self.summarizer(text, max_length=max_len, min_length=15, do_sample=False)
        return res[0]["summary_text"]

    def answer_question(self, question: str, context: str) -> str:
        """Performs reading comprehension using a pretrained Q&A transformer model."""
        res = self.qa_model(question=question, context=context)
        return res["answer"]
