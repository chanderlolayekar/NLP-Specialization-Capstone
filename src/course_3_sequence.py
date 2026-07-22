import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    GRU,
    TimeDistributed,
    Dense,
    Dropout
)


def build_ner_model(
    vocab_size: int,
    num_tags: int,
    max_len: int = 50,
    embedding_dim: int = 64,
    rnn_units: int = 64,
    use_gru: bool = False
) -> tf.keras.Model:
    """
    Builds a Bidirectional LSTM or GRU model for Named Entity Recognition (NER).

    Parameters:
        vocab_size: Size of the token vocabulary
        num_tags: Number of distinct entity tags (e.g., PER, LOC, ORG, O)
        max_len: Fixed length of padded input sequences
        embedding_dim: Dimension of dense embedding vectors
        rnn_units: Number of units in hidden recurrent states
        use_gru: If True, uses GRU instead of LSTM
    """
    model = Sequential()
    
    # Input layer matching sequence length
    model.add(Input(shape=(max_len,)))
    
    # Mask zero-padding so padded tokens don't affect gradients
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True))
    
    # Recurrent Layer (Bidirectional LSTM or GRU)
    rnn_layer = GRU(rnn_units, return_sequences=True) if use_gru else LSTM(rnn_units, return_sequences=True)
    model.add(Bidirectional(rnn_layer))
    
    model.add(Dropout(0.3))
    
    # TimeDistributed applies Dense classification across all time-steps (tokens)
    model.add(TimeDistributed(Dense(num_tags, activation="softmax")))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


class SequenceModels:
    """Course 3: High-Level Sequence Models Class."""
    
    @staticmethod
    def train_ner_step(
        model: tf.keras.Model, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        epochs: int = 1
    ):
        """Fits the NER model on tokenized integer sequences and label sequences."""
        return model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
