"""
Course 3: Sequence Models
Covers Named Entity Recognition (NER) with Bidirectional LSTMs and
Siamese Networks for sentence similarity using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Dict, Any


def build_ner_model(
    vocab_size: int = 1000, 
    num_tags: int = 10, 
    embedding_dim: int = 64, 
    lstm_units: int = 64
) -> Model:
    """
    Builds a Bidirectional LSTM model for Named Entity Recognition (NER).
    
    Parameters:
        vocab_size: Size of input vocabulary
        num_tags: Number of distinct entity tags
        embedding_dim: Dimension of word embeddings
        lstm_units: Hidden units for LSTM layer
        
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=(None,), dtype="int32", name="input_tokens")
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    outputs = layers.Dense(num_tags, activation="softmax", name="ner_tags")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="ner_bilstm")
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model


def build_siamese_network(
    vocab_size: int = 1000, 
    embedding_dim: int = 64, 
    lstm_units: int = 64
) -> Model:
    """
    Builds a Siamese LSTM network for sentence/sequence similarity scoring.
    
    Parameters:
        vocab_size: Size of input vocabulary
        embedding_dim: Dimension of word embeddings
        lstm_units: Hidden units for shared LSTM encoder
        
    Returns:
        Compiled Keras Model taking pair of sequence inputs
    """
    input_a = layers.Input(shape=(None,), dtype="int32", name="sequence_a")
    input_b = layers.Input(shape=(None,), dtype="int32", name="sequence_b")
    
    # Shared layers
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    lstm_layer = layers.LSTM(lstm_units)
    
    # Encode both inputs with shared weights
    encoded_a = lstm_layer(embedding_layer(input_a))
    encoded_b = lstm_layer(embedding_layer(input_b))
    
    # Compute absolute difference distance vector
    distance = layers.Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]), 
        name="absolute_difference"
    )([encoded_a, encoded_b])
    
    outputs = layers.Dense(1, activation="sigmoid", name="similarity_score")(distance)
    
    model = Model(inputs=[input_a, input_b], outputs=outputs, name="siamese_lstm")
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    return model


class SequenceModels:
    """Wrapper class providing high-level interface for sequence model training and evaluation."""

    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ner_model = None
        self.siamese_model = None

    def initialize_ner(self, num_tags: int = 10, lstm_units: int = 64) -> Model:
        """Initializes and builds the NER model instance."""
        self.ner_model = build_ner_model(
            vocab_size=self.vocab_size,
            num_tags=num_tags,
            embedding_dim=self.embedding_dim,
            lstm_units=lstm_units
        )
        return self.ner_model

    def initialize_siamese(self, lstm_units: int = 64) -> Model:
        """Initializes and builds the Siamese network instance."""
        self.siamese_model = build_siamese_network(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=lstm_units
        )
        return self.siamese_model
