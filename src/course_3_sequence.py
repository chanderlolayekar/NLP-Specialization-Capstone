import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Bidirectional, Input

class SequenceModels:
    """
    Course 3: Sequence Models
    Covers: RNNs, LSTMs, GRUs, Siamese Networks, Named Entity Recognition
    """
    def build_sentiment_lstm(self, vocab_size, embedding_dim, max_length):
        """Builds a Bidirectional LSTM/GRU for advanced text classification."""
        model = Sequential([
            Input(shape=(max_length,)),
            Embedding(vocab_size, embedding_dim),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(GRU(32)),
            Dense(24, activation='relu'),
            Dense(1, activation='sigmoid') # Binary sentiment output
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
        
    def build_siamese_network(self):
        """
        Builds a Siamese Neural Network to identify duplicate questions or texts.
        *Note: You will implement custom contrastive loss functions in the course.*
        """
        pass

    def text_generation_rnn(self, seed_text):
        """Generates novel text character-by-character using deep RNNs."""
        pass
