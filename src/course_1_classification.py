import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class NLPClassifier:
    """
    Course 1: Classification & Vector Spaces
    Covers: Feature Engineering, Logistic Regression, Naïve Bayes, Word Embeddings
    """
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.log_model = LogisticRegression()
        self.nb_model = MultinomialNB()
        self.word_embeddings = {} # Dictionary mapping words to vectors
        
    def train_logistic_regression(self, texts, labels):
        """Trains a sentiment analysis model using Logistic Regression."""
        X_features = self.vectorizer.fit_transform(texts)
        self.log_model.fit(X_features, labels)
        return self.log_model

    def train_naive_bayes(self, texts, labels):
        """Trains a Naïve Bayes classifier for text mining."""
        X_features = self.vectorizer.fit_transform(texts)
        self.nb_model.fit(X_features, labels)
        return self.nb_model
        
    def complete_analogy(self, word_a, word_b, word_c):
        """
        Uses cosine similarity and word vectors to complete analogies.
        Equation: word_b - word_a + word_c = predicted_word
        *Note: In the Coursera lab, you will implement this using raw NumPy math.*
        """
        pass
