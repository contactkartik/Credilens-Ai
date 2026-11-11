"""
Feature extraction module for CrediLens AI.
This module will extract NLP features, linguistic patterns, and multimodal features.
"""
import pickle
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatureExtractor:
    """TF-IDF feature extraction for text credibility detection."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 3,
        max_features: Optional[int] = 10000,
        lowercase: bool = True
    ):
        """
        Initialize TF-IDF feature extractor.

        Args:
            ngram_range: The lower and upper boundary of the range of n-values
            min_df: Minimum document frequency
            max_features: Maximum number of features (vocabulary size)
            lowercase: Whether to lowercase text
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.lowercase = lowercase
        self.vectorizer = None

    def fit(self, texts: List[str]) -> 'TFIDFFeatureExtractor':
        """
        Fit TF-IDF vectorizer on training texts.

        Args:
            texts: List of preprocessed text strings

        Returns:
            Self (fitted vectorizer)
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            lowercase=self.lowercase,
            stop_words='english'
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]) -> Tuple:
        """
        Transform texts to TF-IDF feature matrix.

        Args:
            texts: List of preprocessed text strings

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer has not been fitted. Call fit() first.")

        feature_matrix = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        return feature_matrix, feature_names

    def fit_transform(self, texts: List[str]) -> Tuple:
        """
        Fit vectorizer and transform texts in one step.

        Args:
            texts: List of preprocessed text strings

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.fit(texts)
        return self.transform(texts)

    def get_vocabulary_size(self) -> int:
        """
        Get the size of the fitted vocabulary.

        Returns:
            Vocabulary size
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer has not been fitted. Call fit() first.")
        return len(self.vectorizer.vocabulary_)

    def save_vectorizer(self, filepath: str) -> None:
        """
        Save the fitted vectorizer to disk.

        Args:
            filepath: Path to save the vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer has not been fitted. Call fit() first.")
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_vectorizer(self, filepath: str) -> 'TFIDFFeatureExtractor':
        """
        Load a fitted vectorizer from disk.

        Args:
            filepath: Path to load the vectorizer from

        Returns:
            Self (with loaded vectorizer)
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        return self
