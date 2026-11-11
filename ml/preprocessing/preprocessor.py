"""
Data preprocessing module for CrediLens AI.
This module will handle text cleaning, normalization, and preparation for ML models.
"""
import re
from typing import List, Optional


class TextPreprocessor:
    """Text preprocessing utilities for credibility detection."""

    def __init__(self):
        """Initialize the text preprocessor."""
        pass

    def handle_missing_values(self, text: str) -> str:
        """
        Handle missing or empty text values.

        Args:
            text: Input text string

        Returns:
            Processed text string
        """
        if text is None or not isinstance(text, str):
            return ""
        if text.strip() == "":
            return ""
        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.

        Args:
            text: Input text string

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def normalize_special_chars(self, text: str) -> str:
        """
        Normalize special characters while preserving linguistic information.

        Args:
            text: Input text string

        Returns:
            Text with normalized special characters
        """
        # Normalize quotes to standard quotes
        text = re.sub(r'["`]', '"', text)
        text = re.sub(r"'", '"', text)
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        # Normalize ellipsis
        text = re.sub(r'\.{2,}', '...', text)
        return text

    def lowercase_text(self, text: str) -> str:
        """
        Convert text to lowercase.

        Args:
            text: Input text string

        Returns:
            Lowercased text
        """
        return text.lower()

    def preprocess(self, text: str, lowercase: bool = True) -> str:
        """
        Apply full preprocessing pipeline to text.

        Args:
            text: Input text string
            lowercase: Whether to convert to lowercase

        Returns:
            Preprocessed text string
        """
        # Handle missing values
        text = self.handle_missing_values(text)
        if not text:
            return text

        # Normalize special characters
        text = self.normalize_special_chars(text)

        # Normalize whitespace
        text = self.normalize_whitespace(text)

        # Lowercase if requested
        if lowercase:
            text = self.lowercase_text(text)

        return text

    def preprocess_batch(self, texts: List[str], lowercase: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of input text strings
            lowercase: Whether to convert to lowercase

        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess(text, lowercase=lowercase) for text in texts]


def create_binary_label(label: str) -> str:
    """
    Convert six-class label to binary credibility label.

    TRUE_SIDE: true, mostly-true, half-true
    FALSE_SIDE: barely-true, false, pants-fire

    Args:
        label: Original six-class label

    Returns:
        Binary label: 'true_side' or 'false_side'
    """
    true_side_labels = ['true', 'mostly-true', 'half-true']
    false_side_labels = ['barely-true', 'false', 'pants-fire']

    if label in true_side_labels:
        return 'true_side'
    elif label in false_side_labels:
        return 'false_side'
    else:
        raise ValueError(f"Unknown label: {label}")


def create_binary_labels(labels: List[str]) -> List[str]:
    """
    Convert a list of six-class labels to binary credibility labels.

    Args:
        labels: List of original six-class labels

    Returns:
        List of binary labels
    """
    return [create_binary_label(label) for label in labels]
