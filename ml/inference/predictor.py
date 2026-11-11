"""
Inference module for CrediLens AI.
This module will handle model loading and prediction generation.
"""
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import List, Union, Dict
import numpy as np


class RoBERTaPredictor:
    """RoBERTa model predictor for credibility classification."""

    def __init__(self, model_path: str = '../models/roberta'):
        """
        Initialize the RoBERTa predictor.

        Args:
            model_path: Path to the saved model and tokenizer
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_map = {0: 'false_side', 1: 'true_side'}

    def load_model(self):
        """Load the fine-tuned RoBERTa model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def predict(
        self,
        text: Union[str, List[str]],
        max_length: int = 128
    ) -> Union[Dict, List[Dict]]:
        """
        Predict credibility for text.

        Args:
            text: Input text (string or list of strings)
            max_length: Maximum sequence length

        Returns:
            Prediction dictionary or list of dictionaries
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

        # Convert to CPU and numpy
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        # Format results
        results = []
        for i in range(len(text)):
            pred_label = int(predictions[i])
            pred_class = self.label_map[pred_label]
            prob_false = float(probabilities[i][0])
            prob_true = float(probabilities[i][1])

            results.append({
                'text': text[i],
                'prediction': pred_class,
                'confidence': float(max(probabilities[i])),
                'probabilities': {
                    'false_side': prob_false,
                    'true_side': prob_true
                }
            })

        if is_single:
            return results[0]
        return results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 128
    ) -> List[Dict]:
        """
        Predict credibility for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for inference
            max_length: Maximum sequence length

        Returns:
            List of prediction dictionaries
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        all_results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.predict(batch_texts, max_length)
            all_results.extend(batch_results)

        return all_results
