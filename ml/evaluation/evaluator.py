"""
Model evaluation module for CrediLens AI.
This module will handle model evaluation metrics, confusion matrices, and result analysis.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import json


class ModelEvaluator:
    """Model evaluation utilities for credibility detection."""

    def __init__(self):
        """Initialize the model evaluator."""
        pass

    def evaluate(
        self,
        y_true: List,
        y_pred: List,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class ('micro', 'macro', 'weighted')

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return metrics

    def get_confusion_matrix(
        self,
        y_true: List,
        y_pred: List,
        labels: List[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names

        Returns:
            Tuple of (confusion_matrix, label_names)
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if labels is None:
            labels = sorted(list(set(y_true + y_pred)))
        return cm, labels

    def print_metrics(self, metrics: Dict[str, float], model_name: str = "Model") -> None:
        """
        Print evaluation metrics.

        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print(f"\n{model_name} - Evaluation Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"  {metric:15s}: {value:.4f}")

    def print_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        model_name: str = "Model"
    ) -> None:
        """
        Print confusion matrix.

        Args:
            cm: Confusion matrix
            labels: Label names
            model_name: Name of the model
        """
        print(f"\n{model_name} - Confusion Matrix:")
        print("-" * 50)
        print(f"{'':12s}", end='')
        for label in labels:
            print(f"{label:12s}", end='')
        print()
        for i, row in enumerate(cm):
            print(f"{labels[i]:12s}", end='')
            for val in row:
                print(f"{val:12d}", end='')
            print()

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filepath: str
    ) -> None:
        """
        Save evaluation metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            filepath: Path to save the metrics
        """
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        filepath: str
    ) -> None:
        """
        Save confusion matrix to JSON file.

        Args:
            cm: Confusion matrix
            labels: Label names
            filepath: Path to save the confusion matrix
        """
        data = {
            'confusion_matrix': cm.tolist(),
            'labels': labels
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        primary_metric: str = 'macro_f1'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models and rank by primary metric.

        Args:
            results: Dictionary of model_name -> metrics
            primary_metric: Metric to use for ranking

        Returns:
            Dictionary of model_name -> metrics (sorted by primary_metric)
        """
        # Sort by primary metric (descending)
        sorted_results = dict(
            sorted(
                results.items(),
                key=lambda x: x[1][primary_metric],
                reverse=True
            )
        )
        return sorted_results

    def print_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        primary_metric: str = 'macro_f1'
    ) -> None:
        """
        Print a comparison table of model results.

        Args:
            results: Dictionary of model_name -> metrics
            primary_metric: Primary metric to highlight
        """
        print(f"\nModel Comparison (sorted by {primary_metric}):")
        print("=" * 80)

        # Print header
        metrics = list(next(iter(results.values())).keys())
        print(f"{'Model':20s}", end='')
        for metric in metrics:
            marker = "*" if metric == primary_metric else ""
            print(f"{metric + marker:12s}", end='')
        print()
        print("-" * 80)

        # Print each model's results
        for model_name, metrics_dict in results.items():
            print(f"{model_name:20s}", end='')
            for metric in metrics:
                value = metrics_dict[metric]
                print(f"{value:12.4f}", end='')
            print()

        print("=" * 80)
        print(f"* Primary metric for model selection")
