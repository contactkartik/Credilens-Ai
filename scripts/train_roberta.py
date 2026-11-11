"""
Script to fine-tune RoBERTa for binary credibility classification.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml.evaluation.evaluator import ModelEvaluator
import time

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class LIARDataset(Dataset):
    """Custom dataset for LIAR credibility classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

def load_raw_data():
    """Load raw LIAR dataset."""
    print("Loading raw LIAR dataset...")

    columns = [
        'id', 'label', 'statement', 'subjects', 'speaker',
        'speaker_job', 'state_info', 'party_affiliation',
        'barely_true_counts', 'false_counts', 'half_true_counts',
        'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]

    def read_tsv(filepath):
        rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(line.strip().split('\t'))
        return rows

    def parse_data(rows):
        data = []
        for row in rows:
            if len(row) == len(columns):
                data.append(dict(zip(columns, row)))
        return data

    train_rows = read_tsv('../data/raw/train.tsv')
    valid_rows = read_tsv('../data/raw/valid.tsv')
    test_rows = read_tsv('../data/raw/test.tsv')

    train_data = parse_data(train_rows)
    valid_data = parse_data(valid_rows)
    test_data = parse_data(test_rows)

    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(valid_data)} samples")
    print(f"Test: {len(test_data)} samples")

    return train_data, valid_data, test_data

def preprocess_labels(data):
    """Convert labels to binary numeric format."""
    texts = [item['statement'] for item in data]
    labels_six = [item['label'] for item in data]

    # Binary label mapping
    true_side_labels = ['true', 'mostly-true', 'half-true']
    false_side_labels = ['barely-true', 'false', 'pants-fire']

    labels_binary = []
    for label in labels_six:
        if label in true_side_labels:
            labels_binary.append(1)  # true_side
        elif label in false_side_labels:
            labels_binary.append(0)  # false_side
        else:
            raise ValueError(f"Unknown label: {label}")

    return texts, labels_binary

def main():
    print("="*60)
    print("ROBERTA FINE-TUNING FOR CREDIBILITY CLASSIFICATION")
    print("="*60)

    # Create output directories
    os.makedirs('../models/roberta', exist_ok=True)
    os.makedirs('../outputs/evaluation', exist_ok=True)

    # Load raw data
    train_data, valid_data, test_data = load_raw_data()

    # Preprocess labels
    train_texts, train_labels = preprocess_labels(train_data)
    valid_texts, valid_labels = preprocess_labels(valid_data)
    test_texts, test_labels = preprocess_labels(test_data)

    print(f"\nLabel distribution:")
    print(f"Train: {sum(train_labels)} true_side, {len(train_labels)-sum(train_labels)} false_side")
    print(f"Validation: {sum(valid_labels)} true_side, {len(valid_labels)-sum(valid_labels)} false_side")
    print(f"Test: {sum(test_labels)} true_side, {len(test_labels)-sum(test_labels)} false_side")

    # Initialize tokenizer and model
    print("\nInitializing RoBERTa model and tokenizer...")
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print(f"Model: {model_name}")
    print(f"Parameters: {model.num_parameters():,}")

    # Create datasets
    print("\nCreating datasets...")
    max_length = 128
    train_dataset = LIARDataset(train_texts, train_labels, tokenizer, max_length)
    valid_dataset = LIARDataset(valid_texts, valid_labels, tokenizer, max_length)
    test_dataset = LIARDataset(test_texts, test_labels, tokenizer, max_length)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(valid_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Training arguments
    print("\nTraining configuration:")
    output_dir = '../models/roberta/checkpoint'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        learning_rate=2e-5,
        seed=42,
        data_seed=42,
        report_to='none'
    )

    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size (train): {training_args.per_device_train_batch_size}")
    print(f"  Batch size (eval): {training_args.per_device_eval_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    print(f"  Weight decay: {training_args.weight_decay}")
    print(f"  Device: {device}")

    # Initialize trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)

    valid_results = trainer.evaluate(valid_dataset)
    print("\nValidation Metrics:")
    for key, value in valid_results.items():
        if key.startswith('eval_'):
            print(f"  {key.replace('eval_', ''):15s}: {value:.4f}")

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    test_results = trainer.evaluate(test_dataset)
    print("\nTest Metrics:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            print(f"  {key.replace('eval_', ''):15s}: {value:.4f}")

    # Get predictions for confusion matrix
    print("\nGenerating predictions for confusion matrix...")
    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=1)
    y_true = test_predictions.label_ids

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    label_names = ['false_side', 'true_side']

    print("\nConfusion Matrix:")
    print(f"{'':12s}", end='')
    for label in label_names:
        print(f"{label:12s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{label_names[i]:12s}", end='')
        for val in row:
            print(f"{val:12d}", end='')
        print()

    # Compare with baseline
    print("\n" + "="*60)
    print("COMPARISON WITH LOGISTIC REGRESSION BASELINE")
    print("="*60)

    baseline_macro_f1 = 0.5977
    baseline_accuracy = 0.6209

    roberta_macro_f1 = test_results['eval_macro_f1']
    roberta_accuracy = test_results['eval_accuracy']

    print(f"\nLogistic Regression (Baseline):")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  Macro F1: {baseline_macro_f1:.4f}")

    print(f"\nRoBERTa:")
    print(f"  Accuracy: {roberta_accuracy:.4f}")
    print(f"  Macro F1: {roberta_macro_f1:.4f}")

    macro_f1_improvement = roberta_macro_f1 - baseline_macro_f1
    accuracy_improvement = roberta_accuracy - baseline_accuracy

    print(f"\nImprovement:")
    print(f"  Accuracy: {accuracy_improvement:+.4f} ({accuracy_improvement/baseline_accuracy*100:+.2f}%)")
    print(f"  Macro F1: {macro_f1_improvement:+.4f} ({macro_f1_improvement/baseline_macro_f1*100:+.2f}%)")

    # Save model and tokenizer
    print("\n" + "="*60)
    print("SAVING MODEL AND TOKENIZER")
    print("="*60)

    model.save_pretrained('../models/roberta')
    tokenizer.save_pretrained('../models/roberta')
    print("Model and tokenizer saved to models/roberta/")

    # Save evaluation results
    print("\nSaving evaluation results...")

    valid_metrics_dict = {
        'accuracy': float(valid_results['eval_accuracy']),
        'precision': float(valid_results['eval_precision']),
        'recall': float(valid_results['eval_recall']),
        'f1': float(valid_results['eval_f1']),
        'macro_f1': float(valid_results['eval_macro_f1']),
        'weighted_f1': float(valid_results['eval_weighted_f1'])
    }

    test_metrics_dict = {
        'accuracy': float(test_results['eval_accuracy']),
        'precision': float(test_results['eval_precision']),
        'recall': float(test_results['eval_recall']),
        'f1': float(test_results['eval_f1']),
        'macro_f1': float(test_results['eval_macro_f1']),
        'weighted_f1': float(test_results['eval_weighted_f1'])
    }

    with open('../outputs/evaluation/roberta_validation_metrics.json', 'w') as f:
        json.dump(valid_metrics_dict, f, indent=2)
    print("Validation metrics saved to outputs/evaluation/roberta_validation_metrics.json")

    with open('../outputs/evaluation/roberta_test_metrics.json', 'w') as f:
        json.dump(test_metrics_dict, f, indent=2)
    print("Test metrics saved to outputs/evaluation/roberta_test_metrics.json")

    confusion_matrix_dict = {
        'confusion_matrix': cm.tolist(),
        'labels': label_names
    }

    with open('../outputs/evaluation/roberta_confusion_matrix.json', 'w') as f:
        json.dump(confusion_matrix_dict, f, indent=2)
    print("Confusion matrix saved to outputs/evaluation/roberta_confusion_matrix.json")

    # Save training info
    training_info = {
        'model_name': model_name,
        'max_length': max_length,
        'num_train_epochs': training_args.num_train_epochs,
        'per_device_train_batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate,
        'warmup_steps': training_args.warmup_steps,
        'weight_decay': training_args.weight_decay,
        'training_time_seconds': training_time,
        'device': str(device),
        'num_parameters': model.num_parameters()
    }

    with open('../models/roberta/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    print("Training info saved to models/roberta/training_info.json")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    return {
        'model_name': model_name,
        'valid_metrics': valid_metrics_dict,
        'test_metrics': test_metrics_dict,
        'confusion_matrix': cm.tolist(),
        'training_time': training_time,
        'device': str(device)
    }

if __name__ == '__main__':
    results = main()
