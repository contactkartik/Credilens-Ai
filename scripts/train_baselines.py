"""
Script to train and evaluate baseline ML models using TF-IDF features.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ml.evaluation.evaluator import ModelEvaluator

def load_processed_data():
    """Load processed data from Stage 3."""
    print("Loading processed data...")

    with open('../data/processed/train_processed.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('../data/processed/valid_processed.pkl', 'rb') as f:
        valid_data = pickle.load(f)

    with open('../data/processed/test_processed.pkl', 'rb') as f:
        test_data = pickle.load(f)

    print(f"Train: {train_data['features'].shape}")
    print(f"Validation: {valid_data['features'].shape}")
    print(f"Test: {test_data['features'].shape}")

    return train_data, valid_data, test_data

def train_logistic_regression(X_train, y_train, X_valid, y_valid):
    """Train and tune Logistic Regression."""
    print("\n" + "="*60)
    print("Training Logistic Regression")
    print("="*60)

    # Parameter grid for tuning
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }

    # Grid search on validation set
    print("Performing grid search on validation set...")
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )

    # Use training data for CV, validate on validation set
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_valid_pred = best_model.predict(X_valid)

    evaluator = ModelEvaluator()
    valid_metrics = evaluator.evaluate(y_valid, y_valid_pred, average='binary')
    evaluator.print_metrics(valid_metrics, "Logistic Regression (Validation)")

    return best_model, valid_metrics

def train_svm(X_train, y_train, X_valid, y_valid):
    """Train and tune Linear SVM."""
    print("\n" + "="*60)
    print("Training Linear SVM")
    print("="*60)

    # Parameter grid for tuning
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'loss': ['squared_hinge'],
        'max_iter': [1000]
    }

    # Grid search on validation set
    print("Performing grid search on validation set...")
    svm = LinearSVC(random_state=42)
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_valid_pred = best_model.predict(X_valid)

    evaluator = ModelEvaluator()
    valid_metrics = evaluator.evaluate(y_valid, y_valid_pred, average='binary')
    evaluator.print_metrics(valid_metrics, "Linear SVM (Validation)")

    return best_model, valid_metrics

def train_random_forest(X_train, y_train, X_valid, y_valid):
    """Train and tune Random Forest."""
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)

    # Parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Grid search on validation set
    print("Performing grid search on validation set...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_valid_pred = best_model.predict(X_valid)

    evaluator = ModelEvaluator()
    valid_metrics = evaluator.evaluate(y_valid, y_valid_pred, average='binary')
    evaluator.print_metrics(valid_metrics, "Random Forest (Validation)")

    return best_model, valid_metrics

def evaluate_on_test(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    print(f"\n{model_name} - Test Set Evaluation")
    print("-" * 50)

    y_test_pred = model.predict(X_test)

    evaluator = ModelEvaluator()
    test_metrics = evaluator.evaluate(y_test, y_test_pred, average='binary')
    evaluator.print_metrics(test_metrics, f"{model_name} (Test)")

    # Confusion matrix with string labels
    label_map_inverse = {0: 'false_side', 1: 'true_side'}
    y_test_str = [label_map_inverse[label] for label in y_test]
    y_test_pred_str = [label_map_inverse[label] for label in y_test_pred]

    cm, labels = evaluator.get_confusion_matrix(y_test_str, y_test_pred_str, labels=['false_side', 'true_side'])
    evaluator.print_confusion_matrix(cm, labels, model_name)

    return test_metrics, cm, labels

def save_model(model, filepath):
    """Save trained model to disk."""
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def main():
    print("="*60)
    print("BASELINE ML MODEL TRAINING AND EVALUATION")
    print("="*60)

    # Create output directories
    os.makedirs('../models/baseline', exist_ok=True)
    os.makedirs('../outputs/evaluation', exist_ok=True)

    # Load data
    train_data, valid_data, test_data = load_processed_data()

    # Extract features and labels (using binary labels)
    X_train = train_data['features']
    y_train = train_data['labels_binary']
    X_valid = valid_data['features']
    y_valid = valid_data['labels_binary']
    X_test = test_data['features']
    y_test = test_data['labels_binary']

    # Convert string labels to numeric for sklearn
    label_map = {'false_side': 0, 'true_side': 1}
    y_train_numeric = [label_map[label] for label in y_train]
    y_valid_numeric = [label_map[label] for label in y_valid]
    y_test_numeric = [label_map[label] for label in y_test]

    print(f"\nUsing binary labels:")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Validation: {len(y_valid)} samples")
    print(f"  Test: {len(y_test)} samples")
    print(f"  Label mapping: {label_map}")

    # Train models
    models = {}
    valid_results = {}

    # Logistic Regression
    lr_model, lr_valid_metrics = train_logistic_regression(X_train, y_train_numeric, X_valid, y_valid_numeric)
    models['logistic_regression'] = lr_model
    valid_results['logistic_regression'] = lr_valid_metrics

    # Linear SVM
    svm_model, svm_valid_metrics = train_svm(X_train, y_train_numeric, X_valid, y_valid_numeric)
    models['linear_svm'] = svm_model
    valid_results['linear_svm'] = svm_valid_metrics

    # Random Forest
    rf_model, rf_valid_metrics = train_random_forest(X_train, y_train_numeric, X_valid, y_valid_numeric)
    models['random_forest'] = rf_model
    valid_results['random_forest'] = rf_valid_metrics

    # Compare models on validation set
    print("\n" + "="*60)
    print("VALIDATION SET COMPARISON")
    print("="*60)

    evaluator = ModelEvaluator()
    sorted_valid_results = evaluator.compare_models(valid_results, primary_metric='macro_f1')
    evaluator.print_comparison_table(sorted_valid_results, primary_metric='macro_f1')

    # Select best model based on validation macro_f1
    best_model_name = list(sorted_valid_results.keys())[0]
    print(f"\nBest model on validation set: {best_model_name}")
    print(f"Validation Macro F1: {sorted_valid_results[best_model_name]['macro_f1']:.4f}")

    # Evaluate all models on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    test_results = {}
    confusion_matrices = {}

    for model_name, model in models.items():
        test_metrics, cm, labels = evaluate_on_test(model, X_test, y_test_numeric, model_name)
        test_results[model_name] = test_metrics
        confusion_matrices[model_name] = {
            'confusion_matrix': cm,
            'labels': labels
        }

    # Compare models on test set
    print("\n" + "="*60)
    print("TEST SET COMPARISON")
    print("="*60)

    sorted_test_results = evaluator.compare_models(test_results, primary_metric='macro_f1')
    evaluator.print_comparison_table(sorted_test_results, primary_metric='macro_f1')

    # Final best model on test set
    best_test_model_name = list(sorted_test_results.keys())[0]
    print(f"\nBest model on test set: {best_test_model_name}")
    print(f"Test Macro F1: {sorted_test_results[best_test_model_name]['macro_f1']:.4f}")

    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS AND RESULTS")
    print("="*60)

    for model_name, model in models.items():
        model_filename = model_name.replace('_', '-') + '.pkl'
        save_model(model, f'../models/baseline/{model_filename}')

    # Save evaluation results
    evaluator.save_metrics(
        valid_results,
        '../outputs/evaluation/validation_metrics.json'
    )
    print("Validation metrics saved to outputs/evaluation/validation_metrics.json")

    evaluator.save_metrics(
        test_results,
        '../outputs/evaluation/test_metrics.json'
    )
    print("Test metrics saved to outputs/evaluation/test_metrics.json")

    # Save confusion matrices
    for model_name, cm_data in confusion_matrices.items():
        cm_filename = model_name.replace('_', '-') + '_confusion_matrix.json'
        evaluator.save_confusion_matrix(
            cm_data['confusion_matrix'],
            cm_data['labels'],
            f'../outputs/evaluation/{cm_filename}'
        )
        print(f"Confusion matrix saved to outputs/evaluation/{cm_filename}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    return {
        'best_model': best_test_model_name,
        'test_results': test_results,
        'valid_results': valid_results
    }

if __name__ == '__main__':
    results = main()
