# Stage 4: Baseline ML Model Training and Evaluation - Summary

## Models Trained

Three baseline models were trained using TF-IDF features:
1. **Logistic Regression**
2. **Linear SVM**
3. **Random Forest**

## Training Configuration

### Feature Configuration
- **Features**: TF-IDF (unigrams + bigrams)
- **Vocabulary size**: 8,157 features
- **min_df**: 3
- **max_features**: 10,000

### Hyperparameter Tuning
- **Method**: GridSearchCV with 3-fold cross-validation
- **Scoring metric**: F1 Macro
- **Validation set**: Used for model selection

### Data Splits
- **Training**: 10,169 samples
- **Validation**: 1,272 samples (for hyperparameter tuning)
- **Test**: 1,266 samples (for final evaluation only)

### Label Mapping
- **false_side**: 0
- **true_side**: 1

## Validation Set Results

### Logistic Regression
**Best Parameters**: C=1.0, penalty=l2, solver=liblinear, max_iter=1000
**Best CV Score**: 0.5816

| Metric | Value |
|--------|-------|
| Accuracy | 0.6053 |
| Precision | 0.5990 |
| Recall | 0.7345 |
| F1 | 0.6599 |
| **Macro F1** | **0.5949** |
| Weighted F1 | 0.5977 |

### Linear SVM
**Best Parameters**: C=1.0, loss=squared_hinge, penalty=l2, max_iter=1000
**Best CV Score**: 0.5798

| Metric | Value |
|--------|-------|
| Accuracy | 0.5763 |
| Precision | 0.5852 |
| Recall | 0.6425 |
| F1 | 0.6125 |
| **Macro F1** | **0.5725** |
| Weighted F1 | 0.5742 |

### Random Forest
**Best Parameters**: n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=1
**Best CV Score**: 0.5685

| Metric | Value |
|--------|-------|
| Accuracy | 0.6061 |
| Precision | 0.5967 |
| Recall | 0.7541 |
| F1 | 0.6662 |
| **Macro F1** | **0.5929** |
| Weighted F1 | 0.5960 |

### Validation Set Ranking (by Macro F1)
1. **Logistic Regression**: 0.5949
2. **Random Forest**: 0.5929
3. **Linear SVM**: 0.5725

## Test Set Results

### Logistic Regression
| Metric | Value |
|--------|-------|
| Accuracy | 0.6209 |
| Precision | 0.6367 |
| Recall | 0.7633 |
| F1 | 0.6943 |
| **Macro F1** | **0.5977** |
| Weighted F1 | 0.6100 |

**Confusion Matrix**:
```
                false_side  true_side
false_side           241         311
true_side            169         545
```

### Linear SVM
| Metric | Value |
|--------|-------|
| Accuracy | 0.5987 |
| Precision | 0.6334 |
| Recall | 0.6849 |
| F1 | 0.6581 |
| **Macro F1** | **0.5862** |
| Weighted F1 | 0.5954 |

**Confusion Matrix**:
```
                false_side  true_side
false_side           269         283
true_side            225         489
```

### Random Forest
| Metric | Value |
|--------|-------|
| Accuracy | 0.6098 |
| Precision | 0.6259 |
| Recall | 0.7661 |
| F1 | 0.6889 |
| **Macro F1** | **0.5828** |
| Weighted F1 | 0.5964 |

**Confusion Matrix**:
```
                false_side  true_side
false_side           225         327
true_side            167         547
```

### Test Set Ranking (by Macro F1)
1. **Logistic Regression**: 0.5977
2. **Linear SVM**: 0.5862
3. **Random Forest**: 0.5828

## Best Baseline Model

**Logistic Regression** is the best baseline model based on Macro F1 score on both validation and test sets.

### Key Observations
- **Validation Macro F1**: 0.5949
- **Test Macro F1**: 0.5977
- **Test Accuracy**: 0.6209
- **High Recall**: 0.7633 (good at identifying true_side statements)
- **Moderate Precision**: 0.6367 (some false positives)
- **Confusion Matrix**: More false negatives (311) than false positives (169) for false_side

## Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 | Macro F1* | Weighted F1 |
|-------|----------|-----------|--------|-----|-----------|-------------|
| **Logistic Regression** | 0.6209 | 0.6367 | 0.7633 | 0.6943 | **0.5977** | 0.6100 |
| Linear SVM | 0.5987 | 0.6334 | 0.6849 | 0.6581 | 0.5862 | 0.5954 |
| Random Forest | 0.6098 | 0.6259 | 0.7661 | 0.6889 | 0.5828 | 0.5964 |

*Primary metric for model selection

## Confusion Matrix Analysis

### Common Patterns Across Models
1. **High Recall for true_side**: All models have higher recall for true_side (0.68-0.77)
2. **More False Negatives for false_side**: Models tend to misclassify false_side as true_side
3. **Balanced Performance**: Similar accuracy across models (0.60-0.62)

### Logistic Regression Confusion Matrix Details
- **True Positives (TP)**: 545 (correctly identified true_side)
- **True Negatives (TN)**: 241 (correctly identified false_side)
- **False Positives (FP)**: 169 (false_side incorrectly predicted as true_side)
- **False Negatives (FN)**: 311 (true_side incorrectly predicted as false_side)

**Implications**:
- Model is better at identifying true_side statements
- Model has bias towards predicting true_side
- May need class balancing or threshold tuning

## Files Created/Modified

### Created
1. **ml/evaluation/evaluator.py** - Model evaluation utilities
   - ModelEvaluator class
   - evaluate(), get_confusion_matrix() methods
   - save_metrics(), save_confusion_matrix() methods
   - compare_models(), print_comparison_table() methods

2. **scripts/train_baselines.py** - Baseline training script
   - train_logistic_regression()
   - train_svm()
   - train_random_forest()
   - evaluate_on_test()
   - Grid search with hyperparameter tuning

3. **models/baseline/logistic-regression.pkl** - Trained Logistic Regression model
4. **models/baseline/linear-svm.pkl** - Trained Linear SVM model
5. **models/baseline/random-forest.pkl** - Trained Random Forest model

6. **outputs/evaluation/validation_metrics.json** - Validation set metrics for all models
7. **outputs/evaluation/test_metrics.json** - Test set metrics for all models
8. **outputs/evaluation/logistic-regression_confusion_matrix.json** - LR confusion matrix
9. **outputs/evaluation/linear-svm_confusion_matrix.json** - SVM confusion matrix
10. **outputs/evaluation/random-forest_confusion_matrix.json** - RF confusion matrix

11. **docs/stage4_baseline_results.md** - This document

### Modified
- No existing files were modified

## Key Findings

1. **Logistic Regression Performs Best**: Achieves highest Macro F1 (0.5977) on test set
2. **Moderate Performance**: All models achieve ~60% accuracy, indicating room for improvement
3. **Class Imbalance Impact**: Higher recall for true_side suggests bias towards majority class
4. **TF-IDF Limitations**: Bag-of-words features may not capture semantic nuances
5. **Consistent Ranking**: Validation and test rankings are consistent (Logistic Regression best)

## Recommendations for Improvement

1. **Address Class Imbalance**: Use class weights or oversampling techniques
2. **Feature Engineering**: Add linguistic features, sentiment analysis, metadata
3. **Advanced Models**: Try deep learning models (RoBERTa, BERT)
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Threshold Tuning**: Optimize decision threshold for specific use cases

## Next Steps

Stage 4 is complete. The baseline models establish a performance benchmark:
- **Best Baseline**: Logistic Regression (Macro F1: 0.5977)
- **Benchmark Accuracy**: 62.09%

This provides a solid foundation for comparing more advanced models in subsequent stages (e.g., RoBERTa).

## Errors Encountered

**Initial Error**: String labels caused sklearn metrics to fail
- **Fix**: Converted string labels ('false_side', 'true_side') to numeric (0, 1) for training
- **Confusion matrices**: Converted back to string labels for interpretability

All subsequent training and evaluation completed successfully without errors.
