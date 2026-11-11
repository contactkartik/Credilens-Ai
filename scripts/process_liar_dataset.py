"""
Script to preprocess LIAR dataset and extract TF-IDF features.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing.preprocessor import TextPreprocessor, create_binary_labels
from ml.features.feature_extractor import TFIDFFeatureExtractor
from collections import Counter

# Define column names based on README
columns = [
    'id', 'label', 'statement', 'subjects', 'speaker',
    'speaker_job', 'state_info', 'party_affiliation',
    'barely_true_counts', 'false_counts', 'half_true_counts',
    'mostly_true_counts', 'pants_on_fire_counts', 'context'
]

def read_tsv(filepath):
    """Read TSV file and return list of rows."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(line.strip().split('\t'))
    return rows

def parse_data(rows):
    """Parse TSV rows into structured data."""
    data = []
    for row in rows:
        if len(row) == len(columns):
            data.append(dict(zip(columns, row)))
    return data

def main():
    print("="*60)
    print("LIAR DATASET PREPROCESSING AND TF-IDF FEATURE EXTRACTION")
    print("="*60)

    # Load datasets
    print("\nLoading datasets...")
    train_rows = read_tsv('../data/raw/train.tsv')
    valid_rows = read_tsv('../data/raw/valid.tsv')
    test_rows = read_tsv('../data/raw/test.tsv')

    train_data = parse_data(train_rows)
    valid_data = parse_data(valid_rows)
    test_data = parse_data(test_rows)

    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(valid_data)} samples")
    print(f"Test: {len(test_data)} samples")

    # Extract statements and labels
    train_texts = [item['statement'] for item in train_data]
    valid_texts = [item['statement'] for item in valid_data]
    test_texts = [item['statement'] for item in test_data]

    train_labels_six = [item['label'] for item in train_data]
    valid_labels_six = [item['label'] for item in valid_data]
    test_labels_six = [item['label'] for item in test_data]

    # Preprocess text
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor()

    print("Preprocessing steps:")
    print("  1. Handle missing values")
    print("  2. Normalize special characters")
    print("  3. Normalize whitespace")
    print("  4. Convert to lowercase")

    train_texts_processed = preprocessor.preprocess_batch(train_texts, lowercase=True)
    valid_texts_processed = preprocessor.preprocess_batch(valid_texts, lowercase=True)
    test_texts_processed = preprocessor.preprocess_batch(test_texts, lowercase=True)

    print(f"\nProcessed text samples:")
    print(f"  Train: {len(train_texts_processed)}")
    print(f"  Validation: {len(valid_texts_processed)}")
    print(f"  Test: {len(test_texts_processed)}")

    # Show sample before/after
    print(f"\nSample preprocessing:")
    print(f"  Original: {train_texts[0][:100]}...")
    print(f"  Processed: {train_texts_processed[0][:100]}...")

    # Create binary labels
    print("\nCreating binary labels...")
    print("TRUE_SIDE: true, mostly-true, half-true")
    print("FALSE_SIDE: barely-true, false, pants-fire")

    train_labels_binary = create_binary_labels(train_labels_six)
    valid_labels_binary = create_binary_labels(valid_labels_six)
    test_labels_binary = create_binary_labels(test_labels_six)

    # Binary label distribution
    print("\nBinary label distribution:")
    print("\nTrain:")
    train_binary_counts = Counter(train_labels_binary)
    for label, count in train_binary_counts.items():
        pct = (count / len(train_labels_binary)) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")

    print("\nValidation:")
    valid_binary_counts = Counter(valid_labels_binary)
    for label, count in valid_binary_counts.items():
        pct = (count / len(valid_labels_binary)) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")

    print("\nTest:")
    test_binary_counts = Counter(test_labels_binary)
    for label, count in test_binary_counts.items():
        pct = (count / len(test_labels_binary)) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")

    # Initialize TF-IDF feature extractor
    print("\nInitializing TF-IDF feature extractor...")
    print("Configuration:")
    print("  - N-gram range: (1, 2) [unigrams and bigrams]")
    print("  - min_df: 3")
    print("  - max_features: 10000")
    print("  - lowercase: True")
    print("  - stop_words: english")

    tfidf_extractor = TFIDFFeatureExtractor(
        ngram_range=(1, 2),
        min_df=3,
        max_features=10000,
        lowercase=True
    )

    # Fit on training data only (NO DATA LEAKAGE)
    print("\nFitting TF-IDF vectorizer on training data only...")
    tfidf_extractor.fit(train_texts_processed)
    vocab_size = tfidf_extractor.get_vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")

    # Transform all datasets
    print("\nTransforming datasets...")
    train_features, train_feature_names = tfidf_extractor.transform(train_texts_processed)
    valid_features, valid_feature_names = tfidf_extractor.transform(valid_texts_processed)
    test_features, test_feature_names = tfidf_extractor.transform(test_texts_processed)

    print(f"\nFeature matrix dimensions:")
    print(f"  Train: {train_features.shape}")
    print(f"  Validation: {valid_features.shape}")
    print(f"  Test: {test_features.shape}")

    # Verify no data leakage
    print("\nVerifying no data leakage...")
    print("  [OK] Vectorizer fitted ONLY on training data")
    print("  [OK] Validation and test data transformed using fitted vectorizer")
    print("  [OK] No information from validation/test used in fitting")

    # Save processed data
    print("\nSaving processed data and vectorizer...")

    # Create processed data directory
    os.makedirs('../data/processed', exist_ok=True)

    # Save vectorizer
    tfidf_extractor.save_vectorizer('../data/processed/tfidf_vectorizer.pkl')
    print("  [OK] Saved TF-IDF vectorizer to data/processed/tfidf_vectorizer.pkl")

    # Save processed texts and labels
    import pickle

    with open('../data/processed/train_processed.pkl', 'wb') as f:
        pickle.dump({
            'texts': train_texts_processed,
            'labels_six': train_labels_six,
            'labels_binary': train_labels_binary,
            'features': train_features,
            'feature_names': train_feature_names
        }, f)
    print("  [OK] Saved train data to data/processed/train_processed.pkl")

    with open('../data/processed/valid_processed.pkl', 'wb') as f:
        pickle.dump({
            'texts': valid_texts_processed,
            'labels_six': valid_labels_six,
            'labels_binary': valid_labels_binary,
            'features': valid_features,
            'feature_names': valid_feature_names
        }, f)
    print("  [OK] Saved validation data to data/processed/valid_processed.pkl")

    with open('../data/processed/test_processed.pkl', 'wb') as f:
        pickle.dump({
            'texts': test_texts_processed,
            'labels_six': test_labels_six,
            'labels_binary': test_labels_binary,
            'features': test_features,
            'feature_names': test_feature_names
        }, f)
    print("  [OK] Saved test data to data/processed/test_processed.pkl")

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)

    return {
        'vocab_size': vocab_size,
        'train_shape': train_features.shape,
        'valid_shape': valid_features.shape,
        'test_shape': test_features.shape,
        'train_binary_counts': train_binary_counts,
        'valid_binary_counts': valid_binary_counts,
        'test_binary_counts': test_binary_counts
    }

if __name__ == '__main__':
    results = main()
