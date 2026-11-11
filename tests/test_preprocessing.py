"""
Test script to verify preprocessing and TF-IDF pipeline.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing.preprocessor import TextPreprocessor, create_binary_labels
from ml.features.feature_extractor import TFIDFFeatureExtractor
import pickle

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    print("="*60)
    print("TESTING PREPROCESSING AND TF-IDF PIPELINE")
    print("="*60)

    # Test 1: Text Preprocessing
    print("\nTest 1: Text Preprocessing")
    print("-" * 40)

    preprocessor = TextPreprocessor()

    test_texts = [
        "  This is a TEST statement  with  extra   spaces.  ",
        "Normal statement.",
        "",
        None,
        "Statement with \"quotes\" and 'apostrophes'."
    ]

    print("Input texts:")
    for i, text in enumerate(test_texts):
        print(f"  {i}: {repr(text)}")

    processed = preprocessor.preprocess_batch(test_texts, lowercase=True)

    print("\nProcessed texts:")
    for i, text in enumerate(processed):
        print(f"  {i}: {repr(text)}")

    # Verify preprocessing
    assert processed[0] == "this is a test statement with extra spaces.", "Whitespace normalization failed"
    assert processed[1] == "normal statement.", "Basic processing failed"
    assert processed[2] == "", "Empty string handling failed"
    assert processed[3] == "", "None handling failed"
    # Check that preprocessing produces lowercase output
    assert processed[4].islower() or not any(c.isupper() for c in processed[4]), "Lowercasing failed"

    print("\n[OK] All preprocessing tests passed")

    # Test 2: Binary Label Creation
    print("\nTest 2: Binary Label Creation")
    print("-" * 40)

    test_labels = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
    print("Input labels:", test_labels)

    binary_labels = create_binary_labels(test_labels)
    print("Binary labels:", binary_labels)

    expected = ['true_side', 'true_side', 'true_side', 'false_side', 'false_side', 'false_side']
    assert binary_labels == expected, f"Binary label conversion failed: {binary_labels} != {expected}"

    print("\n[OK] Binary label creation test passed")

    # Test 3: TF-IDF Feature Extraction
    print("\nTest 3: TF-IDF Feature Extraction")
    print("-" * 40)

    # Create sample texts
    sample_train = [
        "This is a sample statement about politics.",
        "Another statement with different words.",
        "Politics and government are important topics.",
        "Sample text for testing the pipeline."
    ]

    sample_test = [
        "A new statement about politics and government.",
        "Different text for testing purposes."
    ]

    print(f"Train samples: {len(sample_train)}")
    print(f"Test samples: {len(sample_test)}")

    # Initialize and fit TF-IDF
    tfidf = TFIDFFeatureExtractor(
        ngram_range=(1, 2),
        min_df=1,
        max_features=100,
        lowercase=True
    )

    tfidf.fit(sample_train)
    vocab_size = tfidf.get_vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")

    # Transform
    train_features, train_names = tfidf.transform(sample_train)
    test_features, test_names = tfidf.transform(sample_test)

    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    # Verify shapes
    assert train_features.shape[0] == len(sample_train), "Train feature count mismatch"
    assert test_features.shape[0] == len(sample_test), "Test feature count mismatch"
    assert train_features.shape[1] == vocab_size, "Train feature dimension mismatch"
    assert test_features.shape[1] == vocab_size, "Test feature dimension mismatch"

    print("\n[OK] TF-IDF feature extraction test passed")

    # Test 4: End-to-End Pipeline
    print("\nTest 4: End-to-End Pipeline")
    print("-" * 40)

    raw_text = "  This is a RAW text with MULTIPLE spaces and Mixed Case.  "
    print(f"Raw text: {repr(raw_text)}")

    # Preprocess
    processed = preprocessor.preprocess(raw_text, lowercase=True)
    print(f"Processed: {repr(processed)}")

    # Add to training data and fit
    train_data = sample_train + [processed]
    tfidf_full = TFIDFFeatureExtractor(ngram_range=(1, 2), min_df=1, max_features=100)
    tfidf_full.fit(train_data)

    # Transform
    features, names = tfidf_full.transform([processed])
    print(f"Feature shape: {features.shape}")
    print(f"Non-zero features: {features.nnz}")

    assert features.shape == (1, tfidf_full.get_vocabulary_size()), "End-to-end pipeline failed"

    print("\n[OK] End-to-end pipeline test passed")

    # Test 5: Load Saved Data
    print("\nTest 5: Load Saved Processed Data")
    print("-" * 40)

    try:
        with open('../data/processed/train_processed.pkl', 'rb') as f:
            train_data_loaded = pickle.load(f)

        print(f"Loaded train data keys: {train_data_loaded.keys()}")
        print(f"Number of texts: {len(train_data_loaded['texts'])}")
        print(f"Number of labels (six-class): {len(train_data_loaded['labels_six'])}")
        print(f"Number of labels (binary): {len(train_data_loaded['labels_binary'])}")
        print(f"Feature matrix shape: {train_data_loaded['features'].shape}")

        assert len(train_data_loaded['texts']) == len(train_data_loaded['labels_binary']), "Data length mismatch"
        assert train_data_loaded['features'].shape[0] == len(train_data_loaded['texts']), "Feature count mismatch"

        print("\n[OK] Loaded data verification passed")

    except FileNotFoundError:
        print("[SKIP] Processed data not found (run process_liar_dataset.py first)")

    # Test 6: Load Saved Vectorizer
    print("\nTest 6: Load Saved Vectorizer")
    print("-" * 40)

    try:
        tfidf_loaded = TFIDFFeatureExtractor()
        tfidf_loaded.load_vectorizer('../data/processed/tfidf_vectorizer.pkl')

        vocab_size_loaded = tfidf_loaded.get_vocabulary_size()
        print(f"Loaded vocabulary size: {vocab_size_loaded}")

        # Test transformation with loaded vectorizer
        test_features, _ = tfidf_loaded.transform(["test statement"])
        print(f"Test transformation shape: {test_features.shape}")

        assert test_features.shape[1] == vocab_size_loaded, "Loaded vectorizer transformation failed"

        print("\n[OK] Vectorizer loading test passed")

    except FileNotFoundError:
        print("[SKIP] Vectorizer not found (run process_liar_dataset.py first)")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)

if __name__ == '__main__':
    test_preprocessing_pipeline()
