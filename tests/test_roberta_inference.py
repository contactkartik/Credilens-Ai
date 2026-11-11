"""
Test script to verify RoBERTa model loading and inference.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference.predictor import RoBERTaPredictor

def test_roberta_inference():
    """Test RoBERTa model loading and prediction."""
    print("="*60)
    print("TESTING ROBERTA MODEL INFERENCE")
    print("="*60)

    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = RoBERTaPredictor(model_path='../models/roberta')

    # Load model
    print("\nLoading model...")
    try:
        predictor.load_model()
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return

    # Test single prediction
    print("\nTest 1: Single prediction")
    print("-" * 40)

    test_text = "This is a test statement for credibility detection."
    print(f"Input: {test_text}")

    try:
        result = predictor.predict(test_text)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities:")
        print(f"  false_side: {result['probabilities']['false_side']:.4f}")
        print(f"  true_side: {result['probabilities']['true_side']:.4f}")
        print("[OK] Single prediction successful")
    except Exception as e:
        print(f"[FAIL] Single prediction failed: {e}")
        return

    # Test batch prediction
    print("\nTest 2: Batch prediction")
    print("-" * 40)

    test_texts = [
        "The government is planning to increase taxes next year.",
        "Studies show that exercise improves health significantly.",
        "Politicians never tell the truth about their policies."
    ]

    print(f"Input: {len(test_texts)} texts")

    try:
        results = predictor.predict(test_texts)
        for i, result in enumerate(results):
            print(f"\nText {i+1}: {result['text'][:50]}...")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.4f}")
        print("[OK] Batch prediction successful")
    except Exception as e:
        print(f"[FAIL] Batch prediction failed: {e}")
        return

    # Test with actual LIAR sample
    print("\nTest 3: Prediction on LIAR sample")
    print("-" * 40)

    liar_sample = "Says the Annies List political group supports third-trimester abortions on demand."
    print(f"Input: {liar_sample}")

    try:
        result = predictor.predict(liar_sample)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities:")
        print(f"  false_side: {result['probabilities']['false_side']:.4f}")
        print(f"  true_side: {result['probabilities']['true_side']:.4f}")
        print("[OK] LIAR sample prediction successful")
    except Exception as e:
        print(f"[FAIL] LIAR sample prediction failed: {e}")
        return

    print("\n" + "="*60)
    print("ALL INFERENCE TESTS PASSED")
    print("="*60)

if __name__ == '__main__':
    test_roberta_inference()
