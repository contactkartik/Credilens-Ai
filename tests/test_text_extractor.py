"""
Test script for text extraction service.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.text_extractor import TextExtractor

def test_text_extractor():
    """Test text extraction from various file formats."""
    print("="*60)
    print("TESTING TEXT EXTRACTION SERVICE")
    print("="*60)

    extractor = TextExtractor()

    # Test 1: TXT extraction
    print("\nTest 1: TXT Extraction")
    print("-" * 40)
    txt_path = 'test_samples/sample.txt'
    result = extractor.extract_text(txt_path)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Format: {result['format']}")
        print(f"Length: {result['length']} characters")
        print(f"Text preview: {result['text'][:100]}...")
        print("[OK] TXT extraction successful")
    else:
        print(f"Error: {result['error']}")
        print("[FAIL] TXT extraction failed")

    # Test 2: DOCX extraction
    print("\nTest 2: DOCX Extraction")
    print("-" * 40)
    docx_path = 'test_samples/sample.docx'
    result = extractor.extract_text(docx_path)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Format: {result['format']}")
        print(f"Length: {result['length']} characters")
        print(f"Text preview: {result['text'][:100]}...")
        print("[OK] DOCX extraction successful")
    else:
        print(f"Error: {result['error']}")
        print("[FAIL] DOCX extraction failed")

    # Test 3: PDF extraction
    print("\nTest 3: PDF Extraction")
    print("-" * 40)
    pdf_path = 'test_samples/sample.pdf'
    result = extractor.extract_text(pdf_path)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Format: {result['format']}")
        print(f"Length: {result['length']} characters")
        print(f"Text preview: {result['text'][:100]}...")
        print("[OK] PDF extraction successful")
    else:
        print(f"Error: {result['error']}")
        print("[FAIL] PDF extraction failed")

    # Test 4: Image OCR extraction
    print("\nTest 4: Image OCR Extraction")
    print("-" * 40)
    img_path = 'test_samples/sample.png'
    result = extractor.extract_text(img_path)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Format: {result['format']}")
        print(f"Length: {result['length']} characters")
        print(f"Text preview: {result['text'][:100]}...")
        print("[OK] Image OCR extraction successful")
    else:
        print(f"Error: {result['error']}")
        print("[FAIL] Image OCR extraction failed")

    # Test 5: Non-existent file
    print("\nTest 5: Non-existent File")
    print("-" * 40)
    fake_path = 'test_samples/nonexistent.txt'
    result = extractor.extract_text(fake_path)
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    if not result['success'] and 'not found' in result['error']:
        print("[OK] Non-existent file handled correctly")
    else:
        print("[FAIL] Non-existent file handling failed")

    # Test 6: Unsupported format
    print("\nTest 6: Unsupported Format")
    print("-" * 40)
    # Create a dummy file with unsupported extension
    with open('test_samples/sample.xyz', 'w') as f:
        f.write("dummy content")
    unsupported_path = 'test_samples/sample.xyz'
    result = extractor.extract_text(unsupported_path)
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    if not result['success'] and 'Unsupported' in result['error']:
        print("[OK] Unsupported format handled correctly")
    else:
        print("[FAIL] Unsupported format handling failed")
    # Clean up
    os.remove(unsupported_path)

    print("\n" + "="*60)
    print("TEXT EXTRACTION TESTS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    test_text_extractor()
