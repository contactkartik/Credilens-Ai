"""
API tests for POST /api/predict/file (Stage 7).

Covers TXT, PDF, DOCX and PNG/JPG/JPEG (OCR) uploads through the real
extraction + trained RoBERTa pipeline - predictions are never hardcoded.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / 'tests' / 'test_samples'


@pytest.fixture(scope="session")
def client():
    """Test client that triggers the app lifespan (model warm-up)."""
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, filename, data, content_type='application/octet-stream'):
    """POST a file to the upload endpoint."""
    return client.post(
        '/api/predict/file',
        files={'file': (filename, data, content_type)},
    )


def _assert_valid_prediction(body, expect_in_text=None):
    """The response must follow the standard prediction contract."""
    assert body['prediction'] in ('true_side', 'false_side')
    expected = 'Credible' if body['prediction'] == 'true_side' else 'Not Credible'
    assert body['credibility_category'] == expected
    assert 0.0 <= body['confidence'] <= 1.0
    assert set(body['probabilities']) == {'false_side', 'true_side'}
    assert abs(sum(body['probabilities'].values()) - 1.0) < 1e-4
    if expect_in_text is not None:
        assert expect_in_text.lower() in body['text'].lower()


# ---------------------------------------------------------------------------
# One test per supported input type
# ---------------------------------------------------------------------------

def test_txt_upload(client):
    data = (SAMPLES_DIR / 'sample.txt').read_bytes()
    response = _upload(client, 'sample.txt', data, 'text/plain')

    assert response.status_code == 200
    _assert_valid_prediction(response.json(), expect_in_text='sample text file')


def test_pdf_upload(client):
    data = (SAMPLES_DIR / 'sample.pdf').read_bytes()
    response = _upload(client, 'sample.pdf', data, 'application/pdf')

    assert response.status_code == 200
    _assert_valid_prediction(
        response.json(), expect_in_text='Sample PDF Document'
    )


def test_docx_upload(client):
    data = (SAMPLES_DIR / 'sample.docx').read_bytes()
    response = _upload(
        client, 'sample.docx', data,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    )

    assert response.status_code == 200
    _assert_valid_prediction(
        response.json(), expect_in_text='Sample Document'
    )


def test_png_upload_ocr(client):
    """Screenshots go through PaddleOCR before the RoBERTa model."""
    data = (SAMPLES_DIR / 'sample.png').read_bytes()
    response = _upload(client, 'sample.png', data, 'image/png')

    assert response.status_code == 200
    _assert_valid_prediction(
        response.json(), expect_in_text='Sample Image for OCR Testing'
    )


def test_jpg_upload_ocr(client, tmp_path):
    """JPEG conversion of the same screenshot also goes through OCR."""
    img = Image.open(SAMPLES_DIR / 'sample.png').convert('RGB')
    jpg_path = tmp_path / 'sample.jpg'
    img.save(jpg_path, 'JPEG')

    response = _upload(
        client, 'sample.jpg', jpg_path.read_bytes(), 'image/jpeg'
    )

    assert response.status_code == 200
    _assert_valid_prediction(
        response.json(), expect_in_text='Sample Image for OCR Testing'
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_unsupported_file_type(client):
    response = _upload(client, 'malware.exe', b'not a real file')
    assert response.status_code == 415
    assert 'Unsupported file type' in response.json()['detail']


def test_rejects_empty_file(client):
    response = _upload(client, 'empty.txt', b'')
    assert response.status_code == 400
    assert 'empty' in response.json()['detail'].lower()


def test_rejects_file_with_no_extractable_text(client):
    response = _upload(client, 'blank.txt', b'   \n\t   \n  ')
    assert response.status_code == 422
    assert 'Could not extract text' in response.json()['detail']


def test_rejects_oversized_file(client):
    from app.api.router import MAX_FILE_SIZE

    response = _upload(client, 'huge.txt', b'a' * (MAX_FILE_SIZE + 1))
    assert response.status_code == 413


def test_rejects_missing_file_field(client):
    response = client.post('/api/predict/file')
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Regression: the original text endpoint must still work
# ---------------------------------------------------------------------------

def test_text_predict_endpoint_still_works(client):
    response = client.post(
        '/api/predict',
        json={'text': 'Says the Annies List political group supports '
                      'third-trimester abortions on demand.'},
    )
    assert response.status_code == 200
    _assert_valid_prediction(response.json())
