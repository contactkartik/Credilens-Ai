"""
API router for the backend application.
Routes are added here as features are implemented.
"""
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.database import get_db
from app.models.prediction_log import PredictionLog
from app.schemas.prediction import (
    PredictionHistoryResponse,
    PredictionLogOut,
    PredictionRequest,
    PredictionResponse,
)
from app.services.credibility_service import (
    ModelNotLoadedError,
    get_credibility_service,
)
from app.services.text_extractor import TextExtractor

logger = logging.getLogger("credilens")

router = APIRouter()

# File-upload pipeline (Stage 7): file -> text extraction/OCR -> RoBERTa
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.png', '.jpg', '.jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Stateless service instance (the PaddleOCR instance inside it is cached)
_text_extractor = TextExtractor()


class ExtractionError(RuntimeError):
    """Raised when no usable text can be extracted from an upload."""


def _extract_text_from_upload(suffix: str, data: bytes) -> str:
    """
    Save an uploaded file to a temp path, extract its text, clean up.

    Args:
        suffix: Allowed file extension (including dot), e.g. '.pdf'.
        data: Raw file bytes.

    Returns:
        Extracted, stripped non-empty text.

    Raises:
        ExtractionError: If extraction fails or yields no text.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        extraction = _text_extractor.extract_text(tmp_path)
    except Exception as exc:
        raise ExtractionError(f"{type(exc).__name__}: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not extraction.get('success'):
        raise ExtractionError(extraction.get('error') or 'Text extraction failed')

    text = (extraction.get('text') or '').strip()
    if not text:
        raise ExtractionError('No text could be extracted from the file')
    return text


def _run_prediction(text: str) -> dict:
    """Call the trained RoBERTa model (runs in a worker thread)."""
    return get_credibility_service().predict(text)


def _save_prediction(
    db: Session, result: dict, source: str, filename: str | None = None
) -> None:
    """
    Persist a prediction to the database (best-effort).

    A database failure must never break the prediction endpoint, so any
    error is logged and swallowed.
    """
    try:
        db.add(
            PredictionLog(
                source=source,
                filename=filename,
                text=result.get('text', ''),
                prediction=result.get('prediction', ''),
                credibility_category=result.get('credibility_category', ''),
                confidence=result.get('confidence', 0.0),
                probabilities=result.get('probabilities', {}),
                model=result.get('model', ''),
            )
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Failed to save prediction to database: %s", exc)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    service = get_credibility_service()
    return {
        "status": "healthy",
        "model_loaded": service.is_loaded,
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_credibility(
    request: PredictionRequest, db: Session = Depends(get_db)
):
    """
    Predict the credibility of a statement using the fine-tuned RoBERTa model.

    - Returns 400 if the text is empty or only whitespace.
    - Returns 503 if the model could not be loaded.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: 'text' must be a non-empty string.",
        )

    try:
        result = await run_in_threadpool(_run_prediction, text)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        )

    _save_prediction(db, result, source='text')
    return PredictionResponse(**result)


@router.get("/history", response_model=PredictionHistoryResponse)
async def prediction_history(limit: int = 50, db: Session = Depends(get_db)):
    """
    Return the most recent stored predictions, newest first.

    - `limit` caps the number of results (default 50, max 200).
    - Returns 503 if the database is unreachable.
    """
    limit = max(1, min(limit, 200))
    try:
        rows = db.scalars(
            select(PredictionLog)
            .order_by(PredictionLog.created_at.desc(), PredictionLog.id.desc())
            .limit(limit)
        ).all()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {type(exc).__name__}: {exc}",
        )
    return PredictionHistoryResponse(
        count=len(rows), results=[PredictionLogOut.model_validate(r) for r in rows]
    )


@router.post("/predict/file", response_model=PredictionResponse)
async def predict_credibility_from_file(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """
    Predict credibility of a TXT, PDF, DOCX or PNG/JPG/JPEG upload.

    Pipeline: file -> text extraction (PaddleOCR for images) ->
    fine-tuned RoBERTa model -> same prediction response as /api/predict.

    - Returns 400 for an empty file.
    - Returns 413 if the file exceeds the size limit.
    - Returns 415 for unsupported file types.
    - Returns 422 if no text can be extracted.
    - Returns 503 if the model could not be loaded.
    """
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{suffix or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: uploaded file is empty.",
        )
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: maximum size is {MAX_FILE_SIZE} bytes.",
        )

    # Extraction (incl. OCR) and model inference are CPU-heavy - run the
    # synchronous work in a worker thread so the event loop stays free.
    try:
        text = await run_in_threadpool(_extract_text_from_upload, suffix, data)
    except ExtractionError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from file: {exc}",
        )

    try:
        result = await run_in_threadpool(_run_prediction, text)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        )

    _save_prediction(db, result, source='file', filename=file.filename)
    return PredictionResponse(**result)
