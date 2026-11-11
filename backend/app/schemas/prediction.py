"""
Pydantic schemas for the credibility prediction endpoints.
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for POST /api/predict."""

    text: str = Field(
        ...,
        description="Statement to evaluate for credibility.",
        examples=["Says the Annies List political group supports third-trimester abortions on demand."],
    )


class PredictionResponse(BaseModel):
    """Response body returned by POST /api/predict."""

    text: str
    prediction: str = Field(
        ...,
        description="Raw model label: 'true_side' or 'false_side'.",
    )
    credibility_category: str = Field(
        ...,
        description="Human-readable credibility category derived from the model label.",
    )
    confidence: float = Field(
        ...,
        description="Model confidence for the predicted class (0.0 - 1.0).",
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Per-class probabilities returned by the model.",
    )
    model: str = Field(
        ...,
        description="Name of the model that produced the prediction.",
    )


class PredictionLogOut(BaseModel):
    """A stored prediction returned by GET /api/history."""

    id: int
    created_at: datetime
    source: str
    filename: Optional[str] = None
    text: str
    prediction: str
    credibility_category: str
    confidence: float
    probabilities: Dict[str, float]
    model: str

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    """Response body for GET /api/history."""

    count: int
    results: List[PredictionLogOut]
