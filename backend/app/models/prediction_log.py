"""
ORM model for stored credibility predictions (prediction history).
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PredictionLog(Base):
    """One row per successful /api/predict or /api/predict/file call."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    source: Mapped[str] = mapped_column(String(16))  # 'text' or 'file'
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    prediction: Mapped[str] = mapped_column(String(32))
    credibility_category: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSON)
    model: Mapped[str] = mapped_column(String(128))
