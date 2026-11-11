"""
Credibility prediction service.

Loads the fine-tuned RoBERTa model once and exposes a single prediction
method used by the API. The actual inference is delegated to the existing
ML pipeline code in ``ml/inference/predictor.py`` (Stage 5) - no
predictions are hardcoded here.
"""
import sys
import threading
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings

# Repository root (backend/app/services/ -> backend/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the model is available."""


class CredibilityService:
    """Service wrapper around the trained RoBERTa predictor."""

    # Display names for the model's label set (label mapping only - the
    # prediction itself always comes from the trained model).
    CREDIBILITY_CATEGORIES = {
        "true_side": "Credible",
        "false_side": "Not Credible",
    }

    def __init__(self, model_path: str):
        self.model_path = self._resolve_path(model_path)
        self._predictor = None
        self._load_error: str = None
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_path(model_path: str) -> str:
        """Resolve a possibly relative model path against the repo root."""
        path = Path(model_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path.resolve())

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def load_error(self) -> str:
        return self._load_error

    def load(self) -> bool:
        """
        Load the fine-tuned RoBERTa model (idempotent, thread-safe).

        Returns:
            True if the model is available, False otherwise.
        """
        with self._lock:
            if self._predictor is not None:
                return True
            try:
                # Imported lazily so the API app can start (and report a
                # clean 503) even if the ML dependencies are unavailable.
                from ml.inference.predictor import RoBERTaPredictor

                predictor = RoBERTaPredictor(model_path=self.model_path)
                predictor.load_model()
                self._predictor = predictor
                self._load_error = None
                return True
            except Exception as exc:  # pragma: no cover - failure path
                self._load_error = f"{type(exc).__name__}: {exc}"
                return False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the credibility of a statement using the trained model.

        Args:
            text: Non-empty statement to evaluate.

        Returns:
            Dictionary with prediction, credibility category, confidence
            and per-class probabilities.

        Raises:
            ModelNotLoadedError: If the model could not be loaded.
        """
        if self._predictor is None:
            if not self.load():
                raise ModelNotLoadedError(
                    f"Model failed to load from '{self.model_path}': {self._load_error}"
                )

        result = self._predictor.predict(text)
        prediction = result["prediction"]

        return {
            "text": text,
            "prediction": prediction,
            "credibility_category": self.CREDIBILITY_CATEGORIES.get(
                prediction, "Unknown"
            ),
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "model": "roberta-base (fine-tuned)",
        }


_service: CredibilityService = None


def get_credibility_service() -> CredibilityService:
    """Return the process-wide credibility service singleton."""
    global _service
    if _service is None:
        _service = CredibilityService(model_path=settings.ROBERTA_MODEL_PATH)
    return _service
