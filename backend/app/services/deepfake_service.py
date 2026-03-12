import logging
import numpy as np
from app.ml.deepfake_model import DeepfakeDetector

logger = logging.getLogger(__name__)


class DeepfakeService:
    """Orchestrates deepfake detection on images and video."""

    def __init__(self, detector: DeepfakeDetector):
        self.detector = detector

    async def detect_image(self, image: np.ndarray) -> dict:
        """Check a single image for deepfake indicators."""
        result = self.detector.predict(image)
        logger.info(f"Deepfake check: is_deepfake={result['is_deepfake']}, confidence={result['confidence']:.3f}")
        return {
            "is_deepfake": result["is_deepfake"],
            "confidence": result["confidence"],
            "method": result["method"],
            "message": "Deepfake detected" if result["is_deepfake"] else "Image appears authentic",
        }

    async def detect_video(self, frames: list[np.ndarray]) -> dict:
        """Check video frames for deepfake indicators."""
        result = self.detector.detect_video(frames)
        return {
            "is_deepfake": result["is_deepfake"],
            "confidence": result["confidence"],
            "method": result["method"],
            "message": "Deepfake detected in video" if result["is_deepfake"] else "Video appears authentic",
        }
