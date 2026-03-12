import logging
import numpy as np
from app.ml.liveness_model import LivenessDetector

logger = logging.getLogger(__name__)


class LivenessService:
    """Orchestrates face anti-spoofing checks."""

    def __init__(self, detector: LivenessDetector):
        self.detector = detector

    async def check_liveness(self, frames: list[np.ndarray]) -> dict:
        """Run full liveness detection on frame sequence.
        
        Args:
            frames: List of RGB numpy arrays (video frames or single image)
            
        Returns:
            Dict with is_live, confidence, checks, message
        """
        if not frames:
            return {
                "is_live": False,
                "confidence": 0.0,
                "checks": {},
                "message": "No frames provided for liveness check",
            }

        result = self.detector.check_liveness(frames)
        logger.info(f"Liveness check: is_live={result['is_live']}, confidence={result['confidence']:.3f}")
        return result

    async def check_single_image(self, image: np.ndarray) -> dict:
        """Quick liveness check on a single image (texture + CNN only)."""
        return self.detector.check_liveness([image])
