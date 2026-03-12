import uuid
import logging
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.ml.fingerprint_model import FingerprintProcessor
from app.ml.embeddings import EmbeddingStore
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FingerprintService:
    """Orchestrates fingerprint registration and verification."""

    def __init__(self, processor: FingerprintProcessor, embedding_store: EmbeddingStore):
        self.processor = processor
        self.store = embedding_store

    async def register_fingerprint(
        self, user_id: str, image: np.ndarray, db: AsyncSession
    ) -> dict:
        """Register a fingerprint for a user."""
        try:
            embedding = self.processor.get_embedding(image)
        except Exception as e:
            logger.error(f"Fingerprint processing failed: {e}")
            return {"success": False, "message": f"Could not process fingerprint: {e}", "embedding_id": None}

        fp_index = self.store.fingerprint_index
        internal_id = fp_index.add(user_id, embedding)

        result = await db.execute(select(User).where(User.user_id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if user:
            user.fingerprint_enrolled = True
            user.fingerprint_template_id = str(internal_id)

        logger.info(f"Fingerprint registered for user {user_id}")
        return {
            "success": True,
            "message": "Fingerprint enrolled successfully",
            "embedding_id": str(internal_id),
        }

    async def verify_fingerprint(self, user_id: str, image: np.ndarray) -> dict:
        """Verify fingerprint against stored template."""
        try:
            embedding = self.processor.get_embedding(image)
        except Exception as e:
            return {
                "match": False,
                "score": 0.0,
                "threshold": settings.FINGERPRINT_MATCH_THRESHOLD,
                "message": f"Could not process fingerprint: {e}",
            }

        stored = self.store.fingerprint_index.get_user_embedding(user_id)
        if stored is None:
            return {
                "match": False,
                "score": 0.0,
                "threshold": settings.FINGERPRINT_MATCH_THRESHOLD,
                "message": "No fingerprint enrolled for this user",
            }

        similarity = self.processor.compute_similarity(embedding, stored)
        is_match = similarity >= settings.FINGERPRINT_MATCH_THRESHOLD

        return {
            "match": is_match,
            "score": float(similarity),
            "threshold": settings.FINGERPRINT_MATCH_THRESHOLD,
            "message": "Fingerprint verified" if is_match else "Fingerprint does not match",
        }

    async def identify_fingerprint(self, image: np.ndarray) -> dict:
        """Identify a fingerprint from the database (1:N search)."""
        try:
            embedding = self.processor.get_embedding(image)
        except Exception as e:
            return {"found": False, "user_id": None, "score": 0.0, "message": f"Could not process fingerprint: {e}"}

        results = self.store.fingerprint_index.search(embedding, k=1)
        if not results:
            return {"found": False, "user_id": None, "score": 0.0, "message": "No matching fingerprint found"}

        best = results[0]
        is_match = best["similarity"] >= settings.FINGERPRINT_MATCH_THRESHOLD

        return {
            "found": is_match,
            "user_id": best["user_id"] if is_match else None,
            "score": best["similarity"],
            "message": "Identity found" if is_match else "No confident match",
        }
