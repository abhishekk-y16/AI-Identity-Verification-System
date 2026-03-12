import uuid
import logging
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.ml.face_model import FaceEmbedder
from app.ml.embeddings import EmbeddingStore
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FaceService:
    """Orchestrates face registration, verification (1:1), and identification (1:N)."""

    def __init__(self, face_embedder: FaceEmbedder, embedding_store: EmbeddingStore):
        self.embedder = face_embedder
        self.store = embedding_store

    async def register_face(
        self, user_id: str, image: np.ndarray, db: AsyncSession
    ) -> dict:
        """Register a face for a user — extract embedding, store in FAISS + update DB."""
        embedding = self.embedder.extract_embedding(image)
        if embedding is None:
            return {"success": False, "message": "No face detected in the image", "embedding_id": None}

        # Store in FAISS
        face_index = self.store.face_index
        internal_id = face_index.add(user_id, embedding)

        # Update user record
        result = await db.execute(select(User).where(User.user_id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if user:
            user.face_enrolled = True
            user.face_embedding_id = str(internal_id)

        logger.info(f"Face registered for user {user_id}")
        return {
            "success": True,
            "message": "Face enrolled successfully",
            "embedding_id": str(internal_id),
        }

    async def verify_face(self, user_id: str, image: np.ndarray) -> dict:
        """Verify face against stored embedding for a specific user (1:1)."""
        embedding = self.embedder.extract_embedding(image)
        if embedding is None:
            return {
                "match": False,
                "score": 0.0,
                "threshold": settings.FACE_MATCH_THRESHOLD,
                "message": "No face detected in the image",
            }

        stored = self.store.face_index.get_user_embedding(user_id)
        if stored is None:
            return {
                "match": False,
                "score": 0.0,
                "threshold": settings.FACE_MATCH_THRESHOLD,
                "message": "No face enrolled for this user",
            }

        similarity = self.embedder.compute_similarity(embedding, stored)
        is_match = similarity >= settings.FACE_MATCH_THRESHOLD

        return {
            "match": is_match,
            "score": float(similarity),
            "threshold": settings.FACE_MATCH_THRESHOLD,
            "message": "Face verified" if is_match else "Face does not match",
        }

    async def identify_face(self, image: np.ndarray) -> dict:
        """Identify a face from the database (1:N search)."""
        embedding = self.embedder.extract_embedding(image)
        if embedding is None:
            return {"found": False, "user_id": None, "score": 0.0, "message": "No face detected"}

        results = self.store.face_index.search(embedding, k=1)
        if not results:
            return {"found": False, "user_id": None, "score": 0.0, "message": "No matching face found"}

        best = results[0]
        is_match = best["similarity"] >= settings.FACE_MATCH_THRESHOLD

        return {
            "found": is_match,
            "user_id": best["user_id"] if is_match else None,
            "score": best["similarity"],
            "message": "Identity found" if is_match else "No confident match",
        }
