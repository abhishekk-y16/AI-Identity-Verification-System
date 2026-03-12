import uuid
import logging
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.ml.voice_model import VoiceEmbedder
from app.ml.embeddings import EmbeddingStore
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VoiceService:
    """Orchestrates voice registration and speaker verification."""

    def __init__(self, voice_embedder: VoiceEmbedder, embedding_store: EmbeddingStore):
        self.embedder = voice_embedder
        self.store = embedding_store

    async def register_voice(
        self, user_id: str, audio_bytes: bytes, db: AsyncSession
    ) -> dict:
        """Register a voiceprint for a user."""
        try:
            embedding = self.embedder.extract_embedding(audio_bytes)
        except Exception as e:
            logger.error(f"Voice embedding extraction failed: {e}")
            return {"success": False, "message": f"Could not process audio: {e}", "embedding_id": None}

        voice_index = self.store.voice_index
        internal_id = voice_index.add(user_id, embedding)

        result = await db.execute(select(User).where(User.user_id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if user:
            user.voice_enrolled = True
            user.voice_embedding_id = str(internal_id)

        logger.info(f"Voice registered for user {user_id}")
        return {
            "success": True,
            "message": "Voice enrolled successfully",
            "embedding_id": str(internal_id),
        }

    async def verify_voice(self, user_id: str, audio_bytes: bytes) -> dict:
        """Verify voice against stored voiceprint for a specific user."""
        try:
            embedding = self.embedder.extract_embedding(audio_bytes)
        except Exception as e:
            return {
                "match": False,
                "score": 0.0,
                "threshold": settings.VOICE_MATCH_THRESHOLD,
                "message": f"Could not process audio: {e}",
            }

        stored = self.store.voice_index.get_user_embedding(user_id)
        if stored is None:
            return {
                "match": False,
                "score": 0.0,
                "threshold": settings.VOICE_MATCH_THRESHOLD,
                "message": "No voice enrolled for this user",
            }

        similarity = self.embedder.compute_similarity(embedding, stored)
        is_match = similarity >= settings.VOICE_MATCH_THRESHOLD

        return {
            "match": is_match,
            "score": float(similarity),
            "threshold": settings.VOICE_MATCH_THRESHOLD,
            "message": "Voice verified" if is_match else "Voice does not match",
        }

    async def identify_voice(self, audio_bytes: bytes) -> dict:
        """Identify a speaker from the database (1:N search)."""
        try:
            embedding = self.embedder.extract_embedding(audio_bytes)
        except Exception as e:
            return {"found": False, "user_id": None, "score": 0.0, "message": f"Could not process audio: {e}"}

        results = self.store.voice_index.search(embedding, k=1)
        if not results:
            return {"found": False, "user_id": None, "score": 0.0, "message": "No matching voice found"}

        best = results[0]
        is_match = best["similarity"] >= settings.VOICE_MATCH_THRESHOLD

        return {
            "found": is_match,
            "user_id": best["user_id"] if is_match else None,
            "score": best["similarity"],
            "message": "Identity found" if is_match else "No confident match",
        }
