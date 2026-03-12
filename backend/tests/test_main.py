import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
import numpy as np


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _create_mock_app():
    """Create a FastAPI app with mocked ML models for testing."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Test App")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mock services on app.state
    app.state.face_service = MagicMock()
    app.state.voice_service = MagicMock()
    app.state.fingerprint_service = MagicMock()
    app.state.liveness_service = MagicMock()
    app.state.deepfake_service = MagicMock()
    app.state.document_service = MagicMock()
    app.state.fraud_service = MagicMock()
    app.state.fusion_service = MagicMock()
    app.state.behavioral_service = MagicMock()

    # Register routers
    from app.routers import auth, face, voice, fingerprint, liveness, deepfake, document, fraud, verification, dashboard
    app.include_router(auth.router)
    app.include_router(face.router)
    app.include_router(voice.router)
    app.include_router(fingerprint.router)
    app.include_router(liveness.router)
    app.include_router(deepfake.router)
    app.include_router(document.router)
    app.include_router(fraud.router)
    app.include_router(verification.router)
    app.include_router(dashboard.router)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


# ──────────────────────────────────────────────────────────────
# Unit Tests — Security Utilities
# ──────────────────────────────────────────────────────────────

class TestSecurity:
    def test_hash_and_verify_password(self):
        from app.utils.security import hash_password, verify_password
        hashed = hash_password("testpass123")
        assert hashed != "testpass123"
        assert verify_password("testpass123", hashed) is True
        assert verify_password("wrongpass", hashed) is False

    def test_create_and_decode_token(self):
        from app.utils.security import create_access_token, decode_access_token
        token = create_access_token({"sub": "user123", "role": "user"})
        payload = decode_access_token(token)
        assert payload["sub"] == "user123"
        assert payload["role"] == "user"

    def test_token_has_expiry(self):
        from app.utils.security import create_access_token, decode_access_token
        token = create_access_token({"sub": "u1"})
        payload = decode_access_token(token)
        assert "exp" in payload

    def test_biometric_tokenization(self):
        from app.utils.security import generate_biometric_seed, tokenize_embedding
        seed = generate_biometric_seed()
        embedding = np.random.randn(512).astype(np.float32)
        tokenized = tokenize_embedding(embedding, seed)
        assert tokenized.shape == embedding.shape
        # Should not be the same as original
        assert not np.allclose(tokenized, embedding)

    def test_embedding_hex_roundtrip(self):
        from app.utils.security import embedding_to_hex, hex_to_embedding
        original = np.random.randn(128).astype(np.float32)
        hex_str = embedding_to_hex(original)
        recovered = hex_to_embedding(hex_str)
        np.testing.assert_array_almost_equal(original, recovered)


# ──────────────────────────────────────────────────────────────
# Unit Tests — Image Utilities
# ──────────────────────────────────────────────────────────────

class TestImageUtils:
    def test_decode_encode_roundtrip(self):
        from app.utils.image_utils import decode_image_base64, encode_image_to_base64
        import cv2
        # Create a small test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [255, 0, 0]
        _, buf = cv2.imencode(".png", img)
        import base64
        b64 = base64.b64encode(buf.tobytes()).decode()
        decoded = decode_image_base64(b64)
        assert decoded is not None
        assert decoded.shape[0] == 100
        assert decoded.shape[1] == 100

    def test_resize_image(self):
        from app.utils.image_utils import resize_image
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        resized = resize_image(img, 100, 100)
        assert resized.shape[0] == 100
        assert resized.shape[1] == 100

    def test_validate_image_file(self):
        from app.utils.image_utils import validate_image_file
        # Valid image types
        assert validate_image_file("photo.jpg", 5_000_000) is True
        assert validate_image_file("photo.png", 5_000_000) is True
        # Invalid ext
        assert validate_image_file("file.exe", 5_000_000) is False
        # Too large (>10MB default)
        assert validate_image_file("photo.jpg", 15_000_000) is False


# ──────────────────────────────────────────────────────────────
# Unit Tests — Config
# ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_load(self):
        from app.config import get_settings
        settings = get_settings()
        assert settings.APP_NAME == "AI Identity Verification System"
        assert settings.FACE_MATCH_THRESHOLD > 0
        assert settings.FUSION_WEIGHT_FACE > 0

    def test_thresholds_valid(self):
        from app.config import get_settings
        s = get_settings()
        assert 0 < s.FACE_MATCH_THRESHOLD <= 1
        assert 0 < s.VOICE_MATCH_THRESHOLD <= 1
        assert 0 < s.FINGERPRINT_MATCH_THRESHOLD <= 1


# ──────────────────────────────────────────────────────────────
# Unit Tests — Schemas
# ──────────────────────────────────────────────────────────────

class TestSchemas:
    def test_user_create_validation(self):
        from app.schemas.schemas import UserCreate
        user = UserCreate(name="Test", email="test@example.com", password="password123")
        assert user.email == "test@example.com"

    def test_user_create_invalid_email(self):
        from app.schemas.schemas import UserCreate
        with pytest.raises(Exception):
            UserCreate(name="Test", email="not-an-email", password="password123")


# ──────────────────────────────────────────────────────────────
# Unit Tests — Behavioral Service
# ──────────────────────────────────────────────────────────────

class TestBehavioralService:
    def test_extract_features(self):
        from app.services.behavioral_service import BehavioralService
        svc = BehavioralService()
        events = [
            {"type": "keystroke", "timestamp": 1000, "key_code": 65, "press_duration": 100},
            {"type": "keystroke", "timestamp": 1200, "key_code": 66, "press_duration": 110},
            {"type": "keystroke", "timestamp": 1400, "key_code": 67, "press_duration": 90},
            {"type": "mouse_move", "timestamp": 1000, "x": 0, "y": 0},
            {"type": "mouse_move", "timestamp": 1100, "x": 100, "y": 100},
        ]
        features = svc.extract_features(events)
        assert isinstance(features, dict)
        assert "keystroke_mean_interval" in features or len(features) > 0


# ──────────────────────────────────────────────────────────────
# Unit Tests — Fusion Service
# ──────────────────────────────────────────────────────────────

class TestFusionService:
    def test_compute_fusion(self):
        from app.services.fusion_service import FusionService
        svc = FusionService()
        result = svc.compute_fusion(
            face_score=0.9,
            voice_score=0.8,
            fingerprint_score=0.85,
        )
        assert "overall_score" in result
        assert "risk_level" in result
        assert 0 <= result["overall_score"] <= 1

    def test_single_modality_fusion(self):
        from app.services.fusion_service import FusionService
        svc = FusionService()
        result = svc.compute_fusion(face_score=0.7)
        assert result["overall_score"] > 0

    def test_low_score_high_risk(self):
        from app.services.fusion_service import FusionService
        svc = FusionService()
        result = svc.compute_fusion(face_score=0.1)
        assert result["risk_level"] in ("high", "critical")


# ──────────────────────────────────────────────────────────────
# Unit Tests — Fingerprint Processor (OpenCV pipeline)
# ──────────────────────────────────────────────────────────────

class TestFingerprintProcessor:
    def test_extract_embedding_from_blank(self):
        from app.ml.fingerprint_model import FingerprintProcessor
        proc = FingerprintProcessor(use_cnn=False)
        # A blank image should still return a 256-dim vector
        blank = np.zeros((300, 300), dtype=np.uint8)
        emb = proc.extract_embedding(blank)
        assert emb is not None
        assert emb.shape[0] == 256


# ──────────────────────────────────────────────────────────────
# Integration-style Tests — API Endpoints (with mocked DB/services)
# ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self):
        app = _create_mock_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"


class TestAuthEndpoints:
    @pytest.mark.asyncio
    async def test_register_missing_fields(self):
        app = _create_mock_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/auth/register", json={})
            assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_login_missing_fields(self):
        app = _create_mock_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/auth/login", json={})
            assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_me_unauthenticated(self):
        app = _create_mock_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/auth/me")
            assert resp.status_code == 401


class TestFaceEndpoints:
    @pytest.mark.asyncio
    async def test_register_unauthenticated(self):
        app = _create_mock_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/face/register")
            assert resp.status_code in (401, 422)

    @pytest.mark.asyncio
    async def test_verify_unauthenticated(self):
        app = _create_mock_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/face/verify")
            assert resp.status_code in (401, 422)


class TestDashboardEndpoints:
    @pytest.mark.asyncio
    async def test_stats_unauthenticated(self):
        app = _create_mock_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/dashboard/stats")
            assert resp.status_code == 401
