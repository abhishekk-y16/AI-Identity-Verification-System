import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import init_db, close_db

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — init biometric models + office services on startup."""
    logger.info("🚀 Starting Office Attendance & Identity System...")

    # Create required directories
    os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

    # Initialize database (import models first so tables are registered)
    import app.models as _models  # noqa: F401
    await init_db()
    logger.info("✅ Database initialized")

    # Initialize ML models (face + voice only — used for biometric clock-in)
    logger.info("Loading biometric models...")

    from app.ml.face_model import FaceEmbedder
    from app.ml.embeddings import EmbeddingStore

    face_embedder = FaceEmbedder()
    logger.info("✅ Face model loaded")

    embedding_store = EmbeddingStore(settings.FAISS_INDEX_DIR)
    logger.info("✅ Embedding store loaded")

    voice_embedder = None
    try:
        from app.ml.voice_model import VoiceEmbedder
        voice_embedder = VoiceEmbedder()
        logger.info("✅ Voice model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Voice model not available: {e}")

    # Initialize biometric services (used internally by attendance)
    from app.services.face_service import FaceService
    from app.services.voice_service import VoiceService
    from app.services.attendance_service import AttendanceService

    app.state.face_service = FaceService(face_embedder, embedding_store)
    app.state.voice_service = VoiceService(voice_embedder, embedding_store) if voice_embedder else None
    app.state.attendance_service = AttendanceService(
        app.state.face_service, app.state.voice_service
    )

    logger.info("🎯 All services initialized — system ready!")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    logger.info("Goodbye!")


# ─── Create FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Enterprise Office Attendance & Identity System. "
        "Face + voice biometric clock-in/out, attendance tracking, "
        "leave management, department & shift management, reports & analytics."
    ),
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routers ───────────────────────────────────────────────────────
from app.routers import (
    auth,
    face,
    attendance,
    departments,
    shifts,
    employees,
    leaves,
    alerts,
    reports,
    office_dashboard,
)

app.include_router(auth.router)
app.include_router(face.router)
app.include_router(attendance.router)
app.include_router(departments.router)
app.include_router(shifts.router)
app.include_router(employees.router)
app.include_router(leaves.router)
app.include_router(alerts.router)
app.include_router(reports.router)
app.include_router(office_dashboard.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "face": app.state.face_service is not None,
            "voice": app.state.voice_service is not None,
        },
    }
