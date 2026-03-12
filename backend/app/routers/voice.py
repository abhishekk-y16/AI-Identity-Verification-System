from fastapi import APIRouter, Depends, UploadFile, File, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.schemas import BiometricRegisterResponse, BiometricVerifyResponse
from app.utils.security import get_current_user
from app.utils.audio_utils import read_audio_from_upload, validate_audio_file

router = APIRouter(prefix="/api/voice", tags=["Voice Authentication"])


def get_voice_service(request: Request):
    service = request.app.state.voice_service
    if service is None:
        raise HTTPException(status_code=503, detail="Voice service is not available. SpeechBrain model failed to load.")
    return service


@router.post("/register", response_model=BiometricRegisterResponse)
async def register_voice(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
):
    """Register voice biometric for the authenticated user."""
    validate_audio_file(file)
    audio_bytes = await read_audio_from_upload(file)
    voice_service = get_voice_service(request)
    result = await voice_service.register_voice(current_user["user_id"], audio_bytes, db)
    return BiometricRegisterResponse(**result)


@router.post("/verify", response_model=BiometricVerifyResponse)
async def verify_voice(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Verify voice against enrolled voiceprint."""
    validate_audio_file(file)
    audio_bytes = await read_audio_from_upload(file)
    voice_service = get_voice_service(request)
    result = await voice_service.verify_voice(current_user["user_id"], audio_bytes)
    return BiometricVerifyResponse(**result)
