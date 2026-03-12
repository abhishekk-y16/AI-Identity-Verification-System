import uuid
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from fastapi import Depends
from app.models.user import User
from app.schemas.schemas import IdentifyResponse
from app.utils.image_utils import read_image_from_upload, validate_image_file
from app.utils.audio_utils import read_audio_from_upload, validate_audio_file

router = APIRouter(prefix="/api/identify", tags=["Public Identification (1:N)"])


async def _lookup_user(user_id: str, db: AsyncSession) -> dict:
    """Look up user name and email from the database."""
    result = await db.execute(
        select(User.name, User.email).where(User.user_id == uuid.UUID(user_id))
    )
    row = result.one_or_none()
    if row:
        return {"user_name": row.name, "user_email": row.email}
    return {"user_name": None, "user_email": None}


@router.post("/face", response_model=IdentifyResponse)
async def identify_by_face(
    file: UploadFile = File(...),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    """Identify a person by face — no login required (1:N search)."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    face_service = request.app.state.face_service
    result = await face_service.identify_face(image)

    user_info = {}
    if result["found"] and result["user_id"]:
        user_info = await _lookup_user(result["user_id"], db)

    return IdentifyResponse(
        found=result["found"],
        user_id=result.get("user_id"),
        user_name=user_info.get("user_name"),
        user_email=user_info.get("user_email"),
        score=result["score"],
        modality="face",
        message=result["message"],
    )


@router.post("/voice", response_model=IdentifyResponse)
async def identify_by_voice(
    file: UploadFile = File(...),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    """Identify a person by voice — no login required (1:N search)."""
    voice_service = request.app.state.voice_service
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Voice service is not available.")
    validate_audio_file(file)
    audio_bytes = await read_audio_from_upload(file)
    result = await voice_service.identify_voice(audio_bytes)

    user_info = {}
    if result["found"] and result["user_id"]:
        user_info = await _lookup_user(result["user_id"], db)

    return IdentifyResponse(
        found=result["found"],
        user_id=result.get("user_id"),
        user_name=user_info.get("user_name"),
        user_email=user_info.get("user_email"),
        score=result["score"],
        modality="voice",
        message=result["message"],
    )


@router.post("/fingerprint", response_model=IdentifyResponse)
async def identify_by_fingerprint(
    file: UploadFile = File(...),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    """Identify a person by fingerprint — no login required (1:N search)."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    fp_service = request.app.state.fingerprint_service
    result = await fp_service.identify_fingerprint(image)

    user_info = {}
    if result["found"] and result["user_id"]:
        user_info = await _lookup_user(result["user_id"], db)

    return IdentifyResponse(
        found=result["found"],
        user_id=result.get("user_id"),
        user_name=user_info.get("user_name"),
        user_email=user_info.get("user_email"),
        score=result["score"],
        modality="fingerprint",
        message=result["message"],
    )
