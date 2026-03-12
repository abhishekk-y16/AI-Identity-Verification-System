from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.schemas import BiometricRegisterResponse, BiometricVerifyResponse, FaceIdentifyResponse
from app.utils.security import get_current_user
from app.utils.image_utils import read_image_from_upload, decode_base64_image, validate_image_file

router = APIRouter(prefix="/api/face", tags=["Face Recognition"])


def get_face_service(request: Request):
    return request.app.state.face_service


@router.post("/register", response_model=BiometricRegisterResponse)
async def register_face(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
):
    """Register face biometric for the authenticated user."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    face_service = get_face_service(request)
    result = await face_service.register_face(current_user["user_id"], image, db)
    return BiometricRegisterResponse(**result)


@router.post("/verify", response_model=BiometricVerifyResponse)
async def verify_face(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Verify face against enrolled biometric."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    face_service = get_face_service(request)
    result = await face_service.verify_face(current_user["user_id"], image)
    return BiometricVerifyResponse(**result)


@router.post("/identify", response_model=FaceIdentifyResponse)
async def identify_face(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Identify a face from the database (1:N search)."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    face_service = get_face_service(request)
    result = await face_service.identify_face(image)
    return FaceIdentifyResponse(**result)
