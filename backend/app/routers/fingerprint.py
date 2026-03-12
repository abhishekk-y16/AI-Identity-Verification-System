from fastapi import APIRouter, Depends, UploadFile, File, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.schemas import BiometricRegisterResponse, BiometricVerifyResponse
from app.utils.security import get_current_user
from app.utils.image_utils import read_image_from_upload, validate_image_file

router = APIRouter(prefix="/api/fingerprint", tags=["Fingerprint Recognition"])


def get_fingerprint_service(request: Request):
    return request.app.state.fingerprint_service


@router.post("/register", response_model=BiometricRegisterResponse)
async def register_fingerprint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
):
    """Register fingerprint biometric for the authenticated user."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    fp_service = get_fingerprint_service(request)
    result = await fp_service.register_fingerprint(current_user["user_id"], image, db)
    return BiometricRegisterResponse(**result)


@router.post("/verify", response_model=BiometricVerifyResponse)
async def verify_fingerprint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Verify fingerprint against enrolled template."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    fp_service = get_fingerprint_service(request)
    result = await fp_service.verify_fingerprint(current_user["user_id"], image)
    return BiometricVerifyResponse(**result)
