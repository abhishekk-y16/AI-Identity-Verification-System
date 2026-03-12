from fastapi import APIRouter, Depends, UploadFile, File, Request
from app.schemas.schemas import DocumentVerifyResponse
from app.utils.security import get_current_user
from app.utils.image_utils import read_image_from_upload, validate_image_file

router = APIRouter(prefix="/api/document", tags=["Document KYC"])


def get_document_service(request: Request):
    return request.app.state.document_service


@router.post("/verify", response_model=DocumentVerifyResponse)
async def verify_document(
    document: UploadFile = File(..., description="ID document image (passport, Aadhaar, etc.)"),
    selfie: UploadFile = File(..., description="Live selfie image"),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Compare face on ID document with live selfie for KYC verification."""
    validate_image_file(document)
    validate_image_file(selfie)

    doc_image = await read_image_from_upload(document)
    selfie_image = await read_image_from_upload(selfie)

    doc_service = get_document_service(request)
    result = await doc_service.verify_document(doc_image, selfie_image)
    return DocumentVerifyResponse(**result)
