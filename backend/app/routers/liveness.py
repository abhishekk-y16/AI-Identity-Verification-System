from fastapi import APIRouter, Depends, UploadFile, File, Request
from app.schemas.schemas import LivenessResponse
from app.utils.security import get_current_user
from app.utils.image_utils import read_image_from_upload, validate_image_file, validate_video_file, extract_frames_from_video_bytes

router = APIRouter(prefix="/api/liveness", tags=["Liveness Detection"])


def get_liveness_service(request: Request):
    return request.app.state.liveness_service


@router.post("/check", response_model=LivenessResponse)
async def check_liveness_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Check liveness on a single face image."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    liveness_service = get_liveness_service(request)
    result = await liveness_service.check_single_image(image)
    return LivenessResponse(**result)


@router.post("/check-video", response_model=LivenessResponse)
async def check_liveness_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Check liveness on a video (multi-frame analysis with blink detection)."""
    validate_video_file(file)
    video_bytes = await file.read()
    frames = extract_frames_from_video_bytes(video_bytes, max_frames=30)
    liveness_service = get_liveness_service(request)
    result = await liveness_service.check_liveness(frames)
    return LivenessResponse(**result)
