from fastapi import APIRouter, Depends, UploadFile, File, Request
from app.schemas.schemas import DeepfakeResponse
from app.utils.security import get_current_user
from app.utils.image_utils import read_image_from_upload, validate_image_file, validate_video_file, extract_frames_from_video_bytes

router = APIRouter(prefix="/api/deepfake", tags=["Deepfake Detection"])


def get_deepfake_service(request: Request):
    return request.app.state.deepfake_service


@router.post("/detect", response_model=DeepfakeResponse)
async def detect_deepfake_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Detect if an image contains a deepfake face."""
    validate_image_file(file)
    image = await read_image_from_upload(file)
    deepfake_service = get_deepfake_service(request)
    result = await deepfake_service.detect_image(image)
    return DeepfakeResponse(**result)


@router.post("/detect-video", response_model=DeepfakeResponse)
async def detect_deepfake_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Detect deepfake in video content."""
    validate_video_file(file)
    video_bytes = await file.read()
    frames = extract_frames_from_video_bytes(video_bytes, max_frames=30)
    deepfake_service = get_deepfake_service(request)
    result = await deepfake_service.detect_video(frames)
    return DeepfakeResponse(**result)
