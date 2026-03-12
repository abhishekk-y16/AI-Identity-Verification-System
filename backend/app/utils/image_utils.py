import numpy as np
import cv2
import io
import base64
from PIL import Image
from fastapi import UploadFile


async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read an uploaded image file and return as RGB numpy array."""
    contents = await file.read()
    return decode_image_bytes(contents)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes (JPEG, PNG, etc.) to RGB numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    # Convert BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 encoded image to RGB numpy array."""
    # Remove data URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_str)
    return decode_image_bytes(image_bytes)


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """Encode RGB numpy array to base64 string."""
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image keeping aspect ratio, max dimension = max_size."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames_from_video_bytes(video_bytes: bytes, max_frames: int = 30) -> list[np.ndarray]:
    """Extract frames from video bytes, returning RGB numpy arrays."""
    import tempfile
    import os

    # Write to temp file for cv2.VideoCapture
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)

        frame_idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            frame_idx += 1

        cap.release()
        return frames
    finally:
        os.unlink(tmp_path)


def validate_image_file(file: UploadFile, max_size_mb: int = 10) -> None:
    """Validate uploaded image file type and size."""
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise ValueError(f"Invalid image type: {file.content_type}. Allowed: {allowed_types}")


def validate_video_file(file: UploadFile, max_size_mb: int = 50) -> None:
    """Validate uploaded video file type."""
    allowed_types = {"video/webm", "video/mp4", "video/avi", "video/quicktime"}
    if file.content_type not in allowed_types:
        raise ValueError(f"Invalid video type: {file.content_type}. Allowed: {allowed_types}")
