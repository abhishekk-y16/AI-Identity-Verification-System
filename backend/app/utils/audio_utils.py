import io
import numpy as np
from fastapi import UploadFile


async def read_audio_from_upload(file: UploadFile) -> bytes:
    """Read uploaded audio file and return raw bytes."""
    return await file.read()


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file type."""
    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/webm", "audio/ogg", "audio/mpeg",
        "audio/mp4", "audio/flac",
    }
    if file.content_type and file.content_type not in allowed_types:
        raise ValueError(f"Invalid audio type: {file.content_type}. Allowed: {allowed_types}")
