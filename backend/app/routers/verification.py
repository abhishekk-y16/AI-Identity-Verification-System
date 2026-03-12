import uuid
from fastapi import APIRouter, Depends, UploadFile, File, Request, Form
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.verification_log import VerificationLog, VerificationType, VerificationStatus
from app.schemas.schemas import FullVerificationResponse, ModalityScore, BehavioralSubmitRequest, BehavioralResponse
from app.utils.security import get_current_user
from app.utils.image_utils import (
    read_image_from_upload, validate_image_file, validate_video_file,
    extract_frames_from_video_bytes,
)
from app.utils.audio_utils import read_audio_from_upload, validate_audio_file

router = APIRouter(prefix="/api/verify", tags=["Verification Engine"])


@router.post("/full", response_model=FullVerificationResponse)
async def full_verification(
    face_image: UploadFile = File(None, description="Face image for verification"),
    voice_audio: UploadFile = File(None, description="Voice audio for verification"),
    fingerprint_image: UploadFile = File(None, description="Fingerprint image"),
    liveness_video: UploadFile = File(None, description="Liveness video (optional)"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
):
    """Full multimodal biometric verification pipeline.
    
    Orchestrates: liveness → deepfake → face → voice → fingerprint → fusion → log.
    At least one biometric modality must be provided.
    """
    user_id = current_user["user_id"]
    face_score = None
    voice_score = None
    fingerprint_score = None
    liveness_passed = True
    deepfake_passed = True

    # 1. Liveness check (if video or face image provided)
    if liveness_video:
        validate_video_file(liveness_video)
        video_bytes = await liveness_video.read()
        frames = extract_frames_from_video_bytes(video_bytes)
        liveness_result = await request.app.state.liveness_service.check_liveness(frames)
        liveness_passed = liveness_result["is_live"]
    elif face_image:
        # Quick liveness check on single image
        face_bytes = await face_image.read()
        await face_image.seek(0)  # reset for next read
        from app.utils.image_utils import decode_image_bytes
        face_img = decode_image_bytes(face_bytes)
        liveness_result = await request.app.state.liveness_service.check_single_image(face_img)
        liveness_passed = liveness_result["is_live"]

    # 2. Deepfake check on face image
    if face_image:
        face_img_data = await read_image_from_upload(face_image)
        deepfake_result = await request.app.state.deepfake_service.detect_image(face_img_data)
        deepfake_passed = not deepfake_result["is_deepfake"]

        # 3. Face verification
        face_result = await request.app.state.face_service.verify_face(user_id, face_img_data)
        face_score = face_result["score"]

    # 4. Voice verification
    if voice_audio:
        validate_audio_file(voice_audio)
        audio_bytes = await read_audio_from_upload(voice_audio)
        voice_result = await request.app.state.voice_service.verify_voice(user_id, audio_bytes)
        voice_score = voice_result["score"]

    # 5. Fingerprint verification
    if fingerprint_image:
        validate_image_file(fingerprint_image)
        fp_img = await read_image_from_upload(fingerprint_image)
        fp_result = await request.app.state.fingerprint_service.verify_fingerprint(user_id, fp_img)
        fingerprint_score = fp_result["score"]

    # 6. Fusion scoring
    fusion_result = request.app.state.fusion_service.compute_fusion(
        face_score=face_score,
        voice_score=voice_score,
        fingerprint_score=fingerprint_score,
        liveness_passed=liveness_passed,
        deepfake_passed=deepfake_passed,
    )

    # 7. Fraud check
    fraud_result = await request.app.state.fraud_service.check_fraud(
        user_id=user_id,
        db=db,
        ip_address=request.client.host if request.client else None,
    )
    if fraud_result["is_suspicious"]:
        fusion_result["status"] = VerificationStatus.REVIEW
        fusion_result["message"] += " | FRAUD ALERT: " + "; ".join(fraud_result["reasons"])

    # 8. Log verification
    verification_id = uuid.uuid4()
    log = VerificationLog(
        log_id=verification_id,
        user_id=uuid.UUID(user_id),
        verification_type=VerificationType.FULL,
        score=fusion_result["final_score"],
        status=fusion_result["status"],
        risk_level=fusion_result["risk_level"],
        device_id=request.headers.get("X-Device-ID"),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("User-Agent"),
        extra_metadata={
            "face_score": face_score,
            "voice_score": voice_score,
            "fingerprint_score": fingerprint_score,
            "liveness_passed": liveness_passed,
            "deepfake_passed": deepfake_passed,
            "fraud_suspicious": fraud_result["is_suspicious"],
        },
    )
    db.add(log)

    return FullVerificationResponse(
        verification_id=verification_id,
        final_score=fusion_result["final_score"],
        risk_level=fusion_result["risk_level"],
        status=fusion_result["status"],
        liveness_passed=liveness_passed,
        deepfake_passed=deepfake_passed,
        modality_scores=[ModalityScore(**m) for m in fusion_result["modality_scores"]],
        message=fusion_result["message"],
    )


@router.post("/video")
async def video_verification(
    file: UploadFile = File(..., description="Short verification video"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
):
    """Real-time video verification: extract frames → liveness → face verify → deepfake check."""
    validate_video_file(file)
    video_bytes = await file.read()
    frames = extract_frames_from_video_bytes(video_bytes, max_frames=30)

    if not frames:
        return {"success": False, "message": "Could not extract frames from video"}

    # Liveness on all frames
    liveness_result = await request.app.state.liveness_service.check_liveness(frames)

    # Pick best frame (middle) for face verification
    mid_frame = frames[len(frames) // 2]
    face_result = await request.app.state.face_service.verify_face(
        current_user["user_id"], mid_frame
    )

    # Deepfake on sampled frames
    deepfake_result = await request.app.state.deepfake_service.detect_video(frames)

    return {
        "success": True,
        "liveness": liveness_result,
        "face_verification": face_result,
        "deepfake_check": {
            "is_deepfake": deepfake_result["is_deepfake"],
            "confidence": deepfake_result["confidence"],
        },
        "frames_analyzed": len(frames),
    }


@router.post("/behavioral", response_model=BehavioralResponse)
async def submit_behavioral(
    data: BehavioralSubmitRequest,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Submit behavioral biometric data for analysis."""
    events = [e.model_dump() for e in data.events]
    behavioral_service = request.app.state.behavioral_service
    result = await behavioral_service.analyze(current_user["user_id"], events)
    return BehavioralResponse(**result)
