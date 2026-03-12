import logging
from app.config import get_settings
from app.models.verification_log import RiskLevel, VerificationStatus

logger = logging.getLogger(__name__)
settings = get_settings()


class FusionService:
    """Weighted multimodal biometric score fusion.
    
    Combines face, voice, and fingerprint scores with configurable weights
    to produce a final identity confidence score and risk assessment.
    """

    def __init__(
        self,
        face_weight: float | None = None,
        voice_weight: float | None = None,
        fingerprint_weight: float | None = None,
    ):
        self.face_weight = face_weight or settings.FACE_WEIGHT
        self.voice_weight = voice_weight or settings.VOICE_WEIGHT
        self.fingerprint_weight = fingerprint_weight or settings.FINGERPRINT_WEIGHT

    def compute_fusion(
        self,
        face_score: float | None = None,
        voice_score: float | None = None,
        fingerprint_score: float | None = None,
        liveness_passed: bool = True,
        deepfake_passed: bool = True,
    ) -> dict:
        """Compute weighted fusion of available modality scores.
        
        Modalities that are not provided (None) are excluded, and weights
        are re-normalized among the available modalities.
        """
        modalities = []
        total_weight = 0.0

        if face_score is not None:
            modalities.append({
                "modality": "face",
                "score": float(face_score),
                "weight": self.face_weight,
                "weighted_score": face_score * self.face_weight,
                "passed": face_score >= settings.FACE_MATCH_THRESHOLD,
            })
            total_weight += self.face_weight

        if voice_score is not None:
            modalities.append({
                "modality": "voice",
                "score": float(voice_score),
                "weight": self.voice_weight,
                "weighted_score": voice_score * self.voice_weight,
                "passed": voice_score >= settings.VOICE_MATCH_THRESHOLD,
            })
            total_weight += self.voice_weight

        if fingerprint_score is not None:
            modalities.append({
                "modality": "fingerprint",
                "score": float(fingerprint_score),
                "weight": self.fingerprint_weight,
                "weighted_score": fingerprint_score * self.fingerprint_weight,
                "passed": fingerprint_score >= settings.FINGERPRINT_MATCH_THRESHOLD,
            })
            total_weight += self.fingerprint_weight

        if total_weight == 0:
            return {
                "final_score": 0.0,
                "risk_level": RiskLevel.HIGH,
                "status": VerificationStatus.REJECTED,
                "modality_scores": [],
                "message": "No biometric data provided",
            }

        # Normalize
        raw_score = sum(m["weighted_score"] for m in modalities) / total_weight

        # Apply penalties for security checks
        if not liveness_passed:
            raw_score *= 0.7  # Moderate penalty for failed liveness
        if not deepfake_passed:
            raw_score *= 0.2  # Heavy penalty for detected deepfake

        final_score = max(0.0, min(1.0, raw_score))

        # Risk level
        if final_score >= 0.85:
            risk_level = RiskLevel.LOW
        elif final_score >= 0.6:
            risk_level = RiskLevel.MEDIUM
        elif final_score >= 0.3:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        # Verification decision
        if final_score >= 0.7 and deepfake_passed:
            status = VerificationStatus.VERIFIED
        elif final_score >= 0.5:
            status = VerificationStatus.REVIEW
        else:
            status = VerificationStatus.REJECTED

        message = (
            f"Identity Match Score: {final_score * 100:.1f}% | "
            f"Risk Score: {risk_level.value.upper()} | "
            f"Verification Status: {status.value.upper()}"
        )

        return {
            "final_score": float(final_score),
            "risk_level": risk_level,
            "status": status,
            "modality_scores": modalities,
            "message": message,
        }
