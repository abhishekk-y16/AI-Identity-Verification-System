import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from app.models.verification_log import VerificationType, VerificationStatus, RiskLevel


# ─── Auth Schemas ───────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    user_id: uuid.UUID
    name: str
    email: str
    role: str
    face_enrolled: bool
    voice_enrolled: bool
    fingerprint_enrolled: bool
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ─── Biometric Schemas ──────────────────────────────────────────────────────
class BiometricRegisterResponse(BaseModel):
    success: bool
    message: str
    embedding_id: str | None = None


class BiometricVerifyResponse(BaseModel):
    match: bool
    score: float
    threshold: float
    message: str


class FaceIdentifyResponse(BaseModel):
    found: bool
    user_id: uuid.UUID | None = None
    score: float = 0.0
    message: str


class IdentifyResponse(BaseModel):
    found: bool
    user_id: uuid.UUID | None = None
    user_name: str | None = None
    user_email: str | None = None
    score: float = 0.0
    modality: str
    message: str


# ─── Liveness Schemas ───────────────────────────────────────────────────────
class LivenessResponse(BaseModel):
    is_live: bool
    confidence: float
    checks: dict[str, bool | float] = {}
    message: str


# ─── Deepfake Schemas ───────────────────────────────────────────────────────
class DeepfakeResponse(BaseModel):
    is_deepfake: bool
    confidence: float
    method: str
    message: str


# ─── Document KYC Schemas ───────────────────────────────────────────────────
class DocumentVerifyResponse(BaseModel):
    document_type: str
    extracted_name: str | None = None
    face_match_score: float
    face_match: bool
    ocr_data: dict = {}
    message: str


# ─── Fusion / Full Verification Schemas ─────────────────────────────────────
class ModalityScore(BaseModel):
    modality: str
    score: float
    weight: float
    weighted_score: float
    passed: bool


class FullVerificationResponse(BaseModel):
    verification_id: uuid.UUID
    final_score: float
    risk_level: RiskLevel
    status: VerificationStatus
    liveness_passed: bool
    deepfake_passed: bool
    modality_scores: list[ModalityScore]
    message: str


# ─── Fraud Schemas ──────────────────────────────────────────────────────────
class FraudCheckResponse(BaseModel):
    is_suspicious: bool
    fraud_score: float
    reasons: list[str]
    risk_level: RiskLevel


# ─── Behavioral Schemas ─────────────────────────────────────────────────────
class BehavioralEvent(BaseModel):
    event_type: str  # keystroke, mouse_move, scroll, swipe
    timestamp: float
    data: dict


class BehavioralSubmitRequest(BaseModel):
    events: list[BehavioralEvent]
    session_id: str


class BehavioralResponse(BaseModel):
    behavior_match_score: float
    anomalies: list[str]
    message: str


# ─── Verification Log Schemas ───────────────────────────────────────────────
class VerificationLogResponse(BaseModel):
    log_id: uuid.UUID
    user_id: uuid.UUID
    verification_type: VerificationType
    score: float | None
    status: VerificationStatus
    risk_level: RiskLevel | None
    device_id: str | None
    ip_address: str | None
    timestamp: datetime
    extra_metadata: dict | None

    class Config:
        from_attributes = True


# ─── Dashboard Schemas ──────────────────────────────────────────────────────
class DashboardStats(BaseModel):
    total_users: int
    total_verifications: int
    successful_verifications: int
    failed_verifications: int
    success_rate: float
    fraud_alerts_count: int
    avg_face_score: float | None
    avg_voice_score: float | None
    avg_fingerprint_score: float | None


class DashboardTimeSeriesPoint(BaseModel):
    date: str
    count: int
    success_count: int
    failure_count: int


class FraudAlertResponse(BaseModel):
    alert_id: uuid.UUID
    user_id: uuid.UUID
    alert_type: str
    severity: RiskLevel
    description: str
    resolved: bool
    created_at: datetime

    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int
