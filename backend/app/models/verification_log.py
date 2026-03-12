import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, JSON, func, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import enum


class VerificationType(str, enum.Enum):
    FACE = "face"
    VOICE = "voice"
    FINGERPRINT = "fingerprint"
    LIVENESS = "liveness"
    DEEPFAKE = "deepfake"
    DOCUMENT = "document"
    BEHAVIORAL = "behavioral"
    FULL = "full"


class VerificationStatus(str, enum.Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    REVIEW = "review"
    PENDING = "pending"
    ERROR = "error"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationLog(Base):
    __tablename__ = "verification_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True
    )
    verification_type: Mapped[str] = mapped_column(
        SAEnum(VerificationType), nullable=False
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        SAEnum(VerificationStatus), nullable=False, default=VerificationStatus.PENDING
    )
    risk_level: Mapped[str | None] = mapped_column(
        SAEnum(RiskLevel), nullable=True
    )
    device_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    extra_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    user = relationship("User", back_populates="verification_logs")

    def __repr__(self):
        return f"<VerificationLog {self.log_id} type={self.verification_type} status={self.status}>"


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True
    )
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(
        SAEnum(RiskLevel), nullable=False, default=RiskLevel.MEDIUM
    )
    description: Mapped[str] = mapped_column(String(1000), nullable=False)
    extra_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    resolved: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
