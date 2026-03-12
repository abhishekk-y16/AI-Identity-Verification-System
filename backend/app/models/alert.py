import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, ForeignKey, Enum as SAEnum, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class AlertType(str, enum.Enum):
    LATE_ARRIVAL = "late_arrival"
    ABSENT = "absent"
    EARLY_DEPARTURE = "early_departure"
    UNAUTHORIZED_IP = "unauthorized_ip"
    BIOMETRIC_FAIL = "biometric_fail"


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True, index=True
    )
    alert_type: Mapped[str] = mapped_column(SAEnum(AlertType), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(String(1000), nullable=False)
    severity: Mapped[str] = mapped_column(
        SAEnum(AlertSeverity), nullable=False, default=AlertSeverity.INFO
    )
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    user = relationship("User", lazy="selectin")

    def __repr__(self):
        return f"<Alert {self.alert_type} {self.severity}>"
