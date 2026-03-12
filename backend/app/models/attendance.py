import uuid
import enum
from datetime import datetime, date
from sqlalchemy import (
    String, Float, Integer, Date, DateTime, Boolean,
    ForeignKey, Enum as SAEnum, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class PunchType(str, enum.Enum):
    CLOCK_IN = "clock_in"
    CLOCK_OUT = "clock_out"


class PunchStatus(str, enum.Enum):
    ON_TIME = "on_time"
    LATE = "late"
    EARLY_DEPARTURE = "early_departure"
    OVERTIME = "overtime"


class DayStatus(str, enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"
    HALF_DAY = "half_day"
    ON_LEAVE = "on_leave"
    HOLIDAY = "holiday"


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True
    )
    punch_type: Mapped[str] = mapped_column(SAEnum(PunchType), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    face_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    voice_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    photo_proof: Mapped[str | None] = mapped_column(String(500), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    device_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        SAEnum(PunchStatus), nullable=False, default=PunchStatus.ON_TIME
    )
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)

    user = relationship("User", lazy="selectin")

    def __repr__(self):
        return f"<AttendanceRecord {self.user_id} {self.punch_type} {self.timestamp}>"


class DailySummary(Base):
    __tablename__ = "daily_summaries"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_daily_summary_user_date"),
    )

    summary_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    first_clock_in: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_clock_out: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_hours: Mapped[float] = mapped_column(Float, default=0.0)
    overtime_hours: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(
        SAEnum(DayStatus), nullable=False, default=DayStatus.ABSENT
    )
    late_minutes: Mapped[int] = mapped_column(Integer, default=0)
    early_departure_minutes: Mapped[int] = mapped_column(Integer, default=0)

    user = relationship("User", lazy="selectin")

    def __repr__(self):
        return f"<DailySummary {self.user_id} {self.date} {self.status}>"
