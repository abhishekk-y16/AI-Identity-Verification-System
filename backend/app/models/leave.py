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


class LeaveStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class LeaveType(Base):
    __tablename__ = "leave_types"

    leave_type_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    days_per_year: Mapped[int] = mapped_column(Integer, nullable=False, default=12)
    carry_forward: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    def __repr__(self):
        return f"<LeaveType {self.name}>"


class LeaveBalance(Base):
    __tablename__ = "leave_balances"
    __table_args__ = (
        UniqueConstraint("user_id", "leave_type_id", "year", name="uq_leave_balance"),
    )

    balance_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True
    )
    leave_type_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("leave_types.leave_type_id"), nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    total_days: Mapped[float] = mapped_column(Float, nullable=False)
    used_days: Mapped[float] = mapped_column(Float, default=0.0)
    remaining_days: Mapped[float] = mapped_column(Float, nullable=False)

    user = relationship("User", lazy="selectin")
    leave_type = relationship("LeaveType", lazy="selectin")

    def __repr__(self):
        return f"<LeaveBalance {self.user_id} {self.year} remaining={self.remaining_days}>"


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True
    )
    leave_type_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("leave_types.leave_type_id"), nullable=False
    )
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    days_count: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String(1000), nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(LeaveStatus), nullable=False, default=LeaveStatus.PENDING
    )
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True
    )
    admin_remarks: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", foreign_keys=[user_id], lazy="selectin")
    approver = relationship("User", foreign_keys=[approved_by], lazy="selectin")
    leave_type = relationship("LeaveType", lazy="selectin")

    def __repr__(self):
        return f"<LeaveRequest {self.user_id} {self.start_date}-{self.end_date} {self.status}>"
