import uuid
from datetime import date, datetime, time
from pydantic import BaseModel, Field


# ─── Department ──────────────────────────────────────────────────────────────

class DepartmentCreate(BaseModel):
    name: str = Field(..., max_length=255)
    description: str | None = None
    head_id: uuid.UUID | None = None


class DepartmentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    head_id: uuid.UUID | None = None


class DepartmentResponse(BaseModel):
    department_id: uuid.UUID
    name: str
    description: str | None
    head_id: uuid.UUID | None
    head_name: str | None = None
    employee_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Shift ───────────────────────────────────────────────────────────────────

class ShiftCreate(BaseModel):
    name: str = Field(..., max_length=100)
    start_time: time
    end_time: time
    grace_minutes: int = 15


class ShiftUpdate(BaseModel):
    name: str | None = None
    start_time: time | None = None
    end_time: time | None = None
    grace_minutes: int | None = None
    is_active: bool | None = None


class ShiftResponse(BaseModel):
    shift_id: uuid.UUID
    name: str
    start_time: time
    end_time: time
    grace_minutes: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Employee Profile ────────────────────────────────────────────────────────

class EmployeeProfileCreate(BaseModel):
    user_id: uuid.UUID
    employee_code: str = Field(..., max_length=50)
    department_id: uuid.UUID
    shift_id: uuid.UUID
    designation: str | None = None
    phone: str | None = None
    join_date: date | None = None
    allowed_ips: list[str] | None = None


class EmployeeProfileUpdate(BaseModel):
    department_id: uuid.UUID | None = None
    shift_id: uuid.UUID | None = None
    designation: str | None = None
    phone: str | None = None
    join_date: date | None = None
    allowed_ips: list[str] | None = None
    is_active: bool | None = None


class EmployeeProfileResponse(BaseModel):
    profile_id: uuid.UUID
    user_id: uuid.UUID
    employee_code: str
    department_id: uuid.UUID
    shift_id: uuid.UUID
    designation: str | None
    phone: str | None
    join_date: date | None
    allowed_ips: list[str] | None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class EmployeeDirectoryResponse(BaseModel):
    profile_id: uuid.UUID
    user_id: uuid.UUID
    employee_code: str
    full_name: str
    email: str
    designation: str | None = None
    phone: str | None = None
    department_name: str | None = None
    shift_name: str | None = None
    is_active: bool = True


# ─── Attendance ──────────────────────────────────────────────────────────────

class ClockRequest(BaseModel):
    face_image: str  # base64-encoded JPEG
    voice_audio: str  # base64-encoded audio 
    device_id: str | None = None


class AttendanceRecordResponse(BaseModel):
    record_id: uuid.UUID
    user_id: uuid.UUID
    punch_type: str
    timestamp: datetime
    face_score: float | None
    voice_score: float | None
    ip_address: str | None
    status: str
    notes: str | None

    class Config:
        from_attributes = True


class DailySummaryResponse(BaseModel):
    summary_id: uuid.UUID
    user_id: uuid.UUID
    date: date
    first_clock_in: datetime | None
    last_clock_out: datetime | None
    total_hours: float
    overtime_hours: float
    status: str
    late_minutes: int
    early_departure_minutes: int

    class Config:
        from_attributes = True


class TodayStatusResponse(BaseModel):
    user_id: uuid.UUID
    employee_name: str
    employee_code: str
    department: str
    shift: str
    is_clocked_in: bool
    last_punch_time: datetime | None
    last_punch_type: str | None
    status: str | None  # on_time, late, etc.


# ─── Leave ───────────────────────────────────────────────────────────────────

class LeaveTypeCreate(BaseModel):
    name: str = Field(..., max_length=100)
    days_per_year: int
    carry_forward: bool = False
    description: str | None = None


class LeaveTypeResponse(BaseModel):
    leave_type_id: uuid.UUID
    name: str
    days_per_year: int
    carry_forward: bool
    description: str | None

    class Config:
        from_attributes = True


class LeaveBalanceResponse(BaseModel):
    balance_id: uuid.UUID
    leave_type_id: uuid.UUID
    year: int
    total_days: float
    used_days: float
    remaining_days: float

    class Config:
        from_attributes = True


class LeaveRequestCreate(BaseModel):
    leave_type_id: uuid.UUID
    start_date: date
    end_date: date
    reason: str = Field(..., max_length=1000)


class LeaveApprovalRequest(BaseModel):
    remarks: str | None = None


class LeaveRequestResponse(BaseModel):
    request_id: uuid.UUID
    user_id: uuid.UUID
    user_name: str | None = None
    leave_type_id: uuid.UUID
    leave_type_name: str | None = None
    start_date: date
    end_date: date
    days_count: float
    reason: str
    status: str
    approved_by: uuid.UUID | None
    approver_name: str | None = None
    admin_remarks: str | None
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Alert ───────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    alert_id: uuid.UUID
    user_id: uuid.UUID | None
    user_name: str | None = None
    alert_type: str
    title: str
    message: str
    severity: str
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Reports ─────────────────────────────────────────────────────────────────

class AttendanceReportFilter(BaseModel):
    start_date: date
    end_date: date
    department_id: uuid.UUID | None = None
    employee_id: uuid.UUID | None = None


class EmployeeAttendanceSummary(BaseModel):
    user_id: uuid.UUID
    employee_name: str
    employee_code: str
    department: str
    total_days: int
    present_days: int
    absent_days: int
    late_days: int
    leave_days: int
    total_hours: float
    overtime_hours: float
    avg_clock_in_time: str | None
    avg_clock_out_time: str | None


class AttendanceReportResponse(BaseModel):
    start_date: date
    end_date: date
    total_employees: int
    summaries: list[EmployeeAttendanceSummary]


# ─── Office Dashboard Stats ─────────────────────────────────────────────────

class OfficeStatsResponse(BaseModel):
    total_employees: int
    present_today: int
    absent_today: int
    late_today: int
    on_leave_today: int
    avg_clock_in_time: str | None
    departments: list[dict] = []


class OfficeTSPoint(BaseModel):
    date: date
    present: int
    absent: int
    late: int
    on_leave: int


# ─── Pagination ──────────────────────────────────────────────────────────────

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int
