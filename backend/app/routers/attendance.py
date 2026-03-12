import uuid
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.utils.security import get_current_user, require_admin
from app.services.alert_service import AlertService
from app.schemas.office_schemas import (
    ClockRequest, AttendanceRecordResponse, DailySummaryResponse,
)
from app.models.alert import AlertType, AlertSeverity

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])


@router.post("/clock-in")
async def clock_in(
    data: ClockRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = request.app.state.attendance_service
    alert_svc = AlertService(db)
    ip = request.client.host if request.client else "unknown"

    result = await svc.clock_in(
        user_id=current_user["user_id"],
        face_b64=data.face_image,
        voice_b64=data.voice_audio,
        ip_address=ip,
        device_id=data.device_id,
        db=db,
    )

    if not result["success"]:
        alert_type_str = result.get("alert_type")
        if alert_type_str == "unauthorized_ip":
            await alert_svc.create_alert(
                uuid.UUID(current_user["user_id"]),
                AlertType.UNAUTHORIZED_IP,
                "Unauthorized Clock-in Attempt",
                f"Employee tried to clock in from IP {ip}",
                AlertSeverity.WARNING,
            )
        elif alert_type_str == "biometric_fail":
            await alert_svc.create_alert(
                uuid.UUID(current_user["user_id"]),
                AlertType.BIOMETRIC_FAIL,
                "Biometric Verification Failed",
                f"Face score: {result.get('face_score', 0):.2f}, Voice score: {result.get('voice_score', 0):.2f}",
                AlertSeverity.WARNING,
            )
        raise HTTPException(status_code=400, detail=result["message"])

    # Create late alert if needed
    if result.get("status") == "late" and result.get("late_minutes", 0) > 0:
        await alert_svc.create_alert(
            uuid.UUID(current_user["user_id"]),
            AlertType.LATE_ARRIVAL,
            "Late Arrival",
            f"Employee arrived {result['late_minutes']} minutes late",
            AlertSeverity.WARNING if result["late_minutes"] > 30 else AlertSeverity.INFO,
        )

    return result


@router.post("/clock-out")
async def clock_out(
    data: ClockRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = request.app.state.attendance_service
    ip = request.client.host if request.client else "unknown"

    result = await svc.clock_out(
        user_id=current_user["user_id"],
        face_b64=data.face_image,
        voice_b64=data.voice_audio,
        ip_address=ip,
        device_id=data.device_id,
        db=db,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    # Early departure alert
    if result.get("status") == "early_departure":
        alert_svc = AlertService(db)
        await alert_svc.create_alert(
            uuid.UUID(current_user["user_id"]),
            AlertType.EARLY_DEPARTURE,
            "Early Departure",
            "Employee left before shift end",
            AlertSeverity.INFO,
        )

    return result


@router.get("/today")
async def get_today(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = request.app.state.attendance_service
    records = await svc.get_today_records(current_user["user_id"], db)
    return [AttendanceRecordResponse.model_validate(r) for r in records]


@router.get("/history")
async def get_history(
    start_date: date,
    end_date: date,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = request.app.state.attendance_service
    records = await svc.get_history(current_user["user_id"], start_date, end_date, db)
    return [AttendanceRecordResponse.model_validate(r) for r in records]


@router.get("/employee/{employee_id}/history")
async def get_employee_history(
    employee_id: uuid.UUID,
    start_date: date,
    end_date: date,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user["role"] not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Not authorized")
    svc = request.app.state.attendance_service
    records = await svc.get_history(str(employee_id), start_date, end_date, db)
    return [AttendanceRecordResponse.model_validate(r) for r in records]


@router.post("/daily-summary")
async def compute_daily_summary(
    target_date: date | None = None,
    request: Request = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    svc = request.app.state.attendance_service
    d = target_date or date.today()
    count = await svc.compute_daily_summary(d, db)
    return {"date": d.isoformat(), "summaries_computed": count}
