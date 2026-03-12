from datetime import date
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.utils.security import require_manager
from app.services.report_service import ReportService
from app.schemas.office_schemas import AttendanceReportResponse
import io
import uuid

router = APIRouter(prefix="/api/reports", tags=["Reports"])


def _report_service(db: AsyncSession = Depends(get_db)) -> ReportService:
    return ReportService(db)


@router.get("/attendance", response_model=AttendanceReportResponse)
async def attendance_report(
    start_date: date,
    end_date: date,
    department_id: uuid.UUID | None = None,
    current_user: dict = Depends(require_manager),
    svc: ReportService = Depends(_report_service),
):
    return await svc.get_attendance_report(start_date, end_date, department_id)


@router.get("/overtime")
async def overtime_report(
    start_date: date,
    end_date: date,
    department_id: uuid.UUID | None = None,
    current_user: dict = Depends(require_manager),
    svc: ReportService = Depends(_report_service),
):
    return await svc.get_overtime_report(start_date, end_date, department_id)


@router.get("/export/csv")
async def export_csv(
    start_date: date,
    end_date: date,
    department_id: uuid.UUID | None = None,
    current_user: dict = Depends(require_manager),
    svc: ReportService = Depends(_report_service),
):
    csv_bytes = await svc.export_csv(start_date, end_date, department_id)
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{start_date}_{end_date}.csv"},
    )
