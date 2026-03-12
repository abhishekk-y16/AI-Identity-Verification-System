import uuid
import csv
import io
import logging
from datetime import date
from sqlalchemy import select, and_, cast, Date, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.attendance import DailySummary, DayStatus, AttendanceRecord, PunchType
from app.models.employee import EmployeeProfile
from app.models.user import User

logger = logging.getLogger(__name__)


class ReportService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_attendance_report(
        self, start_date: date, end_date: date,
        department_id: uuid.UUID | None = None,
    ) -> dict:
        query = (
            select(EmployeeProfile)
            .where(EmployeeProfile.is_active == True)
        )
        if department_id:
            query = query.where(EmployeeProfile.department_id == department_id)

        profiles_result = await self.db.execute(query)
        profiles = profiles_result.scalars().all()

        total_work_days = (end_date - start_date).days + 1
        summaries = []

        for p in profiles:
            uid = p.user_id
            ds_result = await self.db.execute(
                select(DailySummary).where(
                    and_(
                        DailySummary.user_id == uid,
                        DailySummary.date >= start_date,
                        DailySummary.date <= end_date,
                    )
                )
            )
            daily = ds_result.scalars().all()

            present = sum(1 for d in daily if d.status == DayStatus.PRESENT)
            absent = sum(1 for d in daily if d.status == DayStatus.ABSENT)
            leave = sum(1 for d in daily if d.status == DayStatus.ON_LEAVE)
            late = sum(1 for d in daily if d.late_minutes > 0)
            total_hrs = sum(d.total_hours for d in daily)
            ot_hrs = sum(d.overtime_hours for d in daily)

            summaries.append({
                "user_id": str(uid),
                "employee_name": "",
                "employee_code": p.employee_code,
                "department": "",
                "total_days": total_work_days,
                "present_days": present,
                "absent_days": absent,
                "late_days": late,
                "leave_days": leave,
                "total_hours": round(total_hrs, 2),
                "overtime_hours": round(ot_hrs, 2),
                "avg_clock_in_time": None,
                "avg_clock_out_time": None,
            })

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_employees": len(summaries),
            "summaries": summaries,
        }

    async def get_overtime_report(
        self, start_date: date, end_date: date,
        department_id: uuid.UUID | None = None,
    ) -> list[dict]:
        query = select(EmployeeProfile).where(EmployeeProfile.is_active == True)
        if department_id:
            query = query.where(EmployeeProfile.department_id == department_id)

        profiles_result = await self.db.execute(query)
        profiles = profiles_result.scalars().all()
        rows = []

        for p in profiles:
            ds_result = await self.db.execute(
                select(DailySummary).where(
                    and_(
                        DailySummary.user_id == p.user_id,
                        DailySummary.date >= start_date,
                        DailySummary.date <= end_date,
                        DailySummary.overtime_hours > 0,
                    )
                )
            )
            daily = ds_result.scalars().all()
            if not daily:
                continue
            total_ot = sum(d.overtime_hours for d in daily)
            rows.append({
                "user_id": str(p.user_id),
                "employee_name": "",
                "employee_code": p.employee_code,
                "department": "",
                "overtime_days": len(daily),
                "total_overtime_hours": round(total_ot, 2),
            })

        return sorted(rows, key=lambda r: r["total_overtime_hours"], reverse=True)

    async def export_csv(self, start_date: date, end_date: date, department_id: uuid.UUID | None = None) -> bytes:
        report_data = await self.get_attendance_report(start_date, end_date, department_id)
        output = io.StringIO()
        summaries = report_data.get("summaries", [])
        if not summaries:
            return b""
        writer = csv.DictWriter(output, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
        return output.getvalue().encode("utf-8")
