import uuid
import logging
from datetime import date, datetime, timedelta, timezone
from sqlalchemy import select, and_, cast, Date, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.attendance import AttendanceRecord, DailySummary, DayStatus, PunchType
from app.models.employee import EmployeeProfile
from app.models.leave import LeaveRequest, LeaveStatus
from app.models.user import User

logger = logging.getLogger(__name__)


class OfficeStatsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_today_stats(self) -> dict:
        today = date.today()

        emp_count = (await self.db.execute(
            select(func.count(EmployeeProfile.profile_id))
            .where(EmployeeProfile.is_active == True)
        )).scalar() or 0

        clocked_in_result = await self.db.execute(
            select(AttendanceRecord.user_id)
            .where(
                and_(
                    AttendanceRecord.punch_type == PunchType.CLOCK_IN,
                    cast(AttendanceRecord.timestamp, Date) == today,
                )
            )
            .distinct()
        )
        present_ids = set(row[0] for row in clocked_in_result.all())

        late_result = await self.db.execute(
            select(AttendanceRecord.user_id)
            .where(
                and_(
                    AttendanceRecord.punch_type == PunchType.CLOCK_IN,
                    cast(AttendanceRecord.timestamp, Date) == today,
                    AttendanceRecord.status == "late",
                )
            )
            .distinct()
        )
        late_ids = set(row[0] for row in late_result.all())

        leave_result = await self.db.execute(
            select(LeaveRequest.user_id)
            .where(
                and_(
                    LeaveRequest.status == LeaveStatus.APPROVED,
                    LeaveRequest.start_date <= today,
                    LeaveRequest.end_date >= today,
                )
            )
            .distinct()
        )
        leave_ids = set(row[0] for row in leave_result.all())

        absent = emp_count - len(present_ids) - len(leave_ids)
        if absent < 0:
            absent = 0

        return {
            "total_employees": emp_count,
            "present_today": len(present_ids),
            "absent_today": absent,
            "late_today": len(late_ids),
            "on_leave_today": len(leave_ids),
            "avg_clock_in_time": None,
            "departments": [],
        }

    async def get_department_breakdown(self) -> list[dict]:
        today = date.today()

        profiles_result = await self.db.execute(
            select(EmployeeProfile).where(EmployeeProfile.is_active == True)
        )
        profiles = profiles_result.scalars().all()

        dept_map: dict[str, dict] = {}
        for p in profiles:
            dept_name = p.department.name if p.department else "Unknown"
            if dept_name not in dept_map:
                dept_map[dept_name] = {"total": 0, "present": 0, "absent": 0, "late": 0, "on_leave": 0}
            dept_map[dept_name]["total"] += 1

        clocked_result = await self.db.execute(
            select(AttendanceRecord.user_id, AttendanceRecord.status)
            .where(
                and_(
                    AttendanceRecord.punch_type == PunchType.CLOCK_IN,
                    cast(AttendanceRecord.timestamp, Date) == today,
                )
            )
        )
        clocked = clocked_result.all()
        present_map = {}
        for uid, st in clocked:
            present_map[uid] = st

        for p in profiles:
            dept_name = p.department.name if p.department else "Unknown"
            if p.user_id in present_map:
                dept_map[dept_name]["present"] += 1
                if present_map[p.user_id] == "late":
                    dept_map[dept_name]["late"] += 1

        for dept_name, data in dept_map.items():
            data["absent"] = data["total"] - data["present"] - data["on_leave"]
            if data["absent"] < 0:
                data["absent"] = 0

        return [
            {"department": name, **data}
            for name, data in sorted(dept_map.items())
        ]

    async def get_timeseries(self, days: int) -> list[dict]:
        today = date.today()
        start = today - timedelta(days=days)

        result = await self.db.execute(
            select(DailySummary.date, DailySummary.status, func.count())
            .where(DailySummary.date >= start)
            .group_by(DailySummary.date, DailySummary.status)
            .order_by(DailySummary.date)
        )
        rows = result.all()

        date_map: dict[date, dict] = {}
        for d, status_val, cnt in rows:
            if d not in date_map:
                date_map[d] = {"present": 0, "absent": 0, "late": 0, "on_leave": 0}
            if status_val == DayStatus.PRESENT or status_val == "present":
                date_map[d]["present"] += cnt
            elif status_val == DayStatus.ABSENT or status_val == "absent":
                date_map[d]["absent"] += cnt
            elif status_val == DayStatus.ON_LEAVE or status_val == "on_leave":
                date_map[d]["on_leave"] += cnt

        return [
            {"date": d.isoformat(), **counts}
            for d, counts in sorted(date_map.items())
        ]

    async def get_live_status(self) -> list[dict]:
        today = date.today()

        profiles_result = await self.db.execute(
            select(EmployeeProfile).where(EmployeeProfile.is_active == True)
        )
        profiles = profiles_result.scalars().all()

        statuses = []
        for p in profiles:
            punch_result = await self.db.execute(
                select(AttendanceRecord)
                .where(
                    and_(
                        AttendanceRecord.user_id == p.user_id,
                        cast(AttendanceRecord.timestamp, Date) == today,
                    )
                )
                .order_by(AttendanceRecord.timestamp.desc())
                .limit(1)
            )
            last_punch = punch_result.scalar_one_or_none()

            statuses.append({
                "user_id": str(p.user_id),
                "employee_name": "",
                "employee_code": p.employee_code,
                "department": "",
                "shift": "",
                "is_clocked_in": bool(last_punch and last_punch.punch_type == PunchType.CLOCK_IN),
                "last_punch_time": last_punch.timestamp.isoformat() if last_punch else None,
                "last_punch_type": last_punch.punch_type.value if last_punch else None,
                "status": last_punch.status.value if last_punch else None,
            })

        return statuses
