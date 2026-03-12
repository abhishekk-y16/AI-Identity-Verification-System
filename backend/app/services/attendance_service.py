import uuid
import base64
import logging
from datetime import datetime, date, time, timedelta, timezone
from sqlalchemy import select, and_, func, cast, Date
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.attendance import AttendanceRecord, DailySummary, PunchType, PunchStatus, DayStatus
from app.models.employee import EmployeeProfile
from app.models.leave import LeaveRequest, LeaveStatus

logger = logging.getLogger(__name__)


class AttendanceService:
    def __init__(self, face_service, voice_service):
        self.face_service = face_service
        self.voice_service = voice_service

    async def _verify_biometrics(self, user_id: str, face_b64: str, voice_b64: str) -> dict:
        """Verify face + voice biometrics. Returns scores dict."""
        face_score = 0.0
        voice_score = 0.0

        # Verify face
        try:
            face_bytes = base64.b64decode(face_b64)
            face_result = await self.face_service.verify_face(user_id, face_bytes)
            face_score = face_result.get("score", 0.0)
        except Exception as e:
            logger.warning(f"Face verification failed for {user_id}: {e}")

        # Verify voice
        try:
            if self.voice_service:
                voice_bytes = base64.b64decode(voice_b64)
                voice_result = await self.voice_service.verify_voice(user_id, voice_bytes)
                voice_score = voice_result.get("score", 0.0)
        except Exception as e:
            logger.warning(f"Voice verification failed for {user_id}: {e}")

        passed = face_score >= 0.5  # face is mandatory
        return {"face_score": face_score, "voice_score": voice_score, "passed": passed}

    def _check_ip_allowed(self, ip_address: str, allowed_ips: list[str] | None) -> bool:
        if not allowed_ips:
            return True
        return ip_address in allowed_ips

    def _compute_punch_status(
        self, punch_type: PunchType, timestamp: datetime,
        shift_start: time, shift_end: time, grace_minutes: int,
    ) -> PunchStatus:
        punch_time = timestamp.time()
        if punch_type == PunchType.CLOCK_IN:
            limit = (
                datetime.combine(date.today(), shift_start)
                + timedelta(minutes=grace_minutes)
            ).time()
            return PunchStatus.LATE if punch_time > limit else PunchStatus.ON_TIME
        else:
            if punch_time < shift_end:
                return PunchStatus.EARLY_DEPARTURE
            overtime_limit = (
                datetime.combine(date.today(), shift_end)
                + timedelta(minutes=30)
            ).time()
            return PunchStatus.OVERTIME if punch_time > overtime_limit else PunchStatus.ON_TIME

    async def clock_in(
        self, user_id: str, face_b64: str, voice_b64: str,
        ip_address: str, device_id: str | None, db: AsyncSession,
    ) -> dict:
        # Get employee profile
        result = await db.execute(
            select(EmployeeProfile).where(EmployeeProfile.user_id == uuid.UUID(user_id))
        )
        profile = result.scalar_one_or_none()
        if not profile:
            return {"success": False, "message": "Employee profile not found"}

        # Check IP geo-fence
        if not self._check_ip_allowed(ip_address, profile.allowed_ips):
            return {
                "success": False,
                "message": "Clock-in not allowed from this network",
                "alert_type": "unauthorized_ip",
            }

        # Verify biometrics
        bio = await self._verify_biometrics(user_id, face_b64, voice_b64)
        if not bio["passed"]:
            return {
                "success": False,
                "message": "Biometric verification failed",
                "face_score": bio["face_score"],
                "voice_score": bio["voice_score"],
                "alert_type": "biometric_fail",
            }

        # Compute status
        now = datetime.now(timezone.utc)
        shift = profile.shift
        status = self._compute_punch_status(
            PunchType.CLOCK_IN, now, shift.start_time, shift.end_time, shift.grace_minutes
        )

        record = AttendanceRecord(
            user_id=uuid.UUID(user_id),
            punch_type=PunchType.CLOCK_IN,
            timestamp=now,
            face_score=bio["face_score"],
            voice_score=bio["voice_score"],
            ip_address=ip_address,
            device_id=device_id,
            status=status,
        )
        db.add(record)
        await db.flush()

        late_minutes = 0
        if status == PunchStatus.LATE:
            shift_dt = datetime.combine(now.date(), shift.start_time, tzinfo=timezone.utc)
            late_minutes = int((now - shift_dt).total_seconds() / 60)

        return {
            "success": True,
            "record_id": str(record.record_id),
            "status": status.value,
            "timestamp": now.isoformat(),
            "face_score": bio["face_score"],
            "voice_score": bio["voice_score"],
            "late_minutes": late_minutes,
            "message": f"Clocked in — {status.value.replace('_', ' ').title()}",
        }

    async def clock_out(
        self, user_id: str, face_b64: str, voice_b64: str,
        ip_address: str, device_id: str | None, db: AsyncSession,
    ) -> dict:
        result = await db.execute(
            select(EmployeeProfile).where(EmployeeProfile.user_id == uuid.UUID(user_id))
        )
        profile = result.scalar_one_or_none()
        if not profile:
            return {"success": False, "message": "Employee profile not found"}

        bio = await self._verify_biometrics(user_id, face_b64, voice_b64)
        if not bio["passed"]:
            return {
                "success": False,
                "message": "Biometric verification failed",
                "face_score": bio["face_score"],
                "voice_score": bio["voice_score"],
            }

        now = datetime.now(timezone.utc)
        shift = profile.shift
        status = self._compute_punch_status(
            PunchType.CLOCK_OUT, now, shift.start_time, shift.end_time, shift.grace_minutes
        )

        record = AttendanceRecord(
            user_id=uuid.UUID(user_id),
            punch_type=PunchType.CLOCK_OUT,
            timestamp=now,
            face_score=bio["face_score"],
            voice_score=bio["voice_score"],
            ip_address=ip_address,
            device_id=device_id,
            status=status,
        )
        db.add(record)
        await db.flush()

        return {
            "success": True,
            "record_id": str(record.record_id),
            "status": status.value,
            "timestamp": now.isoformat(),
            "face_score": bio["face_score"],
            "voice_score": bio["voice_score"],
            "message": f"Clocked out — {status.value.replace('_', ' ').title()}",
        }

    async def get_today_records(self, user_id: str, db: AsyncSession) -> list:
        today = date.today()
        result = await db.execute(
            select(AttendanceRecord)
            .where(
                and_(
                    AttendanceRecord.user_id == uuid.UUID(user_id),
                    cast(AttendanceRecord.timestamp, Date) == today,
                )
            )
            .order_by(AttendanceRecord.timestamp)
        )
        return result.scalars().all()

    async def get_history(
        self, user_id: str, start_date: date, end_date: date, db: AsyncSession,
    ) -> list:
        result = await db.execute(
            select(AttendanceRecord)
            .where(
                and_(
                    AttendanceRecord.user_id == uuid.UUID(user_id),
                    cast(AttendanceRecord.timestamp, Date) >= start_date,
                    cast(AttendanceRecord.timestamp, Date) <= end_date,
                )
            )
            .order_by(AttendanceRecord.timestamp.desc())
        )
        return result.scalars().all()

    async def compute_daily_summary(self, target_date: date, db: AsyncSession) -> int:
        """Compute daily summaries for all employees. Returns count of summaries created."""
        profiles_result = await db.execute(
            select(EmployeeProfile).where(EmployeeProfile.is_active == True)
        )
        profiles = profiles_result.scalars().all()
        count = 0

        for profile in profiles:
            uid = profile.user_id

            # Check for leave
            leave_result = await db.execute(
                select(LeaveRequest).where(
                    and_(
                        LeaveRequest.user_id == uid,
                        LeaveRequest.status == LeaveStatus.APPROVED,
                        LeaveRequest.start_date <= target_date,
                        LeaveRequest.end_date >= target_date,
                    )
                )
            )
            on_leave = leave_result.scalar_one_or_none() is not None

            # Get punches
            punches_result = await db.execute(
                select(AttendanceRecord)
                .where(
                    and_(
                        AttendanceRecord.user_id == uid,
                        cast(AttendanceRecord.timestamp, Date) == target_date,
                    )
                )
                .order_by(AttendanceRecord.timestamp)
            )
            punches = punches_result.scalars().all()

            clock_ins = [p for p in punches if p.punch_type == PunchType.CLOCK_IN]
            clock_outs = [p for p in punches if p.punch_type == PunchType.CLOCK_OUT]

            first_in = clock_ins[0].timestamp if clock_ins else None
            last_out = clock_outs[-1].timestamp if clock_outs else None

            total_hours = 0.0
            if first_in and last_out and last_out > first_in:
                total_hours = round((last_out - first_in).total_seconds() / 3600, 2)

            shift = profile.shift
            shift_hours = (
                datetime.combine(target_date, shift.end_time)
                - datetime.combine(target_date, shift.start_time)
            ).total_seconds() / 3600
            overtime_hours = max(0, round(total_hours - shift_hours, 2))

            # Determine status
            if on_leave:
                day_status = DayStatus.ON_LEAVE
            elif not clock_ins:
                day_status = DayStatus.ABSENT
            elif total_hours < shift_hours / 2:
                day_status = DayStatus.HALF_DAY
            else:
                day_status = DayStatus.PRESENT

            late_mins = 0
            if first_in:
                shift_start_dt = datetime.combine(target_date, shift.start_time, tzinfo=first_in.tzinfo)
                diff = (first_in - shift_start_dt).total_seconds() / 60
                late_mins = max(0, int(diff))

            early_mins = 0
            if last_out:
                shift_end_dt = datetime.combine(target_date, shift.end_time, tzinfo=last_out.tzinfo)
                diff = (shift_end_dt - last_out).total_seconds() / 60
                early_mins = max(0, int(diff))

            # Upsert summary
            existing = await db.execute(
                select(DailySummary).where(
                    and_(DailySummary.user_id == uid, DailySummary.date == target_date)
                )
            )
            summary = existing.scalar_one_or_none()
            if summary:
                summary.first_clock_in = first_in
                summary.last_clock_out = last_out
                summary.total_hours = total_hours
                summary.overtime_hours = overtime_hours
                summary.status = day_status
                summary.late_minutes = late_mins
                summary.early_departure_minutes = early_mins
            else:
                summary = DailySummary(
                    user_id=uid,
                    date=target_date,
                    first_clock_in=first_in,
                    last_clock_out=last_out,
                    total_hours=total_hours,
                    overtime_hours=overtime_hours,
                    status=day_status,
                    late_minutes=late_mins,
                    early_departure_minutes=early_mins,
                )
                db.add(summary)
            count += 1

        return count
