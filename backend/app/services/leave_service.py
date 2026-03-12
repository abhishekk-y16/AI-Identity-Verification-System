import uuid
import logging
from datetime import date
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.leave import LeaveType, LeaveBalance, LeaveRequest, LeaveStatus

logger = logging.getLogger(__name__)


class LeaveService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_leave_types(self) -> list:
        result = await self.db.execute(select(LeaveType).order_by(LeaveType.name))
        return result.scalars().all()

    async def create_leave_type(self, data: dict) -> LeaveType:
        lt = LeaveType(
            name=data["name"],
            days_per_year=data["days_per_year"],
            carry_forward=data.get("carry_forward", False),
            description=data.get("description"),
        )
        self.db.add(lt)
        await self.db.flush()
        return lt

    async def initialize_balances(self, user_id: str, year: int) -> list:
        uid = uuid.UUID(user_id)
        types = await self.get_leave_types()
        balances = []
        for lt in types:
            existing = await self.db.execute(
                select(LeaveBalance).where(
                    and_(
                        LeaveBalance.user_id == uid,
                        LeaveBalance.leave_type_id == lt.leave_type_id,
                        LeaveBalance.year == year,
                    )
                )
            )
            if existing.scalar_one_or_none():
                continue
            bal = LeaveBalance(
                user_id=uid,
                leave_type_id=lt.leave_type_id,
                year=year,
                total_days=float(lt.days_per_year),
                used_days=0.0,
                remaining_days=float(lt.days_per_year),
            )
            self.db.add(bal)
            balances.append(bal)
        await self.db.flush()
        return balances

    async def get_balances(self, user_id: str, year: int) -> list:
        result = await self.db.execute(
            select(LeaveBalance).where(
                and_(LeaveBalance.user_id == uuid.UUID(user_id), LeaveBalance.year == year)
            )
        )
        return result.scalars().all()

    async def request_leave(self, user_id: str, data: dict) -> LeaveRequest:
        uid = uuid.UUID(user_id)
        leave_type_id = uuid.UUID(str(data["leave_type_id"]))
        start_date = data["start_date"]
        end_date = data["end_date"]
        reason = data.get("reason", "")

        days_count = float((end_date - start_date).days + 1)
        if days_count <= 0:
            raise ValueError("End date must be after start date")

        year = start_date.year
        bal_result = await self.db.execute(
            select(LeaveBalance).where(
                and_(
                    LeaveBalance.user_id == uid,
                    LeaveBalance.leave_type_id == leave_type_id,
                    LeaveBalance.year == year,
                )
            )
        )
        balance = bal_result.scalar_one_or_none()
        if not balance or balance.remaining_days < days_count:
            raise ValueError("Insufficient leave balance")

        req = LeaveRequest(
            user_id=uid,
            leave_type_id=leave_type_id,
            start_date=start_date,
            end_date=end_date,
            days_count=days_count,
            reason=reason,
        )
        self.db.add(req)
        await self.db.flush()
        return req

    async def get_user_requests(self, user_id: str) -> list:
        result = await self.db.execute(
            select(LeaveRequest)
            .where(LeaveRequest.user_id == uuid.UUID(user_id))
            .order_by(LeaveRequest.created_at.desc())
        )
        return result.scalars().all()

    async def get_pending_requests(self) -> list:
        result = await self.db.execute(
            select(LeaveRequest)
            .where(LeaveRequest.status == LeaveStatus.PENDING)
            .order_by(LeaveRequest.created_at)
        )
        return result.scalars().all()

    async def approve_leave(self, request_id: str, admin_id: str, remarks: str | None) -> LeaveRequest:
        result = await self.db.execute(
            select(LeaveRequest).where(LeaveRequest.request_id == uuid.UUID(request_id))
        )
        req = result.scalar_one_or_none()
        if not req:
            return None
        if req.status != LeaveStatus.PENDING:
            raise ValueError("Request already processed")

        req.status = LeaveStatus.APPROVED
        req.approved_by = uuid.UUID(admin_id)
        req.admin_remarks = remarks

        bal_result = await self.db.execute(
            select(LeaveBalance).where(
                and_(
                    LeaveBalance.user_id == req.user_id,
                    LeaveBalance.leave_type_id == req.leave_type_id,
                    LeaveBalance.year == req.start_date.year,
                )
            )
        )
        balance = bal_result.scalar_one_or_none()
        if balance:
            balance.used_days += req.days_count
            balance.remaining_days = balance.total_days - balance.used_days

        return req

    async def reject_leave(self, request_id: str, admin_id: str, remarks: str | None) -> LeaveRequest:
        result = await self.db.execute(
            select(LeaveRequest).where(LeaveRequest.request_id == uuid.UUID(request_id))
        )
        req = result.scalar_one_or_none()
        if not req:
            return None
        if req.status != LeaveStatus.PENDING:
            raise ValueError("Request already processed")

        req.status = LeaveStatus.REJECTED
        req.approved_by = uuid.UUID(admin_id)
        req.admin_remarks = remarks
        return req
