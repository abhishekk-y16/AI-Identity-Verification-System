import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.utils.security import get_current_user, require_admin, require_manager
from app.services.leave_service import LeaveService
from app.schemas.office_schemas import (
    LeaveTypeCreate,
    LeaveTypeResponse,
    LeaveBalanceResponse,
    LeaveRequestCreate,
    LeaveRequestResponse,
    LeaveApprovalRequest,
)

router = APIRouter(prefix="/api/leaves", tags=["Leaves"])


def _leave_service(db: AsyncSession = Depends(get_db)) -> LeaveService:
    return LeaveService(db)


# ---------- Leave Types ----------
@router.get("/types", response_model=list[LeaveTypeResponse])
async def list_leave_types(svc: LeaveService = Depends(_leave_service)):
    types = await svc.get_leave_types()
    return [LeaveTypeResponse.model_validate(t) for t in types]


@router.post("/types", response_model=LeaveTypeResponse, status_code=201)
async def create_leave_type(
    data: LeaveTypeCreate,
    current_user: dict = Depends(require_admin),
    svc: LeaveService = Depends(_leave_service),
):
    lt = await svc.create_leave_type(data.model_dump())
    return LeaveTypeResponse.model_validate(lt)


# ---------- Balances ----------
@router.get("/my-balance", response_model=list[LeaveBalanceResponse])
async def my_balances(
    year: int | None = None,
    current_user: dict = Depends(get_current_user),
    svc: LeaveService = Depends(_leave_service),
):
    from datetime import datetime

    y = year or datetime.utcnow().year
    balances = await svc.get_balances(current_user["user_id"], y)
    return [LeaveBalanceResponse.model_validate(b) for b in balances]


@router.post("/initialize-balances")
async def initialize_balances(
    user_id: uuid.UUID,
    year: int,
    current_user: dict = Depends(require_admin),
    svc: LeaveService = Depends(_leave_service),
):
    await svc.initialize_balances(str(user_id), year)
    return {"message": "Balances initialized"}


# ---------- Leave Requests ----------
@router.post("/request", response_model=LeaveRequestResponse, status_code=201)
async def request_leave(
    data: LeaveRequestCreate,
    current_user: dict = Depends(get_current_user),
    svc: LeaveService = Depends(_leave_service),
):
    req = await svc.request_leave(current_user["user_id"], data.model_dump())
    return LeaveRequestResponse.model_validate(req)


@router.get("/my-requests", response_model=list[LeaveRequestResponse])
async def my_leave_requests(
    current_user: dict = Depends(get_current_user),
    svc: LeaveService = Depends(_leave_service),
):
    reqs = await svc.get_user_requests(current_user["user_id"])
    return [LeaveRequestResponse.model_validate(r) for r in reqs]


@router.get("/pending", response_model=list[LeaveRequestResponse])
async def pending_requests(
    current_user: dict = Depends(require_manager),
    svc: LeaveService = Depends(_leave_service),
):
    reqs = await svc.get_pending_requests()
    return [LeaveRequestResponse.model_validate(r) for r in reqs]


@router.put("/{request_id}/approve", response_model=LeaveRequestResponse)
async def approve_leave(
    request_id: uuid.UUID,
    data: LeaveApprovalRequest,
    current_user: dict = Depends(require_manager),
    svc: LeaveService = Depends(_leave_service),
):
    req = await svc.approve_leave(str(request_id), current_user["user_id"], data.remarks)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    return LeaveRequestResponse.model_validate(req)


@router.put("/{request_id}/reject", response_model=LeaveRequestResponse)
async def reject_leave(
    request_id: uuid.UUID,
    data: LeaveApprovalRequest,
    current_user: dict = Depends(require_manager),
    svc: LeaveService = Depends(_leave_service),
):
    req = await svc.reject_leave(str(request_id), current_user["user_id"], data.remarks)
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    return LeaveRequestResponse.model_validate(req)
