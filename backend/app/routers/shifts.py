import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.utils.security import require_admin
from app.models.shift import Shift
from app.schemas.office_schemas import ShiftCreate, ShiftUpdate, ShiftResponse

router = APIRouter(prefix="/api/shifts", tags=["Shifts"])


@router.get("/", response_model=list[ShiftResponse])
async def list_shifts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Shift).order_by(Shift.name))
    return [ShiftResponse.model_validate(s) for s in result.scalars().all()]


@router.post("/", response_model=ShiftResponse, status_code=201)
async def create_shift(
    data: ShiftCreate,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    shift = Shift(
        name=data.name,
        start_time=data.start_time,
        end_time=data.end_time,
        grace_minutes=data.grace_minutes,
    )
    db.add(shift)
    await db.flush()
    return ShiftResponse.model_validate(shift)


@router.put("/{shift_id}", response_model=ShiftResponse)
async def update_shift(
    shift_id: uuid.UUID,
    data: ShiftUpdate,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Shift).where(Shift.shift_id == shift_id))
    shift = result.scalar_one_or_none()
    if not shift:
        raise HTTPException(status_code=404, detail="Shift not found")
    for field, val in data.model_dump(exclude_unset=True).items():
        setattr(shift, field, val)
    return ShiftResponse.model_validate(shift)


@router.delete("/{shift_id}")
async def delete_shift(
    shift_id: uuid.UUID,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Shift).where(Shift.shift_id == shift_id))
    shift = result.scalar_one_or_none()
    if not shift:
        raise HTTPException(status_code=404, detail="Shift not found")
    if shift.employees:
        raise HTTPException(status_code=400, detail="Cannot delete shift assigned to employees")
    await db.delete(shift)
    return {"message": "Shift deleted"}
