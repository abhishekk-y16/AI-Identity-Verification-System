import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.utils.security import get_current_user, require_admin, require_manager
from app.models.employee import EmployeeProfile
from app.models.user import User
from app.models.department import Department
from app.models.shift import Shift
from app.schemas.office_schemas import (
    EmployeeProfileCreate,
    EmployeeProfileUpdate,
    EmployeeProfileResponse,
    EmployeeDirectoryResponse,
)

router = APIRouter(prefix="/api/employees", tags=["Employees"])


@router.get("/directory", response_model=list[EmployeeDirectoryResponse])
async def employee_directory(
    department_id: uuid.UUID | None = None,
    search: str | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(EmployeeProfile)
        .options(
            selectinload(EmployeeProfile.user),
            selectinload(EmployeeProfile.department),
            selectinload(EmployeeProfile.shift),
        )
        .where(EmployeeProfile.is_active == True)
    )
    if department_id:
        query = query.where(EmployeeProfile.department_id == department_id)
    if search:
        query = query.join(User).where(User.name.ilike(f"%{search}%"))

    result = await db.execute(query.order_by(EmployeeProfile.employee_code))
    profiles = result.scalars().all()
    out = []
    for p in profiles:
        out.append(
            EmployeeDirectoryResponse(
                profile_id=p.profile_id,
                user_id=p.user_id,
                employee_code=p.employee_code,
                full_name=p.user.name if p.user else "",
                email=p.user.email if p.user else "",
                department_name=p.department.name if p.department else None,
                shift_name=p.shift.name if p.shift else None,
                designation=p.designation,
                phone=p.phone,
                is_active=p.is_active,
            )
        )
    return out


@router.get("/me", response_model=EmployeeProfileResponse)
async def get_my_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmployeeProfile).where(EmployeeProfile.user_id == current_user["user_id"])
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Employee profile not found")
    return EmployeeProfileResponse.model_validate(profile)


@router.get("/{employee_id}", response_model=EmployeeProfileResponse)
async def get_employee(
    employee_id: uuid.UUID,
    current_user: dict = Depends(require_manager),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmployeeProfile).where(EmployeeProfile.profile_id == employee_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Employee not found")
    return EmployeeProfileResponse.model_validate(profile)


@router.post("/", response_model=EmployeeProfileResponse, status_code=201)
async def create_employee_profile(
    data: EmployeeProfileCreate,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    # Verify user exists
    user_result = await db.execute(select(User).where(User.user_id == data.user_id))
    if not user_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="User not found")

    existing = await db.execute(
        select(EmployeeProfile).where(EmployeeProfile.user_id == data.user_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Profile already exists for this user")

    profile = EmployeeProfile(**data.model_dump())
    db.add(profile)
    await db.flush()
    return EmployeeProfileResponse.model_validate(profile)


@router.put("/{employee_id}", response_model=EmployeeProfileResponse)
async def update_employee_profile(
    employee_id: uuid.UUID,
    data: EmployeeProfileUpdate,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmployeeProfile).where(EmployeeProfile.profile_id == employee_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Employee not found")
    for field, val in data.model_dump(exclude_unset=True).items():
        setattr(profile, field, val)
    return EmployeeProfileResponse.model_validate(profile)
