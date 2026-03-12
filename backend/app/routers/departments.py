import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.utils.security import require_admin
from app.models.department import Department
from app.schemas.office_schemas import DepartmentCreate, DepartmentUpdate, DepartmentResponse

router = APIRouter(prefix="/api/departments", tags=["Departments"])


@router.get("/", response_model=list[DepartmentResponse])
async def list_departments(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Department).order_by(Department.name))
    depts = result.scalars().all()
    return [
        DepartmentResponse(
            department_id=d.department_id,
            name=d.name,
            description=d.description,
            head_id=d.head_id,
            head_name=d.head.name if d.head else None,
            employee_count=len(d.employees) if d.employees else 0,
            created_at=d.created_at,
        )
        for d in depts
    ]


@router.post("/", response_model=DepartmentResponse, status_code=201)
async def create_department(
    data: DepartmentCreate,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    dept = Department(name=data.name, description=data.description, head_id=data.head_id)
    db.add(dept)
    await db.flush()
    return DepartmentResponse(
        department_id=dept.department_id,
        name=dept.name,
        description=dept.description,
        head_id=dept.head_id,
        head_name=None,
        employee_count=0,
        created_at=dept.created_at,
    )


@router.put("/{department_id}", response_model=DepartmentResponse)
async def update_department(
    department_id: uuid.UUID,
    data: DepartmentUpdate,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Department).where(Department.department_id == department_id))
    dept = result.scalar_one_or_none()
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    if data.name is not None:
        dept.name = data.name
    if data.description is not None:
        dept.description = data.description
    if data.head_id is not None:
        dept.head_id = data.head_id
    return DepartmentResponse(
        department_id=dept.department_id,
        name=dept.name,
        description=dept.description,
        head_id=dept.head_id,
        head_name=dept.head.name if dept.head else None,
        employee_count=len(dept.employees) if dept.employees else 0,
        created_at=dept.created_at,
    )


@router.delete("/{department_id}")
async def delete_department(
    department_id: uuid.UUID,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Department).where(Department.department_id == department_id))
    dept = result.scalar_one_or_none()
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    if dept.employees:
        raise HTTPException(status_code=400, detail="Cannot delete department with employees")
    await db.delete(dept)
    return {"message": "Department deleted"}
