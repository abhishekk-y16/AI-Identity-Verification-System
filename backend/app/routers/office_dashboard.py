from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.utils.security import require_manager
from app.services.office_stats_service import OfficeStatsService
from app.schemas.office_schemas import OfficeStatsResponse
import uuid

router = APIRouter(prefix="/api/office", tags=["Office Dashboard"])


def _stats_service(db: AsyncSession = Depends(get_db)) -> OfficeStatsService:
    return OfficeStatsService(db)


@router.get("/stats", response_model=OfficeStatsResponse)
async def today_stats(
    current_user: dict = Depends(require_manager),
    svc: OfficeStatsService = Depends(_stats_service),
):
    return await svc.get_today_stats()


@router.get("/stats/departments")
async def department_breakdown(
    current_user: dict = Depends(require_manager),
    svc: OfficeStatsService = Depends(_stats_service),
):
    return await svc.get_department_breakdown()


@router.get("/timeseries")
async def timeseries(
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(require_manager),
    svc: OfficeStatsService = Depends(_stats_service),
):
    return await svc.get_timeseries(days)


@router.get("/live")
async def live_status(
    current_user: dict = Depends(require_manager),
    svc: OfficeStatsService = Depends(_stats_service),
):
    return await svc.get_live_status()
