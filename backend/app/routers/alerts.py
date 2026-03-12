import uuid
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.utils.security import get_current_user, require_admin
from app.services.alert_service import AlertService
from app.schemas.office_schemas import AlertResponse

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


def _alert_service(db: AsyncSession = Depends(get_db)) -> AlertService:
    return AlertService(db)


@router.get("/", response_model=dict)
async def get_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_read: bool | None = None,
    alert_type: str | None = None,
    current_user: dict = Depends(get_current_user),
    svc: AlertService = Depends(_alert_service),
):
    # Admins see all; employees see their own
    user_id = None if current_user.get("role") == "admin" else current_user["user_id"]
    alerts, total = await svc.get_alerts(
        page=page, page_size=page_size, is_read=is_read, alert_type=alert_type, user_id=user_id
    )
    return {
        "items": [AlertResponse.model_validate(a) for a in alerts],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/unread-count")
async def unread_count(
    current_user: dict = Depends(get_current_user),
    svc: AlertService = Depends(_alert_service),
):
    user_id = None if current_user.get("role") == "admin" else current_user["user_id"]
    count = await svc.get_unread_count(user_id)
    return {"unread_count": count}


@router.put("/{alert_id}/read")
async def mark_read(
    alert_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    svc: AlertService = Depends(_alert_service),
):
    await svc.mark_read(str(alert_id))
    return {"message": "Marked as read"}


@router.put("/read-all")
async def mark_all_read(
    current_user: dict = Depends(get_current_user),
    svc: AlertService = Depends(_alert_service),
):
    user_id = None if current_user.get("role") == "admin" else current_user["user_id"]
    await svc.mark_all_read(user_id)
    return {"message": "All alerts marked as read"}
