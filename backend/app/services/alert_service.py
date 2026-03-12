import uuid
import logging
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.alert import Alert, AlertType, AlertSeverity

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_alert(
        self, user_id: uuid.UUID | None, alert_type: AlertType,
        title: str, message: str, severity: AlertSeverity,
    ) -> Alert:
        alert = Alert(
            user_id=user_id,
            alert_type=alert_type,
            title=title,
            message=message,
            severity=severity,
        )
        self.db.add(alert)
        await self.db.flush()
        logger.info(f"Alert created: {alert_type.value} - {title}")
        return alert

    async def get_alerts(
        self, page: int = 1, page_size: int = 20,
        is_read: bool | None = None, alert_type: str | None = None,
        user_id: str | None = None,
    ) -> tuple[list, int]:
        query = select(Alert)
        count_query = select(func.count(Alert.alert_id))

        if user_id:
            query = query.where(Alert.user_id == user_id)
            count_query = count_query.where(Alert.user_id == user_id)
        if is_read is not None:
            query = query.where(Alert.is_read == is_read)
            count_query = count_query.where(Alert.is_read == is_read)
        if alert_type:
            query = query.where(Alert.alert_type == alert_type)
            count_query = count_query.where(Alert.alert_type == alert_type)

        total = (await self.db.execute(count_query)).scalar() or 0
        result = await self.db.execute(
            query.order_by(Alert.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return result.scalars().all(), total

    async def get_unread_count(self, user_id: str | None = None) -> int:
        query = select(func.count(Alert.alert_id)).where(Alert.is_read == False)
        if user_id:
            query = query.where(Alert.user_id == user_id)
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def mark_read(self, alert_id: str) -> bool:
        result = await self.db.execute(
            select(Alert).where(Alert.alert_id == alert_id)
        )
        alert = result.scalar_one_or_none()
        if alert:
            alert.is_read = True
            return True
        return False

    async def mark_all_read(self, user_id: str | None = None) -> int:
        query = select(Alert).where(Alert.is_read == False)
        if user_id:
            query = query.where(Alert.user_id == user_id)
        result = await self.db.execute(query)
        alerts = result.scalars().all()
        for a in alerts:
            a.is_read = True
        return len(alerts)
