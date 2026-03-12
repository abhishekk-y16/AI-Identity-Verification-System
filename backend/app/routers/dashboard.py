import uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, case
from app.database import get_db
from app.models.user import User
from app.models.verification_log import (
    VerificationLog, VerificationType, VerificationStatus, FraudAlert, RiskLevel,
)
from app.schemas.schemas import (
    DashboardStats, DashboardTimeSeriesPoint, VerificationLogResponse,
    FraudAlertResponse, PaginatedResponse, UserResponse,
)
from app.utils.security import require_admin

router = APIRouter(prefix="/api/dashboard", tags=["Admin Dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_stats(
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get overall system statistics."""
    # Total users
    user_count = (await db.execute(select(func.count(User.user_id)))).scalar() or 0

    # Total verifications
    total_verifications = (
        await db.execute(select(func.count(VerificationLog.log_id)))
    ).scalar() or 0

    # Successful
    successful = (
        await db.execute(
            select(func.count(VerificationLog.log_id)).where(
                VerificationLog.status == VerificationStatus.VERIFIED
            )
        )
    ).scalar() or 0

    # Failed
    failed = (
        await db.execute(
            select(func.count(VerificationLog.log_id)).where(
                VerificationLog.status == VerificationStatus.REJECTED
            )
        )
    ).scalar() or 0

    # Success rate
    success_rate = (successful / total_verifications * 100) if total_verifications > 0 else 0.0

    # Fraud alerts
    fraud_count = (
        await db.execute(
            select(func.count(FraudAlert.alert_id)).where(FraudAlert.resolved == False)
        )
    ).scalar() or 0

    # Average scores per modality
    avg_face = (
        await db.execute(
            select(func.avg(VerificationLog.score)).where(
                VerificationLog.verification_type == VerificationType.FACE
            )
        )
    ).scalar()

    avg_voice = (
        await db.execute(
            select(func.avg(VerificationLog.score)).where(
                VerificationLog.verification_type == VerificationType.VOICE
            )
        )
    ).scalar()

    avg_fingerprint = (
        await db.execute(
            select(func.avg(VerificationLog.score)).where(
                VerificationLog.verification_type == VerificationType.FINGERPRINT
            )
        )
    ).scalar()

    return DashboardStats(
        total_users=user_count,
        total_verifications=total_verifications,
        successful_verifications=successful,
        failed_verifications=failed,
        success_rate=round(success_rate, 2),
        fraud_alerts_count=fraud_count,
        avg_face_score=round(avg_face, 4) if avg_face else None,
        avg_voice_score=round(avg_voice, 4) if avg_voice else None,
        avg_fingerprint_score=round(avg_fingerprint, 4) if avg_fingerprint else None,
    )


@router.get("/logs")
async def get_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str | None = None,
    verification_type: str | None = None,
    status: str | None = None,
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated verification logs with optional filters."""
    query = select(VerificationLog)

    if user_id:
        query = query.where(VerificationLog.user_id == uuid.UUID(user_id))
    if verification_type:
        query = query.where(VerificationLog.verification_type == verification_type)
    if status:
        query = query.where(VerificationLog.status == status)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Paginate
    query = query.order_by(desc(VerificationLog.timestamp)).offset(
        (page - 1) * page_size
    ).limit(page_size)

    result = await db.execute(query)
    logs = result.scalars().all()

    return PaginatedResponse(
        items=[VerificationLogResponse.model_validate(log) for log in logs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/users")
async def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated user list."""
    total = (await db.execute(select(func.count(User.user_id)))).scalar() or 0
    result = await db.execute(
        select(User).order_by(desc(User.created_at)).offset(
            (page - 1) * page_size
        ).limit(page_size)
    )
    users = result.scalars().all()

    return PaginatedResponse(
        items=[UserResponse.model_validate(u) for u in users],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/alerts")
async def get_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    resolved: bool | None = None,
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get fraud alerts."""
    query = select(FraudAlert)
    if resolved is not None:
        query = query.where(FraudAlert.resolved == resolved)

    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar() or 0
    result = await db.execute(
        query.order_by(desc(FraudAlert.created_at)).offset(
            (page - 1) * page_size
        ).limit(page_size)
    )
    alerts = result.scalars().all()

    return PaginatedResponse(
        items=[FraudAlertResponse.model_validate(a) for a in alerts],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/timeseries")
async def get_timeseries(
    days: int = Query(30, ge=1, le=365),
    admin: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get verification time-series data for charts."""
    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    result = await db.execute(
        select(
            func.date(VerificationLog.timestamp).label("date"),
            func.count(VerificationLog.log_id).label("count"),
            func.sum(
                case(
                    (VerificationLog.status == VerificationStatus.VERIFIED, 1),
                    else_=0,
                )
            ).label("success_count"),
            func.sum(
                case(
                    (VerificationLog.status == VerificationStatus.REJECTED, 1),
                    else_=0,
                )
            ).label("failure_count"),
        )
        .where(VerificationLog.timestamp >= start_date)
        .group_by(func.date(VerificationLog.timestamp))
        .order_by(func.date(VerificationLog.timestamp))
    )

    points = []
    for row in result:
        points.append(DashboardTimeSeriesPoint(
            date=str(row.date),
            count=row.count,
            success_count=row.success_count or 0,
            failure_count=row.failure_count or 0,
        ))

    return points
