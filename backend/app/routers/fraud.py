from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.schemas import FraudCheckResponse
from app.utils.security import get_current_user

router = APIRouter(prefix="/api/fraud", tags=["Fraud Detection"])


def get_fraud_service(request: Request):
    return request.app.state.fraud_service


@router.post("/check", response_model=FraudCheckResponse)
async def check_fraud(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
):
    """Run fraud detection checks on the current user's verification patterns."""
    fraud_service = get_fraud_service(request)
    result = await fraud_service.check_fraud(
        user_id=current_user["user_id"],
        db=db,
        ip_address=request.client.host if request.client else None,
    )
    return FraudCheckResponse(**result)
