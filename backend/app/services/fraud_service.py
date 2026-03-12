import uuid
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from app.models.verification_log import VerificationLog, VerificationStatus, FraudAlert, RiskLevel
from sklearn.ensemble import IsolationForest
import numpy as np

logger = logging.getLogger(__name__)


class FraudService:
    """Anomaly detection and rule-based fraud checks on verification patterns."""

    def __init__(self, max_failed_attempts: int = 5, window_minutes: int = 10):
        self.max_failed_attempts = max_failed_attempts
        self.window_minutes = window_minutes
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )
        self._model_trained = False

    async def check_fraud(
        self, user_id: str, db: AsyncSession, ip_address: str | None = None
    ) -> dict:
        """Run fraud detection checks for a user's verification attempt."""
        reasons = []

        # Rule-based: check recent failed attempts
        window_start = datetime.now(timezone.utc) - timedelta(minutes=self.window_minutes)
        result = await db.execute(
            select(func.count(VerificationLog.log_id)).where(
                and_(
                    VerificationLog.user_id == uuid.UUID(user_id),
                    VerificationLog.status == VerificationStatus.REJECTED,
                    VerificationLog.timestamp >= window_start,
                )
            )
        )
        failed_count = result.scalar() or 0

        if failed_count >= self.max_failed_attempts:
            reasons.append(
                f"Excessive failed attempts: {failed_count} in last {self.window_minutes} min"
            )

        # Check for multiple IPs in short window
        if ip_address:
            ip_result = await db.execute(
                select(func.count(func.distinct(VerificationLog.ip_address))).where(
                    and_(
                        VerificationLog.user_id == uuid.UUID(user_id),
                        VerificationLog.timestamp >= window_start,
                    )
                )
            )
            ip_count = ip_result.scalar() or 0
            if ip_count > 3:
                reasons.append(f"Multiple IP addresses detected: {ip_count} IPs in short window")

        # Check rapid-fire attempts (more than 1 per minute)
        one_min_ago = datetime.now(timezone.utc) - timedelta(minutes=1)
        rapid_result = await db.execute(
            select(func.count(VerificationLog.log_id)).where(
                and_(
                    VerificationLog.user_id == uuid.UUID(user_id),
                    VerificationLog.timestamp >= one_min_ago,
                )
            )
        )
        rapid_count = rapid_result.scalar() or 0
        if rapid_count > 3:
            reasons.append(f"Rapid-fire attempts: {rapid_count} in last minute")

        # Compute fraud score
        fraud_score = min(1.0, len(reasons) * 0.35 + (failed_count / 20.0))

        # Determine risk level
        if fraud_score >= 0.7:
            risk = RiskLevel.CRITICAL
        elif fraud_score >= 0.4:
            risk = RiskLevel.HIGH
        elif fraud_score >= 0.2:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        is_suspicious = fraud_score >= 0.4

        # Create fraud alert if suspicious
        if is_suspicious:
            alert = FraudAlert(
                user_id=uuid.UUID(user_id),
                alert_type="verification_anomaly",
                severity=risk,
                description="; ".join(reasons),
                extra_metadata={"fraud_score": fraud_score, "failed_count": failed_count},
            )
            db.add(alert)

        return {
            "is_suspicious": is_suspicious,
            "fraud_score": float(fraud_score),
            "reasons": reasons,
            "risk_level": risk,
        }

    async def train_model(self, db: AsyncSession):
        """Train Isolation Forest on historical verification data."""
        result = await db.execute(
            select(VerificationLog).order_by(VerificationLog.timestamp.desc()).limit(1000)
        )
        logs = result.scalars().all()

        if len(logs) < 50:
            logger.info("Not enough data to train fraud model")
            return

        # Build feature matrix
        features = []
        for log in logs:
            features.append([
                log.score or 0.0,
                1.0 if log.status == VerificationStatus.REJECTED else 0.0,
                log.timestamp.hour / 24.0,
                log.timestamp.weekday() / 7.0,
            ])

        X = np.array(features, dtype=np.float32)
        self.isolation_forest.fit(X)
        self._model_trained = True
        logger.info(f"Fraud model trained on {len(logs)} records")
