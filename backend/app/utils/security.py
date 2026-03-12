import os
import secrets
import hashlib
import hmac
import numpy as np
import bcrypt
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import get_settings

settings = get_settings()

security_scheme = HTTPBearer()


# ─── Password Hashing ──────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# ─── JWT Tokens ─────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> dict:
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identity",
        )
    return {"user_id": user_id, "role": payload.get("role", "user")}


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def require_manager(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") not in ("admin", "manager"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager or admin access required",
        )
    return current_user


# ─── Biometric Tokenization (Cancelable Biometrics) ────────────────────────
def generate_biometric_seed() -> str:
    """Generate a random seed for per-user biometric tokenization."""
    return secrets.token_hex(32)


def tokenize_embedding(embedding: np.ndarray, user_seed: str) -> np.ndarray:
    """Apply cancelable biometric transformation using random projection.
    
    Uses the user's seed to generate a deterministic random projection matrix,
    then transforms the embedding so the original cannot be recovered without the seed.
    """
    rng = np.random.RandomState(
        int(hashlib.sha256(user_seed.encode()).hexdigest()[:8], 16)
    )
    # Generate random orthogonal projection matrix
    dim = len(embedding)
    random_matrix = rng.randn(dim, dim).astype(np.float32)
    # QR decomposition for orthogonal matrix
    q, _ = np.linalg.qr(random_matrix)

    # Apply projection + non-linear transform
    projected = q @ embedding.astype(np.float32)
    # Apply element-wise non-linearity to make inversion harder
    tokenized = np.tanh(projected * 2.0)

    return tokenized


def compare_tokenized(token1: np.ndarray, token2: np.ndarray) -> float:
    """Compare two tokenized embeddings using cosine similarity."""
    dot = np.dot(token1, token2)
    norm1 = np.linalg.norm(token1)
    norm2 = np.linalg.norm(token2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def embedding_to_hex(embedding: np.ndarray) -> str:
    """Serialize embedding to hex string for storage."""
    return embedding.astype(np.float32).tobytes().hex()


def hex_to_embedding(hex_str: str) -> np.ndarray:
    """Deserialize embedding from hex string."""
    return np.frombuffer(bytes.fromhex(hex_str), dtype=np.float32)
