import numpy as np
import hashlib
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TokenizationService:
    """Cancelable biometric template protection.
    
    Transforms biometric embeddings using per-user random projection matrices
    so that raw biometric data is never stored. If a token is compromised,
    a new projection can be generated, invalidating the old tokens.
    """

    def generate_seed(self) -> str:
        """Generate a new random seed for a user."""
        import secrets
        return secrets.token_hex(32)

    def tokenize(self, embedding: np.ndarray, user_seed: str) -> np.ndarray:
        """Apply cancelable biometric transformation.
        
        Uses the user's seed to generate a deterministic random orthogonal matrix,
        then applies a non-linear transform to prevent inversion.
        """
        rng = np.random.RandomState(
            int(hashlib.sha256(user_seed.encode()).hexdigest()[:8], 16)
        )
        dim = len(embedding)
        random_matrix = rng.randn(dim, dim).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)

        projected = q @ embedding.astype(np.float32)
        tokenized = np.tanh(projected * 2.0)

        return tokenized

    def compare(self, token1: np.ndarray, token2: np.ndarray) -> float:
        """Compare two tokenized embeddings."""
        dot = np.dot(token1, token2)
        norm1 = np.linalg.norm(token1)
        norm2 = np.linalg.norm(token2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def revoke_and_reissue(self, embedding: np.ndarray) -> tuple[str, np.ndarray]:
        """Revoke old token by generating new seed + new tokenized embedding."""
        new_seed = self.generate_seed()
        new_token = self.tokenize(embedding, new_seed)
        return new_seed, new_token
