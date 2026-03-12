import numpy as np
import faiss
import os
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class FaissIndexManager:
    """Manages FAISS vector indices for biometric embedding storage and search.
    
    Supports separate indices per biometric type (face, voice, fingerprint).
    Thread-safe with read-write lock for concurrent access.
    """

    def __init__(self, index_dir: str, dimension: int, index_type: str = "face"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.index_type = index_type
        self.lock = threading.RLock()

        # Mapping: internal FAISS id → user_id string
        self.id_map: dict[int, str] = {}
        self.next_id: int = 0

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        # Load existing index if available
        self._load()

        logger.info(
            f"FaissIndexManager ({index_type}) initialized: "
            f"dim={dimension}, vectors={self.index.ntotal}"
        )

    def _index_path(self) -> str:
        return str(self.index_dir / f"{self.index_type}.index")

    def _map_path(self) -> str:
        return str(self.index_dir / f"{self.index_type}_id_map.json")

    def _load(self):
        """Load FAISS index and ID mapping from disk."""
        index_path = self._index_path()
        map_path = self._map_path()

        if os.path.exists(index_path) and os.path.exists(map_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(map_path, "r") as f:
                    data = json.load(f)
                    self.id_map = {int(k): v for k, v in data["id_map"].items()}
                    self.next_id = data["next_id"]
                logger.info(
                    f"Loaded {self.index_type} index: {self.index.ntotal} vectors"
                )
            except Exception as e:
                logger.error(f"Failed to load {self.index_type} index: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.id_map = {}
                self.next_id = 0

    def save(self):
        """Persist FAISS index and ID mapping to disk."""
        with self.lock:
            faiss.write_index(self.index, self._index_path())
            with open(self._map_path(), "w") as f:
                json.dump(
                    {"id_map": {str(k): v for k, v in self.id_map.items()}, "next_id": self.next_id},
                    f,
                )
        logger.info(f"Saved {self.index_type} index: {self.index.ntotal} vectors")

    def add(self, user_id: str, embedding: np.ndarray) -> int:
        """Add an embedding for a user. Returns the internal FAISS id."""
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[1]}"
            )

        with self.lock:
            # Remove old embedding for this user if exists
            self.remove(user_id)

            internal_id = self.next_id
            self.index.add(embedding)
            self.id_map[internal_id] = user_id
            self.next_id += 1

        self.save()
        return internal_id

    def remove(self, user_id: str):
        """Remove all embeddings for a user (by rebuilding index without them)."""
        with self.lock:
            ids_to_remove = [k for k, v in self.id_map.items() if v == user_id]
            if not ids_to_remove:
                return

            # Rebuild index without the removed user's embeddings
            all_embeddings = []
            new_id_map = {}
            new_next_id = 0

            for old_id in sorted(self.id_map.keys()):
                if old_id in ids_to_remove:
                    continue
                # Reconstruct embedding from index
                embedding = self.index.reconstruct(old_id)
                all_embeddings.append(embedding)
                new_id_map[new_next_id] = self.id_map[old_id]
                new_next_id += 1

            self.index = faiss.IndexFlatL2(self.dimension)
            if all_embeddings:
                self.index.add(np.array(all_embeddings, dtype=np.float32))

            self.id_map = new_id_map
            self.next_id = new_next_id

    def search(self, embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Search for nearest neighbors.
        
        Returns list of {user_id, distance, similarity} sorted by distance.
        """
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

        with self.lock:
            if self.index.ntotal == 0:
                return []

            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(embedding, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            if idx in self.id_map:
                # Convert L2 distance to similarity score (0-1)
                similarity = max(0.0, 1.0 - dist / 4.0)  # Normalize assuming max L2 ~ 4.0
                results.append({
                    "user_id": self.id_map[idx],
                    "distance": dist,
                    "similarity": similarity,
                })

        return results

    def get_user_embedding(self, user_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a specific user."""
        with self.lock:
            for internal_id, uid in self.id_map.items():
                if uid == user_id:
                    return self.index.reconstruct(internal_id)
        return None

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal


class EmbeddingStore:
    """Manages multiple FAISS indices for different biometric types."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.indices: dict[str, FaissIndexManager] = {}

    def get_index(self, biometric_type: str, dimension: int) -> FaissIndexManager:
        """Get or create a FAISS index for a biometric type."""
        if biometric_type not in self.indices:
            self.indices[biometric_type] = FaissIndexManager(
                index_dir=self.base_dir,
                dimension=dimension,
                index_type=biometric_type,
            )
        return self.indices[biometric_type]

    @property
    def face_index(self) -> FaissIndexManager:
        return self.get_index("face", 512)

    @property
    def voice_index(self) -> FaissIndexManager:
        return self.get_index("voice", 192)

    @property
    def fingerprint_index(self) -> FaissIndexManager:
        return self.get_index("fingerprint", 256)
