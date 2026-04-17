import numpy as np
import torch
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Face detection, alignment, and embedding extraction using facenet-pytorch.
    
    Uses MTCNN for face detection / alignment and InceptionResnetV1 for 512-dim embeddings.
    """

    def __init__(self, device: str | None = None):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing FaceEmbedder on {self.device}")

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False,  # return only the most confident face
        )

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        logger.info("FaceEmbedder initialized successfully")

    def detect_and_align(self, image: np.ndarray) -> np.ndarray | None:
        """Detect face in image, align and crop to 160x160.
        
        Args:
            image: RGB numpy array (H, W, 3)
            
        Returns:
            Aligned face tensor or None if no face detected
        """
        pil_image = Image.fromarray(image)
        face_tensor = self.mtcnn(pil_image)
        if face_tensor is None:
            return None
        return face_tensor

    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Extract 512-dimensional L2-normalized embedding from aligned face.
        
        Args:
            face_tensor: Aligned face tensor from detect_and_align()
            
        Returns:
            L2-normalized 512-dim numpy array
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)

        with torch.no_grad():
            embedding = self.model(face_tensor)

        embedding = embedding.cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def extract_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """Full pipeline: image → detect → align → embedding.
        
        Args:
            image: RGB numpy array
            
        Returns:
            512-dim normalized embedding or None if no face detected
        """
        face_tensor = self.detect_and_align(image)
        if face_tensor is None:
            logger.warning("No face detected in image")
            return None
        return self.get_embedding(face_tensor)

    @staticmethod
    def compute_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute Euclidean distance between two embeddings."""
        return float(np.linalg.norm(embedding1 - embedding2))

    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings (0 to 1)."""
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
