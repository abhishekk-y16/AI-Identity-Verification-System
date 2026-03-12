import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class FingerprintProcessor:
    """Fingerprint feature extraction using OpenCV image processing pipeline.
    
    Pipeline: grayscale → enhance (CLAHE + Gabor) → binarize → thin → minutiae detection.
    Also supports CNN-based embedding extraction via a ResNet18 feature backbone.
    """

    def __init__(self, use_cnn: bool = False, device: str | None = None):
        self.use_cnn = use_cnn
        self.device = device or "cpu"

        if use_cnn:
            self._init_cnn_model()

        logger.info(f"FingerprintProcessor initialized (CNN={use_cnn})")

    def _init_cnn_model(self):
        """Initialize ResNet18 as a feature extractor for fingerprint embeddings."""
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final classification layer, use 512-dim feature vector
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Enhance fingerprint image for feature extraction."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Resize to standard size
        gray = cv2.resize(gray, (300, 300))

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

        return enhanced

    def extract_minutiae(self, image: np.ndarray) -> dict:
        """Extract minutiae features from fingerprint image.
        
        Returns dict with minutiae points (x, y, angle, type) and ridge count.
        """
        enhanced = self.preprocess(image)

        # Adaptive threshold for binarization
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )

        # Morphological thinning (skeletonization)
        skeleton = self._skeletonize(binary)

        # Detect minutiae from skeleton
        minutiae = self._detect_minutiae_points(skeleton)

        return {
            "minutiae_count": len(minutiae),
            "minutiae_points": minutiae,
            "skeleton_density": float(np.sum(skeleton > 0)) / skeleton.size,
        }

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Morphological skeletonization of binary fingerprint image."""
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()

        while True:
            eroded = cv2.erode(temp, element)
            dilated = cv2.dilate(eroded, element)
            diff = cv2.subtract(temp, dilated)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

        return skeleton

    def _detect_minutiae_points(self, skeleton: np.ndarray) -> list[dict]:
        """Detect ridge endings and bifurcations from skeletonized fingerprint."""
        minutiae = []
        h, w = skeleton.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] == 0:
                    continue

                # Count transitions in 8-connected neighborhood
                neighbors = [
                    skeleton[y - 1, x], skeleton[y - 1, x + 1],
                    skeleton[y, x + 1], skeleton[y + 1, x + 1],
                    skeleton[y + 1, x], skeleton[y + 1, x - 1],
                    skeleton[y, x - 1], skeleton[y - 1, x - 1],
                ]
                cn = sum(neighbors[i] != neighbors[(i + 1) % 8] for i in range(8)) // 2

                if cn == 1:  # Ridge ending
                    minutiae.append({"x": x, "y": y, "type": "ending", "cn": cn})
                elif cn == 3:  # Bifurcation
                    minutiae.append({"x": x, "y": y, "type": "bifurcation", "cn": cn})

        return minutiae[:200]  # Limit to top 200 minutiae

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract a fixed-size embedding from fingerprint image.
        
        Uses CNN if configured, otherwise creates a feature vector from minutiae.
        """
        if self.use_cnn:
            return self._get_cnn_embedding(image)
        return self._get_minutiae_embedding(image)

    def _get_cnn_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract 512-dim embedding using ResNet18."""
        import torch

        enhanced = self.preprocess(image)
        tensor = self.transform(enhanced).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        embedding = embedding.squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _get_minutiae_embedding(self, image: np.ndarray) -> np.ndarray:
        """Create a fixed-size feature vector from minutiae spatial distribution."""
        minutiae_data = self.extract_minutiae(image)
        points = minutiae_data["minutiae_points"]

        # Create a 256-dim feature vector from minutiae spatial histogram
        embedding = np.zeros(256, dtype=np.float32)

        if not points:
            return embedding

        # Spatial histogram (16x16 grid)
        for p in points:
            gx = min(int(p["x"] / 300 * 16), 15)
            gy = min(int(p["y"] / 300 * 16), 15)
            idx = gy * 16 + gx
            if p["type"] == "ending":
                embedding[idx] += 1.0
            else:
                embedding[idx] += 2.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two fingerprint embeddings."""
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
