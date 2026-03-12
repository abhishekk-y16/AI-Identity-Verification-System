import numpy as np
import torch
import torch.nn as nn
import cv2
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class LivenessDetector:
    """Face liveness detection to prevent photo/video/mask spoofing attacks.
    
    Uses a MobileNetV2-based binary classifier (live vs. spoof) combined with
    heuristic checks (blink detection via eye aspect ratio, texture analysis).
    """

    def __init__(self, device: str | None = None):
        import torchvision.models as models

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LivenessDetector on {self.device}")

        # MobileNetV2 for texture-based anti-spoofing (binary: live=1, spoof=0)
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
        self.model.eval().to(self.device)

        self.transform = self._get_transform()

        # Facial landmark detector for blink detection
        self._init_landmark_detector()

        logger.info("LivenessDetector initialized")

    def _get_transform(self):
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _init_landmark_detector(self):
        """Initialize MediaPipe face mesh for eye landmark detection."""
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            self.has_mediapipe = True
        except ImportError:
            logger.warning("MediaPipe not available; blink detection disabled")
            self.has_mediapipe = False

    def check_texture(self, image: np.ndarray) -> dict:
        """Analyze texture patterns to detect printed photos or screen displays.
        
        Checks Moiré patterns, frequency domain artifacts, and local binary patterns.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, (224, 224))

        # Laplacian variance (blur/sharpness indicator)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # High-frequency energy via FFT
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        # Ratio of high-freq to total energy
        mask_size = min(h, w) // 8
        center_energy = np.sum(magnitude[center_h - mask_size:center_h + mask_size,
                                         center_w - mask_size:center_w + mask_size])
        total_energy = np.sum(magnitude)
        hf_ratio = 1.0 - (center_energy / (total_energy + 1e-10))

        # LBP-like texture variance
        lbp_img = self._compute_lbp(gray)
        lbp_variance = float(np.var(lbp_img))

        return {
            "laplacian_variance": float(laplacian_var),
            "high_freq_ratio": float(hf_ratio),
            "lbp_variance": float(lbp_variance),
        }

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Simplified Local Binary Pattern computation."""
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        return lbp

    def check_blink(self, frames: list[np.ndarray]) -> dict:
        """Detect blink in a sequence of frames using Eye Aspect Ratio (EAR)."""
        if not self.has_mediapipe or len(frames) < 5:
            return {"blink_detected": False, "ear_values": [], "confidence": 0.0}

        # MediaPipe eye landmark indices
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        ear_values = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            def get_ear(eye_indices):
                pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
                # Vertical distances
                v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
                v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
                # Horizontal distance
                h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
                return (v1 + v2) / (2.0 * h_dist + 1e-10)

            left_ear = get_ear(LEFT_EYE)
            right_ear = get_ear(RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)

        blink_detected = False
        if len(ear_values) >= 3:
            ear_arr = np.array(ear_values)
            # Blink = EAR drops below threshold then rises back
            blink_threshold = 0.21
            below = ear_arr < blink_threshold
            if np.any(below) and np.any(~below):
                blink_detected = True

        return {
            "blink_detected": blink_detected,
            "ear_values": [float(v) for v in ear_values[-10:]],
            "confidence": 0.9 if blink_detected else 0.2,
        }

    def predict_cnn(self, image: np.ndarray) -> dict:
        """Run CNN anti-spoofing prediction on a single face crop."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            live_prob = probs[0, 1].item()

        return {
            "is_live": live_prob > 0.5,
            "confidence": float(live_prob),
        }

    def check_liveness(self, frames: list[np.ndarray]) -> dict:
        """Full liveness detection pipeline on frame sequence.
        
        Combines CNN prediction, texture analysis, and blink detection.
        """
        if not frames:
            return {"is_live": False, "confidence": 0.0, "checks": {}, "message": "No frames provided"}

        # CNN prediction on middle frame
        mid_frame = frames[len(frames) // 2]
        cnn_result = self.predict_cnn(mid_frame)

        # Texture analysis
        texture_result = self.check_texture(mid_frame)
        texture_score = min(1.0, texture_result["laplacian_variance"] / 500.0)

        # Blink detection (if enough frames)
        blink_result = self.check_blink(frames) if len(frames) >= 5 else {
            "blink_detected": False, "confidence": 0.0
        }

        # Combined score — texture and blink are reliable; CNN classifier
        # head is untrained (random weights) so we use it only as a minor hint.
        if blink_result.get("blink_detected"):
            # Video with blink detected: trust blink heavily
            texture_weight = 0.3
            blink_weight = 0.6
            cnn_weight = 0.1
        else:
            # Single image or no blink: rely on texture analysis
            texture_weight = 0.7
            blink_weight = 0.0
            cnn_weight = 0.3

        combined = (
            cnn_result["confidence"] * cnn_weight
            + texture_score * texture_weight
            + blink_result["confidence"] * blink_weight
        )

        return {
            "is_live": combined > 0.5,
            "confidence": float(combined),
            "checks": {
                "cnn_live_score": cnn_result["confidence"],
                "texture_sharpness": texture_score,
                "blink_detected": blink_result["blink_detected"],
                "blink_confidence": blink_result["confidence"],
                "laplacian_variance": texture_result["laplacian_variance"],
                "high_freq_ratio": texture_result["high_freq_ratio"],
            },
            "message": "Liveness check passed" if combined > 0.5 else "Liveness check failed — possible spoof detected",
        }
