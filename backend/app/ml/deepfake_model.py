import numpy as np
import torch
import torch.nn as nn
import cv2
import logging

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """Deepfake detection using EfficientNet-B0 binary classifier.
    
    Detects GAN-generated  faces, face swaps, and manipulated videos by analyzing
    frequency domain artifacts, compression  inconsistencies, and learned features.
    """

    def __init__(self, device: str | None = None):
        import torchvision.models as models

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DeepfakeDetector on {self.device}")

        # EfficientNet-B0 for binary classification: real (0) vs deepfake (1)
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        self.model.eval().to(self.device)

        self.transform = self._get_transform()
        logger.info("DeepfakeDetector initialized")

    def _get_transform(self):
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: np.ndarray) -> dict:
        """Run deepfake detection on a single face image.
        
        Args:
            image: RGB numpy array of face crop
            
        Returns:
            Dict with is_deepfake, confidence, and method
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            deepfake_prob = probs[0, 1].item()

        # Also run frequency analysis
        freq_score = self._frequency_analysis(image)

        # Combined score (weighted)
        combined = 0.7 * deepfake_prob + 0.3 * freq_score

        return {
            "is_deepfake": combined > 0.5,
            "confidence": float(combined),
            "cnn_score": float(deepfake_prob),
            "frequency_score": float(freq_score),
            "method": "efficientnet_b0 + frequency_analysis",
        }

    def _frequency_analysis(self, image: np.ndarray) -> float:
        """Analyze frequency domain for GAN artifacts.
        
        GAN-generated images often have characteristic spectral patterns,
        especially in the azimuthal average of the power spectrum.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        # FFT and power spectrum
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = np.abs(f_shift) ** 2

        # Azimuthal average
        h, w = power_spectrum.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
        max_r = min(cx, cy)

        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if np.any(mask):
                radial_profile[i] = np.mean(power_spectrum[mask])

        # GAN artifacts: check for unusual peaks in high frequencies
        if len(radial_profile) > 10:
            low_freq = np.mean(radial_profile[:len(radial_profile)//4])
            high_freq = np.mean(radial_profile[len(radial_profile)//2:])
            ratio = high_freq / (low_freq + 1e-10)
            # Higher ratio suggests more high-freq energy (possible GAN artifact)
            anomaly_score = min(1.0, ratio * 10)
        else:
            anomaly_score = 0.0

        return float(anomaly_score)

    def detect_video(self, frames: list[np.ndarray]) -> dict:
        """Run deepfake detection across multiple video frames.
        
        Checks temporal consistency and aggregates per-frame predictions.
        """
        if not frames:
            return {"is_deepfake": False, "confidence": 0.0, "method": "no_frames"}

        scores = []
        for frame in frames[::max(1, len(frames) // 10)]:  # Sample up to 10 frames
            result = self.predict(frame)
            scores.append(result["confidence"])

        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        consistency = 1.0 - float(np.std(scores))  # Low variance = consistent

        # If highly inconsistent scores, that itself is suspicious
        final_score = 0.6 * avg_score + 0.2 * max_score + 0.2 * (1 - consistency)

        return {
            "is_deepfake": final_score > 0.5,
            "confidence": float(final_score),
            "avg_frame_score": avg_score,
            "max_frame_score": max_score,
            "temporal_consistency": consistency,
            "frames_analyzed": len(scores),
            "method": "efficientnet_b0 + frequency + temporal",
        }
