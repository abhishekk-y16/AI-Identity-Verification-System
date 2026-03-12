import numpy as np
import torch
import torchaudio
import logging
import io

# Monkey-patch torchaudio for SpeechBrain compatibility (removed in torchaudio >=2.10)
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: None
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda _: None

logger = logging.getLogger(__name__)


class VoiceEmbedder:
    """Voice embedding extraction using SpeechBrain ECAPA-TDNN.
    
    Produces 192-dimensional speaker embeddings from audio signals.
    """

    def __init__(self, device: str | None = None):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing VoiceEmbedder on {self.device}")

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )
        self.target_sample_rate = 16000
        logger.info("VoiceEmbedder initialized successfully")

    def preprocess_audio(self, audio_bytes: bytes, original_sr: int | None = None) -> torch.Tensor:
        """Load and preprocess audio bytes to 16kHz mono waveform.
        
        Args:
            audio_bytes: Raw audio file bytes (WAV, WebM, etc.)
            original_sr: Original sample rate (auto-detected if None)
            
        Returns:
            Mono waveform tensor at 16kHz
        """
        import soundfile as sf
        from torchaudio.transforms import Resample

        audio_buffer = io.BytesIO(audio_bytes)

        try:
            # Try soundfile first (handles WAV, FLAC, OGG)
            data, sample_rate = sf.read(audio_buffer, dtype="float32")
        except Exception:
            # Fall back to pydub for formats like WebM/MP4
            from pydub import AudioSegment
            audio_buffer.seek(0)
            segment = AudioSegment.from_file(audio_buffer)
            segment = segment.set_channels(1)
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2 ** (segment.sample_width * 8 - 1))
            data = samples
            sample_rate = segment.frame_rate

        # Convert to torch tensor
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            # Multi-channel: average to mono
            waveform = torch.from_numpy(data.T.mean(axis=0, keepdims=True))

        # Resample to 16kHz
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Trim silence using energy-based VAD
        waveform = self._trim_silence(waveform)

        return waveform

    def _trim_silence(self, waveform: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Remove leading and trailing silence from waveform."""
        audio = waveform.squeeze()
        energy = torch.abs(audio)
        mask = energy > threshold
        if mask.any():
            indices = torch.where(mask)[0]
            start, end = indices[0], indices[-1] + 1
            audio = audio[start:end]
        return audio.unsqueeze(0)

    def get_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract 192-dimensional L2-normalized speaker embedding.
        
        Args:
            waveform: Preprocessed mono waveform tensor
            
        Returns:
            L2-normalized 192-dim numpy array
        """
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform.to(self.device))

        embedding = embedding.squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def extract_embedding(self, audio_bytes: bytes) -> np.ndarray:
        """Full pipeline: audio bytes → preprocess → embedding.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            192-dim normalized embedding
        """
        waveform = self.preprocess_audio(audio_bytes)
        return self.get_embedding(waveform)

    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two voice embeddings."""
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))
