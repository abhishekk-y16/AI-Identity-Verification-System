import logging
import numpy as np
import cv2
from app.ml.face_model import FaceEmbedder

logger = logging.getLogger(__name__)


class DocumentService:
    """KYC document verification — extract face from ID document, OCR text, compare with selfie."""

    def __init__(self, face_embedder: FaceEmbedder):
        self.face_embedder = face_embedder
        self._init_ocr()

    def _init_ocr(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=False)
            self.has_ocr = True
        except ImportError:
            logger.warning("EasyOCR not installed; OCR functionality disabled")
            self.has_ocr = False

    def detect_document_type(self, text: str) -> str:
        """Simple heuristic to detect document type from OCR text."""
        text_lower = text.lower()
        if "passport" in text_lower:
            return "passport"
        elif "aadhaar" in text_lower or "unique identification" in text_lower:
            return "aadhaar"
        elif "driver" in text_lower and "licen" in text_lower:
            return "drivers_license"
        elif "voter" in text_lower or "election" in text_lower:
            return "voter_id"
        elif "pan" in text_lower and ("income tax" in text_lower or "permanent account" in text_lower):
            return "pan_card"
        return "unknown"

    def extract_text(self, image: np.ndarray) -> dict:
        """Extract text from document image via OCR."""
        if not self.has_ocr:
            return {"raw_text": "", "lines": [], "document_type": "unknown"}

        results = self.reader.readtext(image)
        lines = [text for _, text, conf in results if conf > 0.3]
        raw_text = " ".join(lines)
        doc_type = self.detect_document_type(raw_text)

        return {
            "raw_text": raw_text,
            "lines": lines,
            "document_type": doc_type,
        }

    async def verify_document(
        self, document_image: np.ndarray, selfie_image: np.ndarray
    ) -> dict:
        """Compare face on ID document with live selfie face.
        
        Steps:
        1. Extract face from document image
        2. Extract face from selfie
        3. Compare embeddings
        4. Extract OCR text from document
        """
        # Extract face from document
        doc_embedding = self.face_embedder.extract_embedding(document_image)
        if doc_embedding is None:
            return {
                "document_type": "unknown",
                "extracted_name": None,
                "face_match_score": 0.0,
                "face_match": False,
                "ocr_data": {},
                "message": "No face detected in the document image",
            }

        # Extract face from selfie
        selfie_embedding = self.face_embedder.extract_embedding(selfie_image)
        if selfie_embedding is None:
            return {
                "document_type": "unknown",
                "extracted_name": None,
                "face_match_score": 0.0,
                "face_match": False,
                "ocr_data": {},
                "message": "No face detected in the selfie",
            }

        # Compare
        similarity = self.face_embedder.compute_similarity(doc_embedding, selfie_embedding)
        face_match = similarity >= 0.5  # Slightly lower threshold for document photos

        # OCR
        ocr_data = self.extract_text(document_image)

        # Try to extract name (first non-document-type multi-word line)
        extracted_name = None
        for line in ocr_data.get("lines", []):
            words = line.strip().split()
            if len(words) >= 2 and all(w.isalpha() for w in words):
                extracted_name = line.strip()
                break

        return {
            "document_type": ocr_data["document_type"],
            "extracted_name": extracted_name,
            "face_match_score": float(similarity),
            "face_match": face_match,
            "ocr_data": ocr_data,
            "message": "Document face matches selfie" if face_match else "Document face does NOT match selfie",
        }
