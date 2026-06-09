"""
utils/embed.py — face embedding extraction.

Converts an aligned 112x112 face crop into a 512-dim unit-norm vector.
Embeddings are comparable across identities using cosine similarity.
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import Optional

from face_id import config


class FaceEmbedder:
    """
    Wraps a MobileFaceNet / ArcFace ONNX model.

    Input:  aligned 112x112 BGR face crop (uint8 or float32)
    Output: 512-dimensional L2-normalised embedding vector (float32)
    """

    def __init__(self):
        if not config.EMBEDDER_MODEL.exists():
            raise FileNotFoundError(
                f"Embedder model not found at {config.EMBEDDER_MODEL}\n"
                f"See face_id/models/README.md for download instructions."
            )
        try:
            self._session = ort.InferenceSession(
                str(config.EMBEDDER_MODEL),
                providers=config.ONNX_PROVIDERS,
            )
        except Exception:
            if "CoreMLExecutionProvider" in config.ONNX_PROVIDERS:
                print("CoreML provider unavailable, falling back to CPU")
            self._session = ort.InferenceSession(
                str(config.EMBEDDER_MODEL),
                providers=["CPUExecutionProvider"],
            )
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, aligned_face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the embedding for one aligned face crop.

        Args:
            aligned_face_bgr: 112 x 112 x 3 uint8 BGR image.

        Returns:
            512-dim float32 unit-norm vector, or None on failure.
        """
        blob = self._preprocess(aligned_face_bgr)
        if blob is None:
            return None

        try:
            outputs = self._session.run(None, {self._input_name: blob})
            embedding = outputs[0][0]  # shape (512,)
            return self._l2_normalize(embedding)
        except Exception:
            return None

    def _preprocess(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Resize, normalize, and reformat for model input."""
        target_w, target_h = config.EMBEDDER_INPUT_SIZE

        if face_bgr is None or face_bgr.size == 0:
            return None

        if face_bgr.shape[:2] != (target_h, target_w):
            face_bgr = cv2.resize(face_bgr, (target_w, target_h))

        face = face_bgr.astype(np.float32)
        face = (face - 127.5) / 128.0          # normalize to [-1, 1]
        face = face[:, :, ::-1]                 # BGR → RGB
        face = face.transpose(2, 0, 1)          # HWC → CHW
        face = np.expand_dims(face, axis=0)     # CHW → NCHW
        return face

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize to unit norm so cosine similarity = dot product."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm


# Module-level singleton
_embedder: Optional[FaceEmbedder] = None


def get_embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FaceEmbedder()
    return _embedder


def compute_embedding(aligned_face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Convenience function using the shared embedder instance."""
    return get_embedder().embed(aligned_face_bgr)
