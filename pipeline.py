"""
pipeline.py — the public API for face_id.

This is the only file external projects need to import from.

    from face_id import enroll, verify

Two main functions:
    enroll(name, image_path_or_array)  →  {"enrolled": True, "name": ..., "count": ...}
    verify(image_path_or_array)        →  {"verified": bool, "match": str, "confidence": float, ...}
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple

from face_id import config
from face_id.utils.detect import detect_faces
from face_id.utils.align import align_face, crop_face
from face_id.utils.embed import compute_embedding
from face_id.utils.similarity import best_match
from face_id.gallery_manager import get_gallery


# ── Type alias ────────────────────────────────────────────────────────────────

ImageInput = Union[str, Path, np.ndarray]


def _load_image(source: ImageInput) -> Optional[np.ndarray]:
    """Accept a file path (str/Path) or a BGR numpy array."""
    if isinstance(source, np.ndarray):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def _process_image(
    image_bgr: np.ndarray,
    return_detection: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[dict]]]:
    """
    Full preprocessing pipeline for one image:
      detect → pick best face → align → embed

    Returns a 512-dim unit-norm embedding, or None if no usable face found.
    If return_detection=True, returns (embedding, detection_dict) or (None, None).
    """
    detections = detect_faces(image_bgr)
    if not detections:
        return (None, None) if return_detection else None

    # Use the highest-confidence detection
    best = detections[0]

    # Attempt landmark-based alignment (preferred)
    aligned = align_face(image_bgr, best.get("landmarks"))

    # Fallback: simple bbox crop if alignment fails
    if aligned is None:
        aligned = crop_face(image_bgr, best["bbox"])

    if aligned is None:
        return (None, None) if return_detection else None

    embedding = compute_embedding(aligned)
    if embedding is None:
        return (None, None) if return_detection else None

    if return_detection:
        return embedding, best
    return embedding


# ── Public API ────────────────────────────────────────────────────────────────

def enroll(name: str, image: ImageInput) -> dict:
    """
    Add a reference image for an identity.

    Call this once or twice per person before running verify().
    More reference images (up to ~5) improve robustness.

    Args:
        name:  Identity label, e.g. "alice" or "target_001".
        image: File path (str/Path) or BGR numpy array.

    Returns:
        {
            "enrolled":  bool  — True if face was found and saved,
            "name":      str   — identity name,
            "count":     int   — total reference images now enrolled for this person,
            "error":     str   — description if enrolled=False,
        }

    Example:
        enroll("alice", "photos/alice_front.jpg")
        enroll("alice", "photos/alice_angle.jpg")
    """
    try:
        image_bgr = _load_image(image)
    except (FileNotFoundError, ValueError) as e:
        return {"enrolled": False, "name": name, "count": 0, "error": str(e)}

    embedding = _process_image(image_bgr)
    if embedding is None:
        return {
            "enrolled": False,
            "name": name,
            "count": 0,
            "error": "No usable face detected in the enrollment image.",
        }

    gallery = get_gallery()
    gallery.add(name, embedding)
    gallery.save(name)

    return {
        "enrolled": True,
        "name": name,
        "count": gallery.count(name),
        "error": None,
    }


def verify(image: ImageInput, threshold: float = None) -> dict:
    """
    Check whether a face in the image matches any enrolled identity.

    Args:
        image:     File path (str/Path) or BGR numpy array (e.g. video frame).
        threshold: Override the similarity threshold from config.py.

    Returns:
        {
            "verified":    bool   — True if a match above threshold was found,
            "match":       str    — name of best match, or None,
            "confidence":  float  — cosine similarity of best match [0–1],
            "all_scores":  dict   — {name: score} for every enrolled identity,
            "face_found":  bool   — whether any face was detected at all,
            "error":       str    — description if something went wrong,
        }

    Example:
        result = verify("camera_frame.jpg")
        if result["verified"]:
            print(f"Matched {result['match']} with confidence {result['confidence']}")
    """
    t = threshold if threshold is not None else config.SIMILARITY_THRESHOLD

    try:
        image_bgr = _load_image(image)
    except (FileNotFoundError, ValueError) as e:
        return {
            "verified": False, "match": None, "confidence": 0.0,
            "all_scores": {}, "face_found": False, "error": str(e),
        }

    embedding = _process_image(image_bgr)
    if embedding is None:
        return {
            "verified": False, "match": None, "confidence": 0.0,
            "all_scores": {}, "face_found": False, "error": None,
        }

    gallery = get_gallery()
    prototypes = gallery.get_prototypes()

    if not prototypes:
        return {
            "verified": False, "match": None, "confidence": 0.0,
            "all_scores": {}, "face_found": True,
            "error": "Gallery is empty. Enroll at least one identity first.",
        }

    result = best_match(embedding, prototypes, threshold=t)
    result["face_found"] = True
    result["error"] = None
    return result


# Convenience aliases that accept numpy arrays directly (useful for video loops)
def enroll_from_array(name: str, image_bgr: np.ndarray) -> dict:
    return enroll(name, image_bgr)


def verify_from_array(image_bgr: np.ndarray, threshold: float = None) -> dict:
    return verify(image_bgr, threshold)


def verify_with_details(image_bgr: np.ndarray, threshold: float = None) -> dict:
    """
    Same as verify_from_array but also returns bbox and landmarks from the
    detection step so the caller can draw without re-running detection.

    Returns the standard verify dict plus:
        "bbox":      [x1, y1, x2, y2] or None
        "landmarks": np.ndarray or None
    """
    t = threshold if threshold is not None else config.SIMILARITY_THRESHOLD

    embedding, detection = _process_image(image_bgr, return_detection=True)
    if embedding is None:
        return {
            "verified": False, "match": None, "confidence": 0.0,
            "all_scores": {}, "face_found": False, "error": None,
            "bbox": None, "landmarks": None,
        }

    gallery = get_gallery()
    prototypes = gallery.get_prototypes()

    if not prototypes:
        return {
            "verified": False, "match": None, "confidence": 0.0,
            "all_scores": {}, "face_found": True,
            "error": "Gallery is empty. Enroll at least one identity first.",
            "bbox": detection["bbox"],
            "landmarks": detection.get("landmarks"),
        }

    result = best_match(embedding, prototypes, threshold=t)
    result["face_found"] = True
    result["error"] = None
    result["bbox"] = detection["bbox"]
    result["landmarks"] = detection.get("landmarks")
    return result
