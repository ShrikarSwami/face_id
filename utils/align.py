"""
utils/align.py — geometric face alignment using 5-point landmarks.

Alignment normalizes the face crop so that eyes and mouth are always
in the same position before the embedding model sees them.
Skipping this step meaningfully degrades similarity scores.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from face_id import config


# Reference landmark positions inside the 112x112 ArcFace canonical crop.
# These are the standard InsightFace / ArcFace alignment targets.
ARCFACE_REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = None,
) -> Optional[np.ndarray]:
    """
    Warp a face crop to canonical alignment using 5-point landmarks.

    Args:
        image_bgr:   Full original image (H x W x 3 BGR).
        landmarks:   5 x 2 float32 array of (x, y) landmark coordinates
                     in original image space, as returned by detect.py.
        output_size: (width, height) of output crop. Defaults to
                     config.EMBEDDER_INPUT_SIZE.

    Returns:
        Aligned face crop as uint8 BGR, or None if alignment fails.
    """
    if output_size is None:
        output_size = config.EMBEDDER_INPUT_SIZE  # (112, 112)

    if landmarks is None or landmarks.shape != (5, 2):
        return None

    try:
        # Estimate the similarity transform (rotation + scale + translation)
        # that maps the detected landmarks → reference landmark positions.
        transform = cv2.estimateAffinePartial2D(
            landmarks,
            ARCFACE_REFERENCE_LANDMARKS,
            method=cv2.LMEDS,
        )[0]

        if transform is None:
            return None

        aligned = cv2.warpAffine(
            image_bgr,
            transform,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned

    except cv2.error:
        return None


def crop_face(
    image_bgr: np.ndarray,
    bbox: list,
    padding: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Simple bounding-box crop with padding, as a fallback when landmarks
    are unavailable or alignment fails.

    Args:
        image_bgr: Full original image.
        bbox:      [x1, y1, x2, y2] in pixel coordinates.
        padding:   Fractional padding to add around the box (0.2 = 20%).

    Returns:
        Cropped face region, or None if the crop is invalid.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, config.EMBEDDER_INPUT_SIZE)
