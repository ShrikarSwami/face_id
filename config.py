"""
config.py — all tunable settings in one place.

When adapting face_id for a new project (drone, web app, etc.),
this is the only file you should need to touch for most changes.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent

# Where ONNX model weights live. See models/README.md for download instructions.
MODELS_DIR = ROOT / "models"

# Where enrolled identity embeddings are saved (.npy files).
GALLERY_DIR = ROOT / "gallery"

# UI assets (app icon, etc.)
ASSETS_DIR = ROOT / "assets"
APP_ICON = ASSETS_DIR / "appIcon.png"

# Model filenames inside MODELS_DIR
DETECTOR_MODEL   = MODELS_DIR / "det_500m.onnx"    # RetinaFace (light variant)
EMBEDDER_MODEL   = MODELS_DIR / "w600k_mbf.onnx"   # MobileFaceNet / ArcFace

# ── Detection settings ─────────────────────────────────────────────────────────

# Minimum face size in pixels to attempt verification.
# Smaller crops are low quality and produce unreliable embeddings.
MIN_FACE_SIZE = 40  # px (width or height)

# Detector confidence threshold — detections below this are ignored.
DETECTION_THRESHOLD = 0.5

# Input size expected by the detector model.
DETECTOR_INPUT_SIZE = (640, 640)

# ── Embedding settings ─────────────────────────────────────────────────────────

# Input size expected by the embedder model.
EMBEDDER_INPUT_SIZE = (112, 112)

# ── Verification / similarity settings ────────────────────────────────────────

# Cosine similarity threshold for a positive match.
# 0.0 = completely different, 1.0 = identical.
# Tune this on your calibration set — don't guess.
#   Loose  (more matches, more false positives): ~0.35
#   Default (balanced):                          ~0.45
#   Strict  (fewer false positives):             ~0.55
SIMILARITY_THRESHOLD = 0.45

# When multiple reference embeddings exist for one identity,
# use the mean prototype embedding for comparison.
USE_PROTOTYPE = True

# ── Runtime ────────────────────────────────────────────────────────────────────

# Execution provider for ONNX Runtime.
# Apple Silicon (M1/M2/M3/M4): use CoreML + CPU
# NVIDIA GPU (gaming PC, Jetson): use CUDA + CPU
# Jetson TensorRT: use TensorRT + CPU
# CPU only fallback: ["CPUExecutionProvider"]
#
# CoreML may require: pip install onnxruntime-silicon
# (standard onnxruntime may include CoreML depending on version)
ONNX_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# ── Quality gating ─────────────────────────────────────────────────────────────

# Minimum landmark confidence before attempting alignment.
# Faces where landmarks are not reliable produce bad alignments.
MIN_LANDMARK_SCORE = 0.6

# Maximum allowed head pose angle (degrees) for a verification attempt.
# Side profiles produce unreliable embeddings.
MAX_YAW_DEGREES = 45
