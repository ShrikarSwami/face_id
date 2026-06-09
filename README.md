# face_id

Lightweight few-shot face verification library.  
Enroll 1–5 reference photos per person. Verify identities from images or live video.  
Portable across projects — drone sim, web app, robotics stack, or anything else.

---

## How it works

```
Reference photo(s)  →  detect  →  align  →  embed  →  save to gallery
Live frame          →  detect  →  align  →  embed  →  compare  →  match / no match
```

No retraining required. New identities are added by enrolling photos, not by modifying the model.  
The embedder (MobileFaceNet / ArcFace) was pretrained on millions of faces and generalizes well from just 1–2 reference images.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

See `models/README.md` for download instructions.

### 3. Enroll a reference identity

```bash
# Enroll one photo
python -m face_id.enroll alice photos/alice_front.jpg

# Enroll two photos (better robustness)
python -m face_id.enroll alice photos/alice_front.jpg photos/alice_angle.jpg

# List enrolled identities
python -m face_id.enroll --list
```

### 4. Verify a face

```bash
# Verify a single image
python -m face_id.verify test_photo.jpg

# Run live webcam verification (press Q to quit)
python -m face_id.verify --live
```

---

## Use as a library (drop into any project)

```python
from face_id import enroll, verify

# Enroll
enroll("alice", "photos/alice.jpg")

# Verify from file
result = verify("frame.jpg")
print(result)
# {"verified": True, "match": "alice", "confidence": 0.87, "face_found": True, "all_scores": {...}, "error": None}

# Verify from numpy array (e.g. in a video loop)
import cv2
from face_id import verify_from_array

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    result = verify_from_array(frame)
    if result["verified"]:
        print(f"Matched: {result['match']} ({result['confidence']:.2f})")
```

---

## Project structure

```
face_id/
  __init__.py          # public API: enroll, verify
  pipeline.py          # orchestrates the full flow
  config.py            # all settings — thresholds, paths, model config
  gallery_manager.py   # saves/loads enrolled embeddings
  enroll.py            # CLI enroll tool
  verify.py            # CLI verify tool + live webcam demo
  utils/
    detect.py          # RetinaFace ONNX wrapper
    align.py           # 5-point landmark alignment
    embed.py           # MobileFaceNet / ArcFace ONNX wrapper
    similarity.py      # cosine similarity + match decision
  models/
    README.md          # download instructions for ONNX weights
  gallery/             # enrolled embeddings saved here (gitignored)
  tests/
    test_enroll.py
    test_verify.py
  requirements.txt
```

---

## Tuning

All tunable settings are in `config.py`.

| Setting | Default | Effect |
|---------|---------|--------|
| `SIMILARITY_THRESHOLD` | 0.45 | Lower → more matches (more false positives). Higher → stricter. |
| `MIN_FACE_SIZE` | 40px | Ignore detections smaller than this. |
| `DETECTION_THRESHOLD` | 0.5 | Detector confidence cutoff. |
| `ONNX_PROVIDERS` | CUDA, CPU | Change to TensorRT for Jetson deployment. |

Calibrate the threshold on real data — don't guess. Collect a small set of
positive pairs (same person) and negative pairs (different people), run both
through verify(), and find the threshold that separates them cleanly.

---

## Deployment targets

| Target | Config change |
|--------|--------------|
| Desktop CPU | `ONNX_PROVIDERS = ["CPUExecutionProvider"]` |
| Desktop GPU | `ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]` |
| Jetson (TensorRT) | Convert to `.trt` engines, use `TensorrtExecutionProvider` |
| ROS 2 node | Call `verify_from_array(frame)` inside your image subscriber callback |

See `models/README.md` for TensorRT conversion commands.

---

## Running tests

```bash
pytest face_id/tests/
```

Tests mock the model calls so they run without ONNX weights downloaded.
