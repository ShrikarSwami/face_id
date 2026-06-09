# Model Weights

This directory is intentionally empty in version control.
Download the ONNX model files and place them here before running face_id.

---

## Required files

| File | Model | Size | Source |
|------|-------|------|--------|
| `det_500m.onnx` | RetinaFace (light) | ~1 MB | InsightFace |
| `w600k_mbf.onnx` | MobileFaceNet / ArcFace | ~4 MB | InsightFace |

---

## Download instructions

### Option A — InsightFace model zoo (recommended)

```python
# Run this once in a Python shell from the project root:
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_sc")   # small / CPU-friendly pack
app.prepare(ctx_id=0, det_size=(640, 640))
```

InsightFace will download models to `~/.insightface/models/buffalo_sc/`.
Copy the relevant `.onnx` files from there into this directory.

The `buffalo_sc` pack contains:
- `det_500m.onnx`  — detector
- `w600k_mbf.onnx` — embedder

### Option B — Direct download

From the InsightFace model zoo:
https://github.com/deepinsight/insightface/tree/master/model_zoo

Look for the `buffalo_sc` or `buffalo_l` pack.

---

## Using a different model

If you swap models, update `config.py`:
- `DETECTOR_MODEL` — path to detector ONNX
- `EMBEDDER_MODEL` — path to embedder ONNX
- `DETECTOR_INPUT_SIZE` — check model spec
- `EMBEDDER_INPUT_SIZE` — check model spec (112x112 for most ArcFace variants)

Also verify the output node parsing in `utils/detect.py → _postprocess()`.
Different RetinaFace exports may have different output node names/shapes.

---

## Jetson / TensorRT deployment

For Jetson deployment, convert the ONNX files to TensorRT engines on the target device:

```bash
trtexec --onnx=det_500m.onnx --saveEngine=det_500m.trt --fp16
trtexec --onnx=w600k_mbf.onnx --saveEngine=w600k_mbf.trt --fp16
```

Then update `config.py` to use `TensorrtExecutionProvider` and point to the `.trt` files.
See Phase 5 docs for the full Jetson deployment guide.
