"""
utils/detect.py — face detection and landmark extraction.

Returns bounding boxes and 5-point landmarks using an InsightFace
SCRFD / RetinaFace ONNX model (det_500m.onnx from buffalo_sc).
Landmarks are used downstream for geometric alignment before embedding.
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Optional

from face_id import config


def _distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape=None) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray, max_shape=None) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> List[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class FaceDetector:
    """
    Wraps an InsightFace detector ONNX model (SCRFD det_500m or legacy RetinaFace).

    Returns a list of detections, each a dict:
        {
            "bbox":       [x1, y1, x2, y2],   # pixels in original image space
            "landmarks":  np.ndarray shape (5,2),  # (x, y) for each landmark
            "confidence": float
        }
    """

    def __init__(self):
        if not config.DETECTOR_MODEL.exists():
            raise FileNotFoundError(
                f"Detector model not found at {config.DETECTOR_MODEL}\n"
                f"See face_id/models/README.md for download instructions."
            )
        try:
            self._session = ort.InferenceSession(
                str(config.DETECTOR_MODEL),
                providers=config.ONNX_PROVIDERS,
            )
        except Exception:
            if "CoreMLExecutionProvider" in config.ONNX_PROVIDERS:
                print("CoreML provider unavailable, falling back to CPU")
            self._session = ort.InferenceSession(
                str(config.DETECTOR_MODEL),
                providers=["CPUExecutionProvider"],
            )
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = config.DETECTOR_INPUT_SIZE  # (W, H)
        self._num_outputs = len(self._session.get_outputs())
        self._scrfd = self._num_outputs == 9
        self._center_cache = {}
        self._nms_thresh = 0.4

    def detect(self, image_bgr: np.ndarray) -> List[dict]:
        """
        Detect all faces in a BGR image (as returned by cv2.imread).

        Args:
            image_bgr: H x W x 3 uint8 BGR image.

        Returns:
            List of detection dicts, sorted by confidence descending.
            Empty list if no faces found above threshold.
        """
        if self._scrfd:
            blob, det_scale = self._preprocess_scrfd(image_bgr)
            outputs = self._session.run(None, {self._input_name: blob})
            detections = self._postprocess_scrfd(outputs, det_scale)
        else:
            orig_h, orig_w = image_bgr.shape[:2]
            blob, scale_x, scale_y = self._preprocess_retina(image_bgr)
            outputs = self._session.run(None, {self._input_name: blob})
            detections = self._postprocess_retina(outputs, scale_x, scale_y)

        detections = [
            d for d in detections
            if (d["bbox"][2] - d["bbox"][0]) >= config.MIN_FACE_SIZE
            and (d["bbox"][3] - d["bbox"][1]) >= config.MIN_FACE_SIZE
        ]
        return sorted(detections, key=lambda d: d["confidence"], reverse=True)

    def _preprocess_scrfd(self, image_bgr: np.ndarray):
        """Aspect-ratio preserving resize + pad for SCRFD (det_500m.onnx)."""
        input_w, input_h = self._input_size
        im_ratio = float(image_bgr.shape[0]) / image_bgr.shape[1]
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_height = input_h
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_w
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image_bgr.shape[0]
        resized = cv2.resize(image_bgr, (new_width, new_height))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / 128.0,
            (input_w, input_h),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob, det_scale

    def _postprocess_scrfd(self, outputs, det_scale: float) -> List[dict]:
        """Decode SCRFD 9-head outputs (buffalo_sc det_500m.onnx).

        Output layout (3 FPN scales merged before NMS):
          outputs[0..2]   scores  — strides 8/16/32 (12800/3200/800 anchors)
          outputs[3..5]   bboxes — distance deltas per scale
          outputs[6..8]   landmarks (5 pts) per scale
        """
        input_h = self._input_size[1]
        input_w = self._input_size[0]
        fmc = 3
        strides = [8, 16, 32]
        num_anchors = 2
        threshold = config.DETECTION_THRESHOLD

        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(strides):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * stride
            kps_preds = outputs[idx + fmc * 2] * stride

            height = input_h // stride
            width = input_w // stride
            key = (height, width, stride)
            if key in self._center_cache:
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self._center_cache) < 100:
                    self._center_cache[key] = anchor_centers

            pos_inds = np.where(scores.reshape(-1) >= threshold)[0]
            if pos_inds.size == 0:
                continue

            bboxes = _distance2bbox(
                anchor_centers, bbox_preds,
                max_shape=(input_h, input_w),
            )
            pos_scores = scores.reshape(-1)[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kps_decoded = _distance2kps(
                anchor_centers, kps_preds,
                max_shape=(input_h, input_w),
            )
            kpss = kps_decoded.reshape((kps_decoded.shape[0], -1, 2))
            kpss_list.append(kpss[pos_inds])

        if not scores_list:
            return []

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list) / det_scale
        kpss = np.concatenate(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32)
        order = scores.argsort()[::-1]
        pre_det = pre_det[order]
        kpss = kpss[order]

        keep = _nms(pre_det, self._nms_thresh)
        detections = []
        for i in keep:
            x1, y1, x2, y2, conf = pre_det[i]
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "landmarks": kpss[i].astype(np.float32),
                "confidence": float(conf),
            })
        return detections

    def _preprocess_retina(self, image_bgr: np.ndarray):
        """Legacy RetinaFace 3-output preprocess."""
        target_w, target_h = self._input_size
        orig_h, orig_w = image_bgr.shape[:2]
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        resized = cv2.resize(image_bgr, (target_w, target_h))
        blob = resized.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        return blob, scale_x, scale_y

    def _postprocess_retina(self, outputs, scale_x, scale_y) -> List[dict]:
        """Legacy RetinaFace 3-output postprocess."""
        detections = []
        try:
            scores = outputs[0]
            bboxes = outputs[1]
            landmarks = outputs[2]
            for i in range(len(scores)):
                conf = float(scores[i][0])
                if conf < config.DETECTION_THRESHOLD:
                    continue
                x1, y1, x2, y2 = bboxes[i]
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                lm = landmarks[i].reshape(5, 2)
                lm[:, 0] *= scale_x
                lm[:, 1] *= scale_y
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "landmarks": lm.astype(np.float32),
                    "confidence": conf,
                })
        except (IndexError, TypeError):
            pass
        return detections


# Module-level singleton so the model loads once per process
_detector: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def detect_faces(image_bgr: np.ndarray) -> List[dict]:
    """Convenience function — detect faces using the shared detector instance."""
    return get_detector().detect(image_bgr)
