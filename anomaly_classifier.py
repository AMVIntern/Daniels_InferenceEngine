# anomaly_classifier.py
"""
Extensible anomaly detection framework.

Architecture
------------
AnomalyDetection        — single detection box with bbox, score, class.
AnomalyResult           — Pydantic model returned by every classifier.
AnomalyClassifierBase   — Protocol every classifier must satisfy.
AnomalyRegistry         — Holds all classifiers; runs them all on an image.
YoloxAnomalyClassifier  — YOLOX ONNX implementation matching training preprocessing.

Adding a new anomaly classifier
--------------------------------
1. Implement AnomalyClassifierBase (or instantiate YoloxAnomalyClassifier with new weights).
2. Call anomaly_registry.register(your_classifier) in fill_estimator_api.py lifespan.
No pipeline or schema code needs to change.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import cv2
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel

# Fixed input geometry — must match training
_INPUT_SIZE = (1024, 1024)   # (H, W)
_STRIDES    = [8, 16, 32]


# ======================
# DATA MODELS
# ======================

class AnomalyDetection(BaseModel):
    """A single detected object from the anomaly model."""
    bbox:       List[int]   # [x1, y1, x2, y2] in original image pixel coordinates
    score:      float
    class_id:   int
    class_name: str


class AnomalyResult(BaseModel):
    """Aggregated result from one anomaly classifier."""
    name:            str
    detected:        bool
    score:           float                    # highest single detection score (0.0 if none)
    class_name:      str                      # top detection class label, or "none"
    detection_count: int = 0                  # number of waste items detected (one per bbox)
    detections:      List[AnomalyDetection] = []   # all boxes above conf_thr


# ======================
# PROTOCOL
# ======================

@runtime_checkable
class AnomalyClassifierBase(Protocol):
    """Structural interface — no inheritance required."""
    name: str

    def classify(self, img_bgr: np.ndarray) -> AnomalyResult:
        """Run inference on a full BGR image; return AnomalyResult."""
        ...


# ======================
# REGISTRY
# ======================

class AnomalyRegistry:
    """
    Holds all registered anomaly classifiers and runs them sequentially.
    Errors in individual classifiers are caught so one failure never
    blocks the rest of the pipeline.
    """

    def __init__(self) -> None:
        self._classifiers: List[AnomalyClassifierBase] = []

    def register(self, classifier: AnomalyClassifierBase) -> None:
        self._classifiers.append(classifier)
        print(f"  [AnomalyRegistry] Registered: '{classifier.name}'")

    def run_all(self, img_bgr: np.ndarray) -> List[AnomalyResult]:
        results: List[AnomalyResult] = []
        for clf in self._classifiers:
            try:
                results.append(clf.classify(img_bgr))
            except Exception as exc:
                print(f"  [AnomalyRegistry] ERROR in '{clf.name}': {exc}")
                results.append(AnomalyResult(
                    name=clf.name, detected=False, score=0.0,
                    class_name="none", detections=[],
                ))
        return results

    def __len__(self) -> int:
        return len(self._classifiers)


# ======================
# YOLOX INFERENCE HELPERS
# ======================

def _letterbox(img: np.ndarray, target_size: tuple):
    """Letterbox resize to target_size (H, W), padding with 114."""
    h0, w0 = img.shape[:2]
    h1, w1 = target_size
    r = min(h1 / h0, w1 / w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((h1, w1, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, r


def _build_grids(input_size: tuple, strides: list):
    grids, expanded_strides = [], []
    h, w = input_size
    for stride in strides:
        hf, wf = h // stride, w // stride
        xv, yv = np.meshgrid(np.arange(wf), np.arange(hf))
        grid   = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, hf * wf, 1), stride, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def _decode(raw: np.ndarray, grids, strides_arr) -> np.ndarray:
    out = raw.copy()
    out[..., :2]  = (raw[..., :2] + grids) * strides_arr
    out[..., 2:4] = np.exp(raw[..., 2:4]) * strides_arr
    return out


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1   = np.maximum(x1[i], x1[order[1:]])
        iy1   = np.maximum(y1[i], y1[order[1:]])
        ix2   = np.minimum(x2[i], x2[order[1:]])
        iy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thr]
    return keep


def _postprocess(
    raw, grids, strides_arr, ratio, orig_shape,
    classes: List[str], conf_thr: float, nms_thr: float = 0.45,
) -> List[AnomalyDetection]:
    """Decode raw ONNX output → list of AnomalyDetection (sorted by score desc)."""
    decoded    = _decode(raw, grids, strides_arr)[0]
    obj_scores = decoded[:, 4]
    cls_scores = decoded[:, 5:5 + len(classes)]
    cls_ids    = cls_scores.argmax(axis=1)
    cls_conf   = cls_scores[np.arange(len(cls_ids)), cls_ids]
    scores     = obj_scores * cls_conf

    mask = scores > conf_thr
    if mask.sum() == 0:
        return []

    fboxes  = decoded[mask, :4]
    fscores = scores[mask]
    fcls    = cls_ids[mask]

    boxes_xyxy = _xywh2xyxy(fboxes)
    h0, w0     = orig_shape[:2]
    results: List[AnomalyDetection] = []

    for cls in np.unique(fcls):
        cm   = fcls == cls
        kept = _nms(boxes_xyxy[cm], fscores[cm], nms_thr)
        for k in kept:
            box   = boxes_xyxy[cm][k] / ratio
            score = float(fscores[cm][k])
            x1 = max(0,  int(box[0]))
            y1 = max(0,  int(box[1]))
            x2 = min(w0, int(box[2]))
            y2 = min(h0, int(box[3]))
            results.append(AnomalyDetection(
                bbox=[x1, y1, x2, y2],
                score=round(score, 6),
                class_id=int(cls),
                class_name=classes[int(cls)],
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ======================
# YOLOX ANOMALY CLASSIFIER
# ======================

class YoloxAnomalyClassifier:
    """
    Anomaly classifier backed by a YOLOX ONNX model.

    Preprocessing matches training exactly:
      letterbox → 1024×1024, HWC→CHW float32, no normalisation.

    Runs on the full BGR image passed to classify(); returns detections
    with absolute pixel-coordinate bounding boxes ready for drawing.

    Parameters
    ----------
    name      : Identifier surfaced in AnomalyResult.name.
    onnx_path : Path to the YOLOX ONNX checkpoint.
    classes   : Class-name list matching training order.
    conf_thr  : Detection confidence threshold (default 0.30).
    nms_thr   : IoU threshold for NMS (default 0.45).
    """

    def __init__(
        self,
        name:      str,
        onnx_path: str,
        classes:   List[str],
        conf_thr:  float = 0.30,
        nms_thr:   float = 0.45,
    ) -> None:
        self.name     = name
        self.classes  = classes
        self.conf_thr = conf_thr
        self.nms_thr  = nms_thr

        print(f"  Loading anomaly model from: {onnx_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active_provider   = self._sess.get_providers()[0]
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._grids, self._strides_arr = _build_grids(_INPUT_SIZE, _STRIDES)

        print(f"  Anomaly model ready  "
              f"(input {_INPUT_SIZE[1]}×{_INPUT_SIZE[0]}, "
              f"classes={classes}, provider={active_provider})")

    def classify(self, img_bgr: np.ndarray) -> AnomalyResult:
        """
        Run anomaly inference on a full BGR image.

        img_bgr   : H×W×3 numpy array in BGR colour order.
        Returns AnomalyResult with detected flag, top score, and all detections.
        """
        padded, ratio = _letterbox(img_bgr, _INPUT_SIZE)
        tensor = padded.transpose(2, 0, 1).astype(np.float32)[np.newaxis]  # [1,3,H,W]

        raw = self._sess.run(
            [self._output_name], {self._input_name: tensor}
        )[0]

        detections = _postprocess(
            raw, self._grids, self._strides_arr,
            ratio, img_bgr.shape,
            self.classes, self.conf_thr, self.nms_thr,
        )

        if not detections:
            return AnomalyResult(
                name=self.name, detected=False,
                score=0.0, class_name="none", detections=[],
            )

        return AnomalyResult(
            name=self.name,
            detected=True,
            score=detections[0].score,
            class_name=detections[0].class_name,
            detection_count=len(detections),
            detections=detections,
        )
