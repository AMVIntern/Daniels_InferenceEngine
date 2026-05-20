# anomaly_classifier.py
"""
Extensible anomaly detection framework.

Architecture
------------
AnomalyResult           — Pydantic model returned by every classifier.
AnomalyClassifierBase   — Protocol (structural interface) every classifier must satisfy.
AnomalyRegistry         — Holds all registered classifiers; runs them all on a crop.
YoloxAnomalyClassifier  — Concrete implementation backed by a YOLOX ONNX model.

Adding a new anomaly classifier
--------------------------------
1. Implement AnomalyClassifierBase (or instantiate YoloxAnomalyClassifier with a new ONNX path).
2. Call anomaly_registry.register(your_classifier) in fill_estimator_api.py lifespan.
No pipeline or schema code needs to change.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel


# ======================
# DATA MODEL
# ======================

class AnomalyResult(BaseModel):
    """Result produced by a single anomaly classifier."""
    name:       str    # classifier identifier
    detected:   bool   # True if any detection exceeds the confidence threshold
    score:      float  # highest detection confidence (0.0 when nothing detected)
    class_name: str    # detected class label, or "none"


# ======================
# PROTOCOL (interface)
# ======================

@runtime_checkable
class AnomalyClassifierBase(Protocol):
    """
    Structural interface every anomaly classifier must satisfy.
    No inheritance required — duck typing is enough.
    """
    name: str

    def classify(self, img_bgr: np.ndarray) -> AnomalyResult:
        """Run inference on a BGR-encoded crop; return an AnomalyResult."""
        ...


# ======================
# REGISTRY
# ======================

class AnomalyRegistry:
    """
    Holds all registered anomaly classifiers and runs them sequentially
    on a given image crop.

    Errors in individual classifiers are caught and logged so that one
    failing model never blocks the rest of the pipeline.
    """

    def __init__(self) -> None:
        self._classifiers: List[AnomalyClassifierBase] = []

    def register(self, classifier: AnomalyClassifierBase) -> None:
        """Add a classifier to the registry."""
        self._classifiers.append(classifier)
        print(f"  [AnomalyRegistry] Registered classifier: '{classifier.name}'")

    def run_all(self, img_bgr: np.ndarray) -> List[AnomalyResult]:
        """Run every registered classifier on img_bgr; return one result each."""
        results: List[AnomalyResult] = []
        for clf in self._classifiers:
            try:
                results.append(clf.classify(img_bgr))
            except Exception as exc:
                print(f"  [AnomalyRegistry] ERROR in '{clf.name}': {exc}")
                results.append(AnomalyResult(
                    name=clf.name, detected=False, score=0.0, class_name="none"
                ))
        return results

    def __len__(self) -> int:
        return len(self._classifiers)


# ======================
# YOLOX IMPLEMENTATION
# ======================

class YoloxAnomalyClassifier:
    """
    Anomaly classifier backed by a YOLOX ONNX model.

    Reuses YoloDetector (same letterbox / decode / NMS path) so the
    anomaly model shares the same inference code as the waste detector.
    Only the ONNX weights, class list, and confidence threshold differ.

    Parameters
    ----------
    name      : Identifier surfaced in AnomalyResult.name.
    onnx_path : Path to the YOLOX ONNX checkpoint.
    classes   : List of class-name strings the model was trained on.
    conf_thr  : Detections below this confidence are ignored.
    nms_thr   : IoU threshold for NMS (default 0.5).
    """

    def __init__(
        self,
        name:      str,
        onnx_path: str,
        classes:   List[str],
        conf_thr:  float = 0.5,
        nms_thr:   float = 0.5,
    ) -> None:
        # Import here to avoid a circular import at module level.
        from yolo_detector import YoloDetector

        self.name      = name
        self._detector = YoloDetector(
            onnx_path,
            conf_thr=conf_thr,
            nms_thr=nms_thr,
            classes=classes,
        )

    def classify(self, img_bgr: np.ndarray) -> AnomalyResult:
        """
        Run YOLOX detection on img_bgr (a BGR-encoded crop).
        Returns detected=True if any box exceeds conf_thr.
        """
        detections = self._detector.detect(img_bgr)
        if not detections:
            return AnomalyResult(
                name=self.name, detected=False, score=0.0, class_name="none"
            )
        best = max(detections, key=lambda d: d["score"])
        return AnomalyResult(
            name=self.name,
            detected=True,
            score=float(best["score"]),
            class_name=best["class"],
        )
