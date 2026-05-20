# yolo_detector.py
# YOLO detection helper — ported from colleague's training script.

import numpy as np
import cv2
import onnxruntime as ort

YOLO_STRIDES  = [8, 16, 32]
YOLO_CLASSES  = ["waste"]


# -----------------------------------------------------------------------
# Helper functions (match training script exactly)
# -----------------------------------------------------------------------

def letterbox(img, target_h, target_w):
    h0, w0 = img.shape[:2]
    r = min(target_h / h0, target_w / w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, r


def build_grids(input_h, input_w, strides):
    grids, stride_arr = [], []
    for s in strides:
        hf, wf = input_h // s, input_w // s
        xv, yv = np.meshgrid(np.arange(wf), np.arange(hf))
        grid   = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        stride_arr.append(np.full((1, hf * wf, 1), s, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(stride_arr, axis=1)


def decode_yolo(raw, grids, strides_arr):
    out = raw.copy()
    out[..., :2]  = (raw[..., :2] + grids) * strides_arr
    out[..., 2:4] = np.exp(raw[..., 2:4]) * strides_arr
    return out


def xywh2xyxy(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes_xyxy, scores, iou_thr):
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thr]
    return keep


def yolo_detect(img_bgr, sess, input_name, output_name,
                in_h, in_w, grids, strides_arr,
                conf_thr, nms_thr, classes):
    padded, ratio = letterbox(img_bgr, in_h, in_w)
    tensor = np.ascontiguousarray(
        padded.transpose(2, 0, 1), dtype=np.float32
    )[np.newaxis]
    raw  = sess.run([output_name], {input_name: tensor})[0]
    dec  = decode_yolo(raw, grids, strides_arr)[0]

    obj    = dec[:, 4]
    cls_s  = dec[:, 5:5 + len(classes)]
    cls_i  = cls_s.argmax(axis=1)
    scores = obj * cls_s[np.arange(len(cls_i)), cls_i]

    keep_mask = scores > conf_thr
    if keep_mask.sum() == 0:
        return []

    fboxes  = xywh2xyxy(dec[keep_mask, :4])
    fscores = scores[keep_mask]
    fclsids = cls_i[keep_mask]

    h0, w0 = img_bgr.shape[:2]
    kept   = nms(fboxes, fscores, nms_thr)

    result = []
    for k in kept:
        box = fboxes[k] / ratio
        result.append({
            "bbox":  [max(0, int(box[0])), max(0, int(box[1])),
                      min(w0, int(box[2])), min(h0, int(box[3]))],
            "score": float(fscores[k]),
            "class": classes[int(fclsids[k])],
        })
    return result


# -----------------------------------------------------------------------
# YoloDetector class
# -----------------------------------------------------------------------

class YoloDetector:
    """Wraps YOLOX ONNX model for waste detection."""

    def __init__(self, onnx_path, conf_thr=0.8, nms_thr=0.8):
        print(f"Loading YOLO from: {onnx_path}")
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name  = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.in_h, self.in_w = shape[2], shape[3]
        self.grids, self.strides_arr = build_grids(self.in_h, self.in_w, YOLO_STRIDES)
        self.conf_thr = conf_thr
        self.nms_thr  = nms_thr
        print(f"YOLO ready  (input {self.in_w}x{self.in_h})")

    def detect(self, img_bgr):
        """
        Run detection on a BGR image.
        Returns list of {"bbox": [x1,y1,x2,y2], "score": float, "class": str}
        """
        return yolo_detect(
            img_bgr,
            self.sess, self.input_name, self.output_name,
            self.in_h, self.in_w, self.grids, self.strides_arr,
            self.conf_thr, self.nms_thr, YOLO_CLASSES
        )
