#!/usr/bin/env python3
"""
FastAPI server for fill level estimation.
Accepts images and returns fill level predictions.
"""

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables from .env file if it exists
load_dotenv()

from anomaly_classifier import AnomalyRegistry, AnomalyResult, YoloxAnomalyClassifier
from fill_interpolator import estimate_fill_from_hit
from sam_segmenter_large import SamSegmenter
from visualizer import draw_fill_lines, draw_hit_points, overlay_mask
from wall_utils import generate_parallel_lines, march_mask
from yolo_detector import YoloDetector

# ======================
# CONFIGURATION
# ======================
MODEL_PATH    = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "Model", "SAM", "BEST_model.pth"))
YOLO_PATH     = os.getenv("YOLO_PATH",  os.path.join(os.path.dirname(__file__), "Model", "YOLO", "yolox_m_daniels_adarsh.onnx"))
YOLO_CONF_THR = float(os.getenv("YOLO_CONF_THR", "0.8"))
YOLO_NMS_THR  = float(os.getenv("YOLO_NMS_THR",  "0.8"))
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "RESULT", "api_results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters
MASK_THRESHOLD = float(os.getenv("MASK_THRESHOLD", "0.30"))
NUM_LINES      = int(os.getenv("NUM_LINES", "100"))
STEP_PIX       = int(os.getenv("STEP_PIX", "1"))
MIN_RUN        = int(os.getenv("MIN_RUN", "3"))
AREA_THRESHOLD = int(os.getenv("AREA_THRESHOLD", "30000"))

# SAM embedding cache + FP16 are read by sam_segmenter_large.py via SAM_CACHE_SIZE / SAM_FP16.

# Anomaly detection
ANOMALY_ENABLED       = os.getenv("ANOMALY_ENABLED", "true").lower() == "true"
ANOMALY_MODEL_PATH    = os.getenv("ANOMALY_MODEL_PATH", os.path.join(os.path.dirname(__file__), "Model", "Anomaly", "yolox_m_washanomaly20MAY.onnx"))
ANOMALY_MODEL_CLASSES = [c.strip() for c in os.getenv("ANOMALY_MODEL_CLASSES", "waste").split(",")]
ANOMALY_CONF_THR      = float(os.getenv("ANOMALY_CONF_THR", "0.5"))

# ======================
# GLOBAL STATE
# ======================
sam              = None   # SamSegmenter
yolo             = None   # YoloDetector (waste)
anomaly_registry = None   # AnomalyRegistry
executor         = None   # ThreadPoolExecutor for async file I/O


# ======================
# UTILITY FUNCTIONS
# ======================

def load_image_from_bytes(image_bytes: bytes):
    """Decode image bytes → (PIL Image RGB, numpy HxWx3 RGB array)."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    if img_np is None or img_np.size == 0:
        raise ValueError("Failed to load image from bytes")
    return image, img_np


def segment_mask(sam, image, bbox, threshold, image_hash: str = None):
    """Run SAM with bbox prompt; return (mask_clean uint8, area int, mask_bin uint8)."""
    mask         = sam.predict_mask(image, bbox=bbox, threshold=threshold, image_hash=image_hash)
    mask_clean   = (mask > 0).astype(np.uint8) * 255
    largest_area = int(mask_clean.sum() // 255)

    if largest_area == 0:
        empty = np.zeros_like(mask_clean)
        return empty, 0, empty.astype(np.uint8)

    mask_bin = (mask_clean > 0).astype(np.uint8)
    return mask_clean, largest_area, mask_bin


def collect_hit_points(mask_bin, parallel_lines, step_pix, min_run):
    """March along all scan lines right→left; return list of hit (x,y) points."""
    hit_points = []
    for pair in parallel_lines:
        if pair is None or len(pair) != 2:
            continue
        p_left, p_right = pair
        hit = march_mask(mask_bin, p_right, p_left, step=step_pix, min_run=min_run)
        if hit:
            hit_points.append(hit)
    return hit_points


def compute_fill_level(hit_points, ref_lines):
    """Return (median_hx, median_hy, fill_pct) or (None, None, None) if no hits."""
    if not hit_points:
        return None, None, None
    xs       = [p[0] for p in hit_points]
    ys       = [p[1] for p in hit_points]
    hx_rep   = float(np.median(xs))
    hy_rep   = float(np.median(ys))
    avg_fill = estimate_fill_from_hit(hx_rep, hy_rep, ref_lines)
    return hx_rep, hy_rep, avg_fill


def build_overlay_image(img_np, mask, hit_points, ref_lines, rep_pt, avg_fill):
    """Render mask + fill lines + hit points + fill-level label onto img_np (RGB)."""
    overlay = img_np.copy()
    overlay = overlay_mask(overlay, mask)
    overlay = draw_fill_lines(overlay, ref_lines, label_mode="vertical")

    hx_rep, hy_rep = rep_pt if rep_pt else (None, None)
    overlay = draw_hit_points(overlay, hit_points, rep_point=rep_pt)

    if rep_pt and avg_fill is not None:
        cv2.circle(overlay, (int(hx_rep), int(hy_rep)), 6, (255, 0, 0), -1)
        cv2.putText(overlay, "REP",
                    (int(hx_rep) + 5, int(hy_rep) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    txt = f"Fill Level: {avg_fill:.2f}%" if avg_fill is not None else "No valid hits"
    cv2.putText(overlay_bgr, txt, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def _sanitise_label(value: Optional[str], fallback: str = "unknown") -> str:
    """
    Return a filesystem-safe label for use in output filenames.
    - None / empty string → fallback ("unknown")
    - Characters outside [A-Za-z0-9_-] are replaced with underscore
    """
    if not value or not value.strip():
        return fallback
    return re.sub(r"[^A-Za-z0-9_\-]", "_", value.strip())


def _warmup():
    """Run one dummy forward pass through every model to compile CUDA kernels."""
    print("  Running warm-up inference...")
    dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB))
    if yolo is not None:
        yolo.detect(dummy_bgr)
    sam.predict_mask(dummy_pil, bbox=[0, 0, 320, 320], threshold=MASK_THRESHOLD)
    if anomaly_registry and len(anomaly_registry) > 0:
        anomaly_registry.run_all(dummy_bgr)
    print("  Warm-up complete — CUDA kernels ready")


def _save_outputs(mask_dir, overlay_dir, output_dir, safe_name,
                  mask_clean, overlay_final, summary: dict):
    """Write mask PNG, overlay JPG, and summary JSON to disk (runs in thread pool)."""
    try:
        mask_path    = os.path.join(mask_dir,    f"{safe_name}_mask.png")
        overlay_path = os.path.join(overlay_dir, f"{safe_name}_overlay.jpg")
        summary_path = os.path.join(output_dir,  f"{safe_name}_summary.json")

        Image.fromarray(mask_clean).save(mask_path)
        Image.fromarray(overlay_final).save(overlay_path, quality=95)
        summary["mask_path"]    = mask_path
        summary["overlay_path"] = overlay_path
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as exc:
        print(f"[ERROR] Failed to save outputs for {safe_name}: {exc}")


# ======================
# FASTAPI APP
# ======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, warm up. Shutdown: release thread pool."""
    global sam, yolo, anomaly_registry, executor

    print("=" * 60)
    print("Loading Fill Level Estimator API...")
    print("=" * 60)

    # [1/4] SAM
    print(f"\n[1/4] Loading SAM model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"SAM model not found at: {MODEL_PATH}")
    sam = SamSegmenter(MODEL_PATH)
    print("  SAM model loaded successfully")

    # [2/4] YOLO (waste detector)
    print(f"\n[2/4] Loading YOLO from: {YOLO_PATH}")
    if os.path.exists(YOLO_PATH):
        yolo = YoloDetector(YOLO_PATH, conf_thr=YOLO_CONF_THR, nms_thr=YOLO_NMS_THR)
        print("  YOLO loaded successfully")
    else:
        print(f"  YOLO not found at {YOLO_PATH} — detection will be skipped")

    # [3/4] Anomaly registry
    print("\n[3/4] Setting up anomaly classifier registry...")
    anomaly_registry = AnomalyRegistry()
    if not ANOMALY_ENABLED:
        print("  Anomaly detection disabled (ANOMALY_ENABLED=false)")
    elif not os.path.exists(ANOMALY_MODEL_PATH):
        print(f"  Anomaly model not found at {ANOMALY_MODEL_PATH} — skipping")
    else:
        clf = YoloxAnomalyClassifier(
            name="yolox_washanomaly",
            onnx_path=ANOMALY_MODEL_PATH,
            classes=ANOMALY_MODEL_CLASSES,
            conf_thr=ANOMALY_CONF_THR,
        )
        anomaly_registry.register(clf)
        print(f"  Anomaly registry ready ({len(anomaly_registry)} classifier(s))")

    # [4/4] Thread pool + warm-up
    print("\n[4/4] Initialising thread pool and warming up models...")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    _warmup()

    print("\n" + "=" * 60)
    print("API ready to accept requests!")
    print("=" * 60 + "\n")

    yield

    print("\nShutting down API...")
    if executor:
        executor.shutdown(wait=True)


app = FastAPI(
    title="Fill Level Estimator API",
    description="API for estimating fill levels in bin images using SAM segmentation",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "Fill Level Estimator API",
        "version": "1.0.0",
        "endpoints": {
            "/":        "This information",
            "/health":  "Health check",
            "/predict": "POST endpoint for fill level prediction",
            "/docs":    "Swagger API documentation",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status":              "healthy",
        "sam_loaded":          sam is not None,
        "yolo_loaded":         yolo is not None,
        "anomaly_classifiers": len(anomaly_registry) if anomaly_registry else 0,
    }


# ======================
# PYDANTIC MODELS
# ======================

class WallLine(BaseModel):
    start: List[int]   # [x, y]
    end:   List[int]   # [x, y]


class FillLine(BaseModel):
    fill: float
    p1:   List[int]    # [x, y]
    p2:   List[int]    # [x, y]


class PredictionResponse(BaseModel):
    success:         bool
    fill_level:      Optional[float]
    num_hits:        int
    largest_area:    int
    forced_zero:     bool
    rep_point:       Optional[List[float]]
    anomaly_results: List[AnomalyResult] = []
    barcode:         Optional[str] = None
    container_type:  Optional[str] = None
    output_path:     Optional[str] = None
    error:           Optional[str] = None


# ======================
# PREDICT ENDPOINT
# ======================

@app.post("/predict", response_model=PredictionResponse)
async def predict_fill_level(
    file:            UploadFile    = File(...,  description="Image file (JPEG/PNG) in BGR format"),
    walls_json:      Optional[str] = Form(None, description="JSON string of WALLS object"),
    fill_lines_json: Optional[str] = Form(None, description="JSON string of FILL_LINES array"),
    barcode:         Optional[str] = Form(None, description="Container barcode (optional, used in output filename)"),
    container_type:  Optional[str] = Form(None, description="Container type identifier (optional, used in output filename)"),
):
    """
    Predict fill level from an image.

    - **file**: JPEG/PNG image in BGR encoding (OpenCV convention).
    - **walls_json**: `{"bottom right": {"start": [x,y], "end": [x,y]}, "top right": {...}}`
    - **fill_lines_json**: `[{"fill": 95.0, "p1": [x1,y1], "p2": [x2,y2]}, ...]`
    - **barcode**: Optional container barcode — appended to saved output filenames.
    - **container_type**: Optional container type — appended to saved output filenames.

    Output images are saved asynchronously to OUTPUT_DIR on the server.
    """
    # Sanitise optional metadata fields up-front so they're safe for filenames
    safe_barcode    = _sanitise_label(barcode)
    safe_container  = _sanitise_label(container_type)

    try:
        # ---- Parse walls ----
        walls = None
        if walls_json:
            try:
                walls = {}
                for label, line_data in json.loads(walls_json).items():
                    start = (int(line_data["start"][0]), int(line_data["start"][1]))
                    end   = (int(line_data["end"][0]),   int(line_data["end"][1]))
                    if start[0] > end[0]:
                        start, end = end, start
                    walls[label.lower()] = {"start": start, "end": end}
                print(f"  Parsed {len(walls)} wall line(s)")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid WALLS JSON: {e}")
            except KeyError as e:
                raise ValueError(f"Missing field in WALLS: {e}")

        # ---- Parse fill lines ----
        ref_lines = None
        if fill_lines_json:
            try:
                ref_lines = []
                for line_data in json.loads(fill_lines_json):
                    col1, col2 = line_data["p1"]
                    row1, row2 = line_data["p2"]
                    ref_lines.append({
                        "fill": float(line_data["fill"]),
                        "p1":   (int(col1), int(row1)),
                        "p2":   (int(col2), int(row2)),
                    })
                ref_lines.sort(key=lambda r: r["fill"], reverse=True)
                print(f"  Parsed {len(ref_lines)} fill reference line(s)")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid FILL_LINES JSON: {e}")
            except KeyError as e:
                raise ValueError(f"Missing field in FILL_LINES: {e}")

        if not walls:
            raise ValueError("walls_json is required and must not be empty")
        if not ref_lines:
            raise ValueError("fill_lines_json is required and must not be empty")

        # ---- Load image ----
        image_bytes = await file.read()
        image_hash  = hashlib.md5(image_bytes).hexdigest()
        image, img_np = load_image_from_bytes(image_bytes)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ---- Filename stem ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_name = f"{timestamp}_{safe_barcode}_{safe_container}_{unique_id}"

        # ---- 1. YOLO detection ----
        bbox = None
        if yolo is not None:
            detections = yolo.detect(img_bgr)
            if detections:
                best = max(detections, key=lambda d: d["score"])
                bbox = best["bbox"]
                print(f"  YOLO bbox: {bbox}  score={best['score']:.2f}")

        if bbox is None:
            return PredictionResponse(
                success=True, fill_level=0.0, num_hits=0, largest_area=0,
                forced_zero=True, rep_point=None, anomaly_results=[],
                barcode=safe_barcode, container_type=safe_container,
                output_path=None, error="YOLO detected no content in image",
            )

        # ---- 2. Anomaly detection (on YOLO crop) ----
        anomaly_results = []
        if anomaly_registry and len(anomaly_registry) > 0:
            x1, y1, x2, y2 = bbox
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                anomaly_results = anomaly_registry.run_all(crop)
                for r in anomaly_results:
                    print(f"  Anomaly [{r.name}]: detected={r.detected}, "
                          f"score={r.score:.2f}, class={r.class_name}")

        # ---- 3. SAM segmentation ----
        mask_clean, largest_area, mask_bin = segment_mask(
            sam, image, bbox, MASK_THRESHOLD, image_hash=image_hash
        )

        # ---- 4. Parallel scan lines ----
        parallel_lines = list(generate_parallel_lines(walls, NUM_LINES))
        if not parallel_lines:
            return PredictionResponse(
                success=False, fill_level=None, num_hits=0,
                largest_area=int(largest_area), forced_zero=False,
                rep_point=None, anomaly_results=anomaly_results,
                barcode=safe_barcode, container_type=safe_container,
                output_path=None, error="Failed to generate parallel lines",
            )

        # ---- 5. Hit points ----
        hit_points = collect_hit_points(mask_bin, parallel_lines, STEP_PIX, MIN_RUN)

        # ---- 6. Fill level ----
        hx_rep = hy_rep = None
        if largest_area < AREA_THRESHOLD:
            avg_fill    = 0.0
            forced_zero = True
        else:
            hx_rep, hy_rep, avg_fill = compute_fill_level(hit_points, ref_lines)
            forced_zero = False

        # ---- 7. Overlay ----
        overlay_final = build_overlay_image(
            img_np, mask_clean, hit_points, ref_lines,
            rep_pt=(hx_rep, hy_rep) if hx_rep is not None else None,
            avg_fill=avg_fill,
        )

        # ---- 8. Async file I/O ----
        mask_dir    = os.path.join(OUTPUT_DIR, "masks")
        overlay_dir = os.path.join(OUTPUT_DIR, "overlays")
        os.makedirs(mask_dir,    exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        summary = {
            "image":           file.filename,
            "barcode":         safe_barcode,
            "container_type":  safe_container,
            "num_hits":        len(hit_points),
            "largest_area":    int(largest_area),
            "avg_fill":        float(avg_fill) if avg_fill is not None else None,
            "forced_zero":     forced_zero,
            "rep_point":       [hx_rep, hy_rep] if hx_rep is not None else None,
            "anomaly_results": [r.model_dump() for r in anomaly_results],
            "mask_path":       None,
            "overlay_path":    None,
        }

        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            executor,
            _save_outputs,
            mask_dir, overlay_dir, OUTPUT_DIR, safe_name,
            mask_clean, overlay_final, summary,
        )

        return PredictionResponse(
            success=True,
            fill_level=float(avg_fill) if avg_fill is not None else None,
            num_hits=len(hit_points),
            largest_area=int(largest_area),
            forced_zero=forced_zero,
            rep_point=[hx_rep, hy_rep] if hx_rep is not None else None,
            anomaly_results=anomaly_results,
            barcode=safe_barcode,
            container_type=safe_container,
            output_path=OUTPUT_DIR,
            error=None,
        )

    except Exception as e:
        return PredictionResponse(
            success=False, fill_level=None, num_hits=0, largest_area=0,
            forced_zero=False, rep_point=None, anomaly_results=[],
            barcode=safe_barcode, container_type=safe_container,
            output_path=None, error=str(e),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Fill Level Estimator API server")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run("fill_estimator_api:app", host=args.host, port=args.port, reload=args.reload)
