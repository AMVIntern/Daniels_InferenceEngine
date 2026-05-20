#!/usr/bin/env python3
"""
FastAPI server for fill level estimation.
Accepts images and returns fill level predictions.
"""

import os
import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Tuple, List, Dict
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from sam_segmenter_large import SamSegmenter
from fill_line_loader import load_fill_level_lines
from fill_interpolator import estimate_fill_from_hit
from wall_utils import parse_walls_csv, generate_parallel_lines, march_mask
from visualizer import draw_fill_lines, draw_hit_points, overlay_mask
from yolo_detector import YoloDetector

# ======================
# CONFIGURATION
# ======================
# Can be overridden via environment variables
MODEL_PATH    = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "Model", "SAM_LARGE_CHECKPOINTS", "BEST_model.pth"))
YOLO_PATH     = os.getenv("YOLO_PATH",  os.path.join(os.path.dirname(__file__), "Model", "YOLO", "yolox_m_daniels_adarsh.onnx"))
YOLO_CONF_THR = float(os.getenv("YOLO_CONF_THR", "0.8"))
YOLO_NMS_THR  = float(os.getenv("YOLO_NMS_THR",  "0.8"))
# WALLS_CSV = os.getenv("WALLS_CSV", os.path.join(os.path.dirname(__file__), "images", "Biggest", "64_wall.csv"))
# FILL_LINES_CSV = os.getenv("FILL_LINES_CSV", os.path.join(os.path.dirname(__file__), "Refrence_fill_levels", "fill_level_lines_bin_64_series.csv"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "RESULT", "api_results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters
MASK_THRESHOLD = float(os.getenv("MASK_THRESHOLD", "0.30"))
NUM_LINES = int(os.getenv("NUM_LINES", "100"))
STEP_PIX = int(os.getenv("STEP_PIX", "1"))
MIN_RUN = int(os.getenv("MIN_RUN", "3"))
AREA_THRESHOLD = int(os.getenv("AREA_THRESHOLD", "30000"))

# ======================
# GLOBAL STATE (loaded at startup)
# ======================
sam  = None
yolo = None
walls = None
ref_lines = None


# ======================
# UTILITY FUNCTIONS
# ======================
def load_image_from_bytes(image_bytes: bytes, is_bgr: bool = True):
    """Load image from bytes and convert to RGB numpy array."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    if img_np is None or img_np.size == 0:
        raise ValueError("Failed to load image from bytes")
    return image, img_np

def segment_mask(sam, image, bbox, threshold):
    """Run SAM with bbox prompt and return clean mask + area + binary mask."""
    mask = sam.predict_mask(image, bbox=bbox, threshold=threshold)
    mask_clean = (mask > 0).astype(np.uint8) * 255

    largest_area = int(mask_clean.sum() // 255)
    if largest_area == 0:
        empty = np.zeros_like(mask_clean)
        return empty, 0, empty.astype(np.uint8)

    mask_bin = (mask_clean > 0).astype(np.uint8)
    return mask_clean, largest_area, mask_bin


def collect_hit_points(mask_bin, parallel_lines, step_pix, min_run):
    """March along all lines and collect hit intersection points."""
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
    """Return representative hit point + estimated fill percentage."""
    if not hit_points:
        return None, None, None

    xs = [p[0] for p in hit_points]
    ys = [p[1] for p in hit_points]

    hx_rep = float(np.median(xs))
    hy_rep = float(np.median(ys))

    avg_fill = estimate_fill_from_hit(hx_rep, hy_rep, ref_lines)

    return hx_rep, hy_rep, avg_fill


def build_overlay_image(img_np, mask, hit_points, ref_lines, rep_pt, avg_fill):
    """Build overlay with mask, fill lines, hit points and label."""
    overlay = img_np.copy()

    overlay = overlay_mask(overlay, mask)
    overlay = draw_fill_lines(overlay, ref_lines, label_mode="vertical")

    hx_rep, hy_rep = rep_pt if rep_pt else (None, None)

    overlay = draw_hit_points(overlay, hit_points, rep_point=rep_pt)

    # Draw representative point
    if rep_pt and avg_fill is not None:
        cv2.circle(overlay, (int(hx_rep), int(hy_rep)), 6, (255,0,0), -1)
        cv2.putText(
            overlay, "REP",
            (int(hx_rep)+5, int(hy_rep)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,0,0),
            2
        )

    # Fill label
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    txt = f"Fill Level: {avg_fill:.2f}%" if avg_fill is not None else "No valid hits"

    cv2.putText(
        overlay_bgr, txt,
        (20,40), cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (0,255,0), 3
    )

    overlay_final = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_final

# ======================
# FASTAPI APP
# ======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global sam, yolo, walls, ref_lines
    
    print("=" * 60)
    print("Loading Fill Level Estimator API...")
    print("=" * 60)
    
    # Load SAM model
    print(f"\n[1/3] Loading SAM model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"SAM model not found at: {MODEL_PATH}")
    sam = SamSegmenter(MODEL_PATH)
    print(" SAM model loaded successfully")

    # Load YOLO (optional — falls back to wall bbox if not found)
    print("yolo path",YOLO_PATH)
    if os.path.exists(YOLO_PATH):
        yolo = YoloDetector(YOLO_PATH, conf_thr=YOLO_CONF_THR, nms_thr=YOLO_NMS_THR)
        print(" YOLO loaded successfully")
    else:
        print(f" YOLO not found at {YOLO_PATH} — will use wall-derived bbox")
    
    print("\n" + "=" * 60)
    print("API ready to accept requests!")
    print("=" * 60 + "\n")
    
    yield
    
    # Shutdown (if needed, cleanup code would go here)
    print("\nShutting down API...")


app = FastAPI(
    title="Fill Level Estimator API",
    description="API for estimating fill levels in bin images using SAM segmentation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fill Level Estimator API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This information",
            "/health": "Health check",
            "/predict": "POST endpoint for fill level prediction",
            "/docs": "Swagger API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": sam is not None,
        "walls_loaded": walls is not None and len(walls) > 0,
        "ref_lines_loaded": ref_lines is not None and len(ref_lines) > 0
    }


class WallLine(BaseModel):
    """Model for a single wall line."""
    start: List[int]  # [x, y]
    end: List[int]    # [x, y]


class FillLine(BaseModel):
    """Model for a single fill reference line."""
    fill: float
    p1: List[int]     # [x, y]
    p2: List[int]     # [x, y]


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    fill_level: Optional[float]
    num_hits: int
    largest_area: int
    forced_zero: bool
    rep_point: Optional[list[float]]
    output_path: Optional[str] = None
    error: Optional[str] = None


@app.post("/predict", response_model=PredictionResponse)
async def predict_fill_level(
    file: UploadFile = File(..., description="Image file (JPEG/PNG) in BGR format"),
    walls_json: Optional[str] = Form(None, description="JSON string of WALLS object"),
    fill_lines_json: Optional[str] = Form(None, description="JSON string of FILL_LINES object")):
    """
    Predict fill level from an image.
    
    Parameters:
    - file: Image file (JPEG/PNG format). The image should be in BGR format (OpenCV format).
            The API will automatically convert it to RGB for processing.
    - walls_json: JSON string containing WALLS object. Format: {"label": {"start": [x, y], "end": [x, y]}, ...}
    - fill_lines_json: JSON string containing FILL_LINES array. Format: [{"fill": float, "p1": [x, y], "p2": [x, y]}, ...]    
    Returns:
    - JSON response with fill level and metadata
    - Images are saved to the results folder (configured via OUTPUT_DIR)
    """
    try:
        # Parse WALLS and FILL_LINES from JSON if provided
        walls = None
        ref_lines = None
        
        if walls_json:
            print("\n[2/3] Parsing WALLS from JSON")
            try:
                walls_dict = json.loads(walls_json)
                print("walls json", walls_dict )
                # Convert to the format expected by wall_utils
                walls = {}
                for label, line_data in walls_dict.items():
                    start = (int(line_data["start"][0]), int(line_data["start"][1]))
                    end   = (int(line_data["end"][0]),   int(line_data["end"][1]))
                    # Normalize so start is always the left (smaller x) endpoint
                    if start[0] > end[0]:
                        start, end = end, start
                    walls[label.lower()] = {
                        "start": start,
                        "end":   end
                    }
                print(f" Parsed {len(walls)} wall line(s) from JSON")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid WALLS JSON format: {str(e)}")
            except KeyError as e:
                raise ValueError(f"Missing required field in WALLS: {str(e)}")
        
        if fill_lines_json:
            print("\n[3/3] Parsing FILL_LINES from JSON")
            try:
                fill_lines_list = json.loads(fill_lines_json)
                # Convert to the format expected by fill_interpolator
                ref_lines = []
                for line_data in fill_lines_list:
                    col1, col2 = line_data["p1"]
                    row1, row2 = line_data["p2"]
                    ref_lines.append({
                        "fill": float(line_data["fill"]),
                        "p1": (int(col1), int(row1)),
                        "p2": (int(col2), int(row2))
                    })
                # Sort by fill percentage (descending) as done in load_fill_level_lines
                ref_lines.sort(key=lambda r: r["fill"], reverse=True)
                print(f" Parsed {len(ref_lines)} fill reference lines from JSON")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid FILL_LINES JSON format: {str(e)}")
            except KeyError as e:
                raise ValueError(f"Missing required field in FILL_LINES: {str(e)}")
        print("DEBUG fill lines sample:", json.dumps([
        {"fill": r["fill"], "p1": list(r["p1"]), "p2": list(r["p2"])}
        for r in ref_lines[:3]
    ], indent=2))
        # Validate that we have both walls and ref_lines
        if walls is None or not walls:
            raise ValueError("WALLS must be provided either as JSON or via CSV")
        if ref_lines is None or not ref_lines:
            raise ValueError("FILL_LINES must be provided either as JSON or via CSV")
        print("DEBUG walls received:", json.dumps(walls, indent=2))

        # Read image bytes (expects BGR-encoded image)
        image_bytes = await file.read()
        
        # Load image and convert from BGR to RGB for processing
        image, img_np = load_image_from_bytes(image_bytes, is_bgr=True)
        
        # Generate unique filename for this prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = str(uuid.uuid4())[:8]
        safe_name = f"{timestamp}_{unique_id}"
        
        # ---- 1. Detect bbox (YOLO first, wall fallback) ----
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        bbox = None

        if yolo is not None:
            detections = yolo.detect(img_bgr)
            if detections:
                best = max(detections, key=lambda d: d["score"])
                bbox = best["bbox"]
                print(f" YOLO bbox: {bbox}  score={best['score']:.2f}")

        if bbox is None:
            
            return PredictionResponse(
                success=True,
                fill_level=0.0,
                num_hits=0,
                largest_area=0,
                forced_zero=True,
                rep_point=None,
                output_path=None,
                error="YOLO detected no content in image"
            )

        # ---- 2. SAM segmentation ----
        mask_clean, largest_area, mask_bin = segment_mask(
            sam, image, bbox, MASK_THRESHOLD
        )
        
        # ---- 3. Generate parallel lines ----
        parallel_lines = list(generate_parallel_lines(walls, NUM_LINES))
        if not parallel_lines:
            return PredictionResponse(
                success=False,
                fill_level=None,
                num_hits=0,
                largest_area=int(largest_area),
                forced_zero=False,
                rep_point=None,
                output_path=None,
                error="Failed to generate parallel lines"
            )
        
        # ---- 4. Collect hits ----
        hit_points = collect_hit_points(
            mask_bin, parallel_lines, STEP_PIX, MIN_RUN
        )
        
        # ---- 5. Determine fill level (with area threshold check) ----
        hx_rep = None
        hy_rep = None
        
        if largest_area < AREA_THRESHOLD:
            avg_fill = 0.0
            forced_zero = True
        else:
            hx_rep, hy_rep, avg_fill = compute_fill_level(hit_points, ref_lines)
            forced_zero = False
            if avg_fill is None:
                avg_fill = None
        
        # ---- 6. Build Overlay ----
        overlay_final = build_overlay_image(
            img_np,
            mask_clean,
            hit_points,
            ref_lines,
            rep_pt=(hx_rep, hy_rep) if hx_rep is not None else None,
            avg_fill=avg_fill
        )
        
        # ---- 7. Save Outputs ----
        # Create subdirectories for masks and overlays
        mask_dir = os.path.join(OUTPUT_DIR, "masks")
        overlay_dir = os.path.join(OUTPUT_DIR, "overlays")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        
        # Save mask
        mask_path = os.path.join(mask_dir, f"{safe_name}_mask.png")
        Image.fromarray(mask_clean).save(mask_path)
        
        # Save overlay
        overlay_path = os.path.join(overlay_dir, f"{safe_name}_overlay.jpg")
        Image.fromarray(overlay_final).save(overlay_path, quality=95)
        
        # Save JSON summary
        summary_path = os.path.join(OUTPUT_DIR, f"{safe_name}_summary.json")
        summary = {
            "image": file.filename,
            "num_hits": len(hit_points),
            "largest_area": int(largest_area),
            "avg_fill": float(avg_fill) if avg_fill is not None else None,
            "forced_zero": forced_zero,
            "rep_point": [hx_rep, hy_rep] if hx_rep is not None else None,
            "mask_path": mask_path,
            "overlay_path": overlay_path
        }
        with open(summary_path, "w") as jf:
            json.dump(summary, jf, indent=2)
        
        # Prepare response
        return PredictionResponse(
            success=True,
            fill_level=float(avg_fill) if avg_fill is not None else None,
            num_hits=len(hit_points),
            largest_area=int(largest_area),
            forced_zero=forced_zero,
            rep_point=[hx_rep, hy_rep] if hx_rep is not None else None,
            output_path=OUTPUT_DIR,
            error=None
        )
        
    except Exception as e:
        return PredictionResponse(
            success=False,
            fill_level=None,
            num_hits=0,
            largest_area=0,
            forced_zero=False,
            rep_point=None,
            output_path=None,
            error=str(e)
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Fill Level Estimator API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "fill_estimator_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

 