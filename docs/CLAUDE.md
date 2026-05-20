# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this system does

A computer-vision inference engine that estimates how full laundry bins are from camera images. The pipeline:

1. **YOLOX** (ONNX, `Model/YOLO/`) detects the waste region bounding box.
2. **YOLOX Anomaly** (ONNX, `Model/Anomaly/`) runs on the cropped region to detect anomalous waste — single class `waste`, input size 1024.
3. **SAM ViT-H** (fine-tuned, `Model/SAM/`) segments the waste using the YOLO bbox + center-point as a two-pass prompt.
4. **Parallel scan lines** sweep across the bin geometry (derived from calibrated wall lines).
5. A **march-mask** algorithm finds where each scan line first hits the waste surface (right → left).
6. The median hit-point x-coordinate is interpolated against calibrated reference lines to produce a **fill percentage**.

## Running the API server

```bash
pip install -r requirements.txt

# Development
python fill_estimator_api.py --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn fill_estimator_api:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Running batch mask inference (offline, no SAM)

```bash
python run_mask_inference.py \
  --input-dir ./input_images \
  --walls-csv images/Biggest/64_wall.csv \
  --fill-lines-csv Refrence_fill_levels/fill_level_lines_bin_64_series_new_27MAR26.csv \
  --output-dir RESULT/mask_inference_results \
  --save-debug
```

Takes pre-generated mask images or green-tinted overlay images; skips YOLO and SAM entirely.

## Environment configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `Model/SAM/BEST_model.pth` | SAM ViT-H checkpoint |
| `YOLO_PATH` | `Model/YOLO/yolox_m_daniels_adarsh.onnx` | Waste detector ONNX |
| `YOLO_CONF_THR` | `0.8` | Waste detector confidence threshold |
| `MASK_THRESHOLD` | `0.30` | SAM sigmoid threshold |
| `AREA_THRESHOLD` | `30000` | Min foreground pixels — below this, fill = 0% |
| `NUM_LINES` | `100` | Number of parallel scan lines |
| `OUTPUT_DIR` | `RESULT/api_results` | Where masks/overlays/JSON summaries are saved |
| `SAM_CACHE_SIZE` | `4` | LRU embedding cache entries (0 = disabled) |
| `SAM_FP16` | `false` | Half-precision SAM encoder on CUDA |
| `ANOMALY_ENABLED` | `true` | Kill-switch for anomaly inference |
| `ANOMALY_MODEL_PATH` | `Model/Anomaly/yolox_m_washanomaly20MAY.onnx` | Anomaly ONNX |
| `ANOMALY_MODEL_CLASSES` | `waste` | Comma-separated class names for anomaly model |
| `ANOMALY_CONF_THR` | `0.5` | Anomaly detection confidence threshold |

## Architecture: module responsibilities

| File | Role |
|---|---|
| `fill_estimator_api.py` | FastAPI app — orchestrates the full pipeline, exposes `/predict` |
| `sam_segmenter_large.py` | SAM ViT-H wrapper: `segment_anything` lib, bbox+center prompt, two-pass refinement, LRU embedding cache |
| `yolo_detector.py` | YOLOX ONNX inference — letterbox, decode, NMS. Accepts `classes` param so both waste and anomaly models reuse the same path |
| `anomaly_classifier.py` | `AnomalyResult`, `AnomalyClassifierBase` protocol, `AnomalyRegistry`, `YoloxAnomalyClassifier` |
| `wall_utils.py` | Generates parallel scan lines from wall geometry; implements `march_mask` |
| `fill_line_loader.py` | Loads reference fill-level CSVs — used by batch script |
| `fill_interpolator.py` | Maps a hit-point x-coordinate to fill % via linear interpolation |
| `visualizer.py` | Draws mask overlays, fill-level reference lines, hit points |
| `run_mask_inference.py` | Standalone batch script — no YOLO/SAM, processes pre-generated masks |

## Anomaly classifier extensibility

Adding a second anomaly model requires only two lines in the `lifespan` startup of `fill_estimator_api.py`:

```python
clf2 = YoloxAnomalyClassifier(name="my_new_model", onnx_path="...", classes=["classA"], conf_thr=0.5)
anomaly_registry.register(clf2)
```

No pipeline or schema code changes needed. `AnomalyRegistry.run_all()` calls every registered classifier and returns one `AnomalyResult` per classifier in `anomaly_results`.

## Wall line key convention

`wall_utils.generate_parallel_lines` expects wall dict keys **`"bottom right"`** and **`"top right"`**. The client must send exactly these keys in `walls_json`.

The `walls.json` file in the repo root uses `"bottom left"` / `"top left"` — legacy reference data, not consumed by the API.

## `/predict` endpoint

`POST /predict` — `multipart/form-data`:

| Field | Required | Format |
|---|---|---|
| `file` | Yes | JPEG/PNG, BGR-encoded (OpenCV convention) |
| `walls_json` | Yes | `{"bottom right": {"start": [x,y], "end": [x,y]}, "top right": {...}}` |
| `fill_lines_json` | Yes | `[{"fill": 95.0, "p1": [x1,y1], "p2": [x2,y2]}, ...]` |
| `barcode` | No | Container barcode string — used in output filenames, echoed in response |
| `container_type` | No | Container type identifier — used in output filenames, echoed in response |

Both optional fields default to `"unknown"` when omitted. Characters outside `[A-Za-z0-9_-]` are replaced with `_` before use in filenames.

Response includes `fill_level`, `anomaly_results`, `barcode`, and `container_type`. All new fields are additive — existing callers are unaffected.

## Output structure

Each `/predict` call saves asynchronously to `OUTPUT_DIR`:
- `masks/{timestamp}_{barcode}_{container_type}_{uuid}_mask.png`
- `overlays/{timestamp}_{barcode}_{container_type}_{uuid}_overlay.jpg`
- `{timestamp}_{barcode}_{container_type}_{uuid}_summary.json` — includes `barcode`, `container_type`, `anomaly_results`

Example: `20260520_101108_BC111_64L_a3f2b1c0_overlay.jpg`
