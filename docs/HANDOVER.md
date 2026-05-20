# HANDOVER.md — Inference Engine

**Date:** 2026-05-20
**Project:** DanielsHealth Washline — Fill Level Estimator API
**Repo root:** `E:\AMV\DanielsHealth_Washline\InferenceEngine\`

---

## What this system does

A FastAPI inference server (GPU PC, Linux, exposed via ngrok) that:
1. Accepts a camera image + wall geometry + fill-level reference lines from a C# WPF client.
2. Runs YOLOX to locate the waste region.
3. Runs a second YOLOX anomaly model on the full image to detect anomalous waste, counts bounding boxes, and saves a visualisation.
4. Segments the waste with SAM ViT-H.
5. Sweeps parallel scan lines across the bin and hits the waste surface to derive a fill percentage.
6. Returns fill level + anomaly summary. Saves mask, overlay, anomaly bbox image, and a JSON summary asynchronously.

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in config
cp .env.example .env

# Development (auto-reload)
python fill_estimator_api.py --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn fill_estimator_api:app --host 0.0.0.0 --port 8000
```

Swagger UI: `http://localhost:8000/docs`

---

## Environment configuration

All config lives in `.env` (copy from `.env.example`). Key variables:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `Model/SAM/BEST_model.pth` | SAM ViT-H checkpoint |
| `YOLO_PATH` | `Model/YOLO/yolox_m_daniels_adarsh.onnx` | Primary waste detector |
| `YOLO_CONF_THR` | `0.8` | Waste detector confidence threshold |
| `YOLO_NMS_THR` | `0.8` | Waste detector NMS IoU threshold |
| `MASK_THRESHOLD` | `0.30` | SAM sigmoid threshold |
| `AREA_THRESHOLD` | `30000` | Min foreground pixels — below this, fill = 0% |
| `NUM_LINES` | `100` | Number of parallel scan lines |
| `SAM_CACHE_SIZE` | `4` | LRU embedding cache size (0 = off) |
| `SAM_FP16` | `false` | Half-precision SAM encoder — validate accuracy before enabling |
| `ANOMALY_ENABLED` | `true` | Set `false` to skip all anomaly inference |
| `ANOMALY_MODEL_PATH` | `Model/Anomaly/yolox_m_washanomaly20MAY.onnx` | Anomaly ONNX checkpoint |
| `ANOMALY_MODEL_CLASSES` | `waste` | Class names (comma-separated, training order) |
| `ANOMALY_CONF_THR` | `0.30` | Anomaly detection confidence threshold |
| `OUTPUT_DIR` | `RESULT/api_results` | Root output folder |

---

## Model files

| Model | Path | Notes |
|---|---|---|
| SAM ViT-H | `Model/SAM/BEST_model.pth` | Fine-tuned; loaded via `segment_anything` library |
| YOLO waste detector | `Model/YOLO/yolox_m_daniels_adarsh.onnx` | Input size read from ONNX graph |
| YOLO anomaly detector | `Model/Anomaly/yolox_m_washanomaly20MAY.onnx` | Fixed 1024×1024 input, single class `waste` |

> Model files are gitignored (`.pth` and `.onnx`). Transfer them separately via secure file copy.

---

## API endpoint

### `POST /predict`

Multipart form fields:

| Field | Required | Description |
|---|---|---|
| `file` | Yes | JPEG/PNG, BGR-encoded (OpenCV convention) |
| `walls_json` | Yes | `{"bottom right": {"start":[x,y], "end":[x,y]}, "top right": {...}}` |
| `fill_lines_json` | Yes | `[{"fill": 95.0, "p1":[x1,y1], "p2":[x2,y2]}, ...]` |
| `barcode` | No | Container barcode — used in output filenames |
| `container_type` | No | Container type identifier — used in output filenames |

Response fields (all existing callers unaffected — new fields are additive):

```json
{
  "success": true,
  "fill_level": 78.44,
  "num_hits": 87,
  "largest_area": 290775,
  "forced_zero": false,
  "rep_point": [844.0, 942.0],
  "anomaly_results": [
    {
      "name": "yolox_washanomaly",
      "detected": true,
      "score": 0.873,
      "class_name": "waste",
      "detection_count": 2
    }
  ],
  "barcode": "BC111",
  "container_type": "64L",
  "output_path": "RESULT/api_results",
  "error": null
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "sam_loaded": true,
  "yolo_loaded": true,
  "anomaly_classifiers": 1
}
```

---

## Output files

Each request saves 4 files asynchronously (non-blocking):

```
RESULT/api_results/
├─ masks/     {ts}_{barcode}_{container}_{uuid}_mask.png
├─ overlays/  {ts}_{barcode}_{container}_{uuid}_overlay.jpg
├─ anomaly/   {ts}_{barcode}_{container}_{uuid}_anomaly.jpg   ← full image + bbox
└─ {ts}_{barcode}_{container}_{uuid}_summary.json             ← includes full detections list
```

---

## Module responsibilities

| File | Role |
|---|---|
| `fill_estimator_api.py` | FastAPI app — orchestrates pipeline, exposes `/predict` |
| `anomaly_classifier.py` | `AnomalyResult` model, `AnomalyClassifierBase` protocol, `AnomalyRegistry`, `YoloxAnomalyClassifier` |
| `sam_segmenter_large.py` | SAM ViT-H wrapper with LRU embedding cache and FP16 support |
| `yolo_detector.py` | YOLOX ONNX wrapper for waste detection |
| `wall_utils.py` | `generate_parallel_lines` + `march_mask` |
| `fill_interpolator.py` | `estimate_fill_from_hit` — hit x-coord → fill % |
| `visualizer.py` | Draws mask overlays, fill lines, hit points |
| `fill_line_loader.py` | CSV loader for fill-level reference lines (used by batch script only) |
| `run_mask_inference.py` | Standalone batch script — processes pre-generated mask images offline |

---

## Adding a second anomaly classifier

Only two lines in `fill_estimator_api.py` inside the `lifespan` startup function:

```python
clf2 = YoloxAnomalyClassifier(
    name="my_model",
    onnx_path="Model/Anomaly/my_model.onnx",
    classes=["classA", "classB"],
    conf_thr=0.5,
)
anomaly_registry.register(clf2)
```

No pipeline or schema changes needed. The new classifier's results appear as an additional entry in `anomaly_results`.

---

## Common issues

**Server starts but CUDA not used**
Check startup logs for `provider=CPUExecutionProvider`. Install `onnxruntime-gpu` and ensure CUDA libraries match the installed version.

**SAM model not found**
`MODEL_PATH` in `.env` must point to the actual `.pth` file. The default path is `Model/SAM/BEST_model.pth` relative to the project root.

**Anomaly model silently skipped**
If `ANOMALY_MODEL_PATH` does not exist at startup, the registry starts empty and `anomaly_results` is `[]` on every request. The startup log will print `Anomaly model not found at ...`.

**fill_level = 0.0 unexpectedly**
- `forced_zero=true` in the response: the segmented mask area fell below `AREA_THRESHOLD` pixels (default 30 000). The bin may be empty or YOLO/SAM underperformed.
- `forced_zero=false` but `fill_level=null`: no scan-line hit points were found. Check `num_hits` and `walls_json` geometry.

**Output files not appearing immediately**
File I/O is async. Files are written in a background thread — they may appear a few hundred milliseconds after the HTTP response is returned.

---

## Key architectural constraints

- Wall lines must use keys **`"bottom right"`** and **`"top right"`** exactly (the `walls.json` in the repo root uses different keys — it is legacy reference data).
- Anomaly model input is always letterboxed to **1024×1024**, no pixel normalisation (`/255` is NOT applied).
- SAM embedding cache is per-process, in-memory only. Restarting the server clears the cache.
- `barcode` and `container_type` characters outside `[A-Za-z0-9_-]` are silently replaced with `_` before use in filenames.

---

## Contacts and references

- `docs/PRD.md` — full product requirements and as-built documentation for all 4 phases
- `docs/CLAUDE.md` — concise developer reference (architecture, env vars, endpoint schema)
- `.env.example` — template for all environment variables
