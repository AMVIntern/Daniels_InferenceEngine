# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this system does

This is a computer-vision inference engine that estimates how full laundry bins are from camera images. The pipeline:

1. **YOLOX** (ONNX) detects the waste region bounding box in the image.
2. **SAM ViT-H** (fine-tuned, `segment_anything` library) segments the waste using the YOLO bbox + center-point as a two-pass prompt.
3. **Parallel scan lines** are generated across the bin geometry (derived from calibrated wall lines).
4. A **march-mask** algorithm sweeps each scan line from right to left to find the waste surface.
5. The median hit point is mapped to a **fill percentage** by interpolating between calibrated reference lines.

## Running the API server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (default: 0.0.0.0:8000)
python fill_estimator_api.py

# With options
python fill_estimator_api.py --host 0.0.0.0 --port 8000 --reload

# Production (via uvicorn directly)
uvicorn fill_estimator_api:app --host 0.0.0.0 --port 8000
```

API docs are at `http://localhost:8000/docs` once running.

## Running batch mask inference (no SAM needed)

```bash
python run_mask_inference.py \
  --input-dir ./input_images \
  --walls-csv images/Biggest/64_wall.csv \
  --fill-lines-csv Refrence_fill_levels/fill_level_lines_bin_64_series_new_27MAR26.csv \
  --output-dir RESULT/mask_inference_results \
  --save-debug
```

This script skips SAM — it reads pre-generated mask images (or overlay images with a green tint) and runs the fill estimation geometry only.

## Environment configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `Model/SAM_LARGE_CHECKPOINTS/BEST_model.pth` | SAM ViT-H checkpoint |
| `YOLO_PATH` | `Model/YOLO/yolox_m_daniels_adarsh.onnx` | YOLOX ONNX model |
| `YOLO_CONF_THR` | `0.8` | YOLO confidence threshold |
| `MASK_THRESHOLD` | `0.30` | SAM sigmoid threshold |
| `AREA_THRESHOLD` | `30000` | Min foreground pixels — below this, fill = 0% |
| `NUM_LINES` | `100` | Number of parallel scan lines |
| `OUTPUT_DIR` | `RESULT/api_results` | Where masks/overlays/JSON summaries are saved |

## Architecture: module responsibilities

| File | Role |
|---|---|
| `fill_estimator_api.py` | FastAPI app — orchestrates the full pipeline, exposes `/predict` |
| `sam_segmenter_large.py` | **Active** SAM wrapper: ViT-H via `segment_anything`, bbox+center prompt, two-pass refinement |
| `sam_segmenter.py` | **Older** SAM wrapper: ViT-B via HuggingFace transformers, grid-point prompting — not used by the API |
| `yolo_detector.py` | YOLOX ONNX inference with manual NMS/decode (single class: `waste`) |
| `wall_utils.py` | Parses wall CSVs, generates parallel scan lines, implements `march_mask` |
| `fill_line_loader.py` | Loads reference fill-level CSV into `[{fill, p1, p2}]` dicts |
| `fill_interpolator.py` | Maps a hit point's x-coordinate to a fill % by linear interpolation between reference lines |
| `visualizer.py` | Draws mask overlays, fill-level lines, hit points onto images |
| `run_mask_inference.py` | Standalone batch script — no SAM, takes mask/overlay images as input |

## Wall line key convention

`wall_utils.generate_parallel_lines` expects wall dict keys **`"bottom right"`** and **`"top right"`**. The API `/predict` endpoint accepts these as `walls_json` (JSON string from the client). The keys in the client request must match these exactly.

The `walls.json` file in the repo root uses `"bottom left"` / `"top left"` — this is a separate/legacy config not consumed by the API directly.

The batch script (`run_mask_inference.py`) reads CSV labels `Bin_Bottom_Right` / `Bin_Top_Right` and maps them to its own internal keys.

## Reference data files

- `Refrence_fill_levels/`: CSVs of calibrated fill-level lines per bin series (14, 22, 32, 64). The `_new_27MAR26.csv` variants are the current calibrations.
- `images/Biggest/64_wall.csv`, `images/Nextbiggest/32_wall.csv`, etc.: Wall line annotations per bin size.
- `fill_lines.json` / `walls.json`: JSON equivalents of the above, used for passing to the API or testing.
- `Model/Anomaly/yolox_m_washanomaly20MAY.onnx`: Anomaly detection model — present but not wired into the current pipeline.

## `/predict` endpoint request format

`POST /predict` uses `multipart/form-data`:
- `file`: image (JPEG/PNG), expected in BGR encoding (OpenCV convention)
- `walls_json`: JSON string — `{"bottom right": {"start": [x,y], "end": [x,y]}, "top right": {...}}`
- `fill_lines_json`: JSON string — `[{"fill": 95.0, "p1": [x1, y1], "p2": [x2, y2]}, ...]`

Both `walls_json` and `fill_lines_json` are required (no CSV fallback in the current server code).

## Output structure

Each `/predict` call saves to `OUTPUT_DIR`:
- `masks/<timestamp>_<uuid>_mask.png`
- `overlays/<timestamp>_<uuid>_overlay.jpg`
- `<timestamp>_<uuid>_summary.json`
