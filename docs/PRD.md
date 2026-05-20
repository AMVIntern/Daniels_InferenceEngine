# PRD.md — Inference Engine: Optimization & Enhancement Plan

> **Status: All four phases implemented and shipped — 2026-05-20**
> Sections below describe both the original intent and the actual as-built behaviour where they diverge.

---

## 1. Overview and Problem Statement

The DanielsHealth Washline Inference Engine is a Python/FastAPI service that estimates how full laundry bins are from camera images. It runs on a GPU PC (Linux), is exposed over ngrok, and is called by a C# WPF client application on a separate machine.

The system had accumulated dead code from earlier development iterations, had not been profiled for latency, was missing a secondary anomaly detection capability that a trained ONNX model already existed for, and did not capture per-request metadata (barcode, container type) that the client already knew at call time. These gaps limited operational usefulness, made the codebase harder to maintain, and left performance headroom unrealised.

This PRD describes four sequential phases of work to address these problems.

---

## 2. Current Architecture (as-built, post all phases)

### 2.1 End-to-end data flow

```
C# WPF Client (camera PC)
  │
  ├─ Captures frame from camera
  ├─ Reads container barcode
  ├─ Looks up container type → derives wall lines + fill-level reference lines
  └─ POST /predict (multipart/form-data)
         file             : JPEG/PNG, BGR-encoded
         walls_json       : {"bottom right": {"start":[x,y], "end":[x,y]}, "top right": {...}}
         fill_lines_json  : [{"fill": 95.0, "p1":[x1,y1], "p2":[x2,y2]}, ...]
         barcode          : optional container barcode string
         container_type   : optional container type identifier

FastAPI Server (GPU PC, Linux, exposed via ngrok)
  fill_estimator_api.py
  │
  ├─ 0. Sanitise barcode + container_type (filesystem-safe labels)
  ├─ 1. Parse walls_json + fill_lines_json from form data
  ├─ 2. Decode image bytes → PIL Image → numpy RGB array; compute MD5 hash
  ├─ 3. YOLO detect (yolo_detector.YoloDetector) — full image
  │      └─ letterbox → ONNX session.run → decode + NMS → best bbox
  │      └─ If no detection → return fill=0, forced_zero=true
  ├─ 4. Anomaly detection (anomaly_classifier.AnomalyRegistry.run_all) — FULL image
  │      └─ YoloxAnomalyClassifier: letterbox to 1024×1024, no /255 norm
  │      └─ Per-class NMS → AnomalyResult (name, detected, score, class_name, detection_count, detections)
  ├─ 5. SAM segment (sam_segmenter_large.SamSegmenter)
  │      └─ Check LRU embedding cache by image MD5 hash
  │      └─ image_encoder (ViT-H, 1024×1024) — only if cache miss
  │      └─ Pass 1: multi-mask decode with bbox + center-point prompts
  │      └─ Pass 2: single-mask decode using Pass-1 union as mask prompt
  │      └─ Sigmoid threshold (default 0.30) → binary mask → clean_mask
  ├─ 6. generate_parallel_lines (wall_utils) → 100 scan lines across bin
  ├─ 7. collect_hit_points: march each scan line right→left to find waste surface
  ├─ 8. compute_fill_level: median hit-x → linear interpolation on ref lines
  ├─ 9. build_overlay_image: mask + fill lines + hit points rendered on frame
  ├─ 10. Async file I/O (ThreadPoolExecutor, non-blocking)
  │      ├─ masks/{ts}_{barcode}_{container}_{uuid}_mask.png
  │      ├─ overlays/{ts}_{barcode}_{container}_{uuid}_overlay.jpg
  │      ├─ anomaly/{ts}_{barcode}_{container}_{uuid}_anomaly.jpg  ← full image with bbox drawn
  │      └─ {ts}_{barcode}_{container}_{uuid}_summary.json
  └─ 11. Return PredictionResponse JSON
          {
            success, fill_level, num_hits, largest_area, forced_zero, rep_point,
            anomaly_results: [{ name, detected, score, class_name, detection_count }],
            barcode, container_type, output_path, error
          }
```

### 2.2 Module inventory (final state)

| File | Status | Role |
|---|---|---|
| `fill_estimator_api.py` | Active | FastAPI app — orchestrates the full pipeline, exposes `/predict` |
| `sam_segmenter_large.py` | Active | SAM ViT-H wrapper: `segment_anything` lib, bbox+center prompt, two-pass refinement, LRU embedding cache |
| `yolo_detector.py` | Active | YOLOX ONNX inference — letterbox, decode, NMS. Accepts `classes` param |
| `anomaly_classifier.py` | Active | `AnomalyResult`, `AnomalyClassifierBase` protocol, `AnomalyRegistry`, `YoloxAnomalyClassifier` |
| `wall_utils.py` | Active | `generate_parallel_lines` + `march_mask` |
| `fill_interpolator.py` | Active | `estimate_fill_from_hit` — x-coordinate → fill % linear interpolation |
| `visualizer.py` | Active | Draws mask overlays, fill-level reference lines, hit points |
| `fill_line_loader.py` | Utility | Used by `run_mask_inference.py` only |
| `run_mask_inference.py` | Standalone utility | Batch offline script — no YOLO/SAM, processes pre-generated masks |
| `sam_segmenter.py` | **Deleted** | Superseded by `sam_segmenter_large.py` — removed in Phase 1 |

### 2.3 Key design decisions

- **Anomaly runs on the full image**, not the YOLO crop. The YOLOX anomaly model was trained on full frames; cropping introduced artifacts. PRD Phase 3 originally specified the crop — overridden during implementation.
- **`AnomalyResultSummary` in API response**: The internal `AnomalyResult` carries the full `detections` list (each bbox, score, class_id) for disk logging. The HTTP response returns a slim `AnomalyResultSummary` (name, detected, score, class_name, detection_count only) to keep the response payload small.
- **Anomaly bbox image saved separately**: Every request saves a full-resolution JPEG with bounding boxes drawn to `RESULT/api_results/anomaly/`, alongside the mask and overlay images.
- **SAM embedding LRU cache**: Keyed by MD5 hash of raw image bytes. Cache size controlled by `SAM_CACHE_SIZE` env var (default 4). Yields near-zero latency for repeated identical frames from a polling client.
- **Async file I/O**: All disk writes (`mask.png`, `overlay.jpg`, `anomaly.jpg`, `summary.json`) offloaded to a `ThreadPoolExecutor` — caller receives HTTP response before writes complete.

---

## 3. Goals and Success Metrics

### Phase 1 — Code Cleanup ✅

| Goal | Status |
|---|---|
| Remove all dead code | Done — dead imports, globals, functions removed |
| No regression | `/predict` response schema unchanged |
| Codebase clarity | Each file has a single clear responsibility |

### Phase 2 — Inference Speed Optimisation ✅

| Goal | Status |
|---|---|
| Reduce end-to-end latency | ONNX `ORT_ENABLE_ALL`, async I/O, warm-up implemented |
| Consistent response time | Warm-up in `lifespan` startup |
| Embedding cache | LRU cache in `sam_segmenter_large.py`, default size 4 |
| SAM FP16 | Optional via `SAM_FP16=true` env var |

### Phase 3 — Anomaly Detection Integration ✅

| Goal | Status |
|---|---|
| Anomaly result in response | `anomaly_results: List[AnomalyResultSummary]` in every `/predict` response |
| Count of detections | `detection_count` (int) in each `AnomalyResultSummary` |
| Bbox image saved | Full image with drawn boxes saved to `api_results/anomaly/` |
| Pluggable architecture | `AnomalyRegistry.register()` — adding a second classifier requires no pipeline changes |

### Phase 4 — Enriched Input/Output ✅

| Goal | Status |
|---|---|
| Barcode + container type accepted | Both optional form fields; backward-compatible |
| Output filenames include metadata | `{ts}_{barcode}_{container}_{uuid}_{suffix}` |
| Metadata echoed in response | `barcode` and `container_type` in `PredictionResponse` |

---

## 4. Phase 1 — Code Cleanup (Implemented)

### 4.1 Changes made

| Item | Action |
|---|---|
| `sam_segmenter.py` | Deleted — entirely superseded by `sam_segmenter_large.py` |
| `load_fill_level_lines` import | Removed from `fill_estimator_api.py` |
| `parse_walls_csv` import | Removed from `fill_estimator_api.py` |
| `point_line_signed_distance` | Removed from `fill_interpolator.py` |
| `walls`, `ref_lines` module globals | Removed from `fill_estimator_api.py`; replaced by local variables |
| `WALLS_CSV`, `FILL_LINES_CSV` env vars | Removed from `.env.example` |
| Step numbering in print statements | Corrected to match the full 7-step pipeline |

---

## 5. Phase 2 — Inference Speed Optimisation (Implemented)

### 5.1 Changes made

| Optimisation | Implementation |
|---|---|
| **Warm-up inference** | `_warmup()` called in `lifespan` startup — runs dummy YOLO + SAM + anomaly pass |
| **SAM embedding LRU cache** | `_EmbedCache` class in `sam_segmenter_large.py`, `OrderedDict`-backed, keyed by image MD5 |
| **Async file I/O** | `_save_outputs()` dispatched via `loop.run_in_executor(executor, ...)` |
| **ONNX session options** | `ORT_ENABLE_ALL` graph optimisation on both YOLO and anomaly sessions |
| **Provider logging** | Active execution provider (CUDA vs CPU) logged at startup for both models |
| **SAM FP16** | `SAM_FP16=true` half-precision encoder — `self.sam = self.sam.half()`, input cast to `torch.float16` |

### 5.2 Configuration

| Env var | Default | Purpose |
|---|---|---|
| `SAM_CACHE_SIZE` | `4` | LRU embedding cache entries (0 = disabled) |
| `SAM_FP16` | `false` | Half-precision SAM encoder on CUDA — validate accuracy before enabling |

---

## 6. Phase 3 — YOLOX Anomaly Model Integration (Implemented)

### 6.1 Architecture (as-built)

```
anomaly_classifier.py
│
├─ AnomalyDetection(BaseModel)
│    bbox: List[int]       # [x1, y1, x2, y2] absolute pixel coords
│    score: float
│    class_id: int
│    class_name: str
│
├─ AnomalyResult(BaseModel)
│    name: str
│    detected: bool
│    score: float           # highest single detection score
│    class_name: str        # top detection class label or "none"
│    detection_count: int   # number of boxes above conf_thr after NMS
│    detections: List[AnomalyDetection]   # all boxes (stored to disk; not in API response)
│
├─ AnomalyClassifierBase (Protocol, runtime_checkable)
│    name: str
│    classify(img_bgr: np.ndarray) -> AnomalyResult
│
├─ AnomalyRegistry
│    register(classifier)
│    run_all(img_bgr) -> List[AnomalyResult]
│    Error isolation: one classifier failure never blocks the rest
│
└─ YoloxAnomalyClassifier
     Fixed 1024×1024 input, letterbox with 114-pad, no /255 normalisation
     Own ONNX session (not shared with YoloDetector)
     Per-class NMS, results sorted by score desc
     conf_thr=0.30, nms_thr=0.45 (defaults)
```

### 6.2 API response (slim summary — no detections array)

```python
class AnomalyResultSummary(BaseModel):
    name:            str
    detected:        bool
    score:           float
    class_name:      str
    detection_count: int = 0
```

Full detection data (with bboxes) is written to `*_summary.json` on disk.

### 6.3 Anomaly scope: full image (deviation from original PRD)

The original PRD specified running anomaly inference on the YOLO-cropped region. During implementation this was changed to run on the **full image** because the anomaly model was trained on full frames and cropping caused coordinate/scale mismatches. Bounding boxes returned by the model are already in full-image pixel coordinates.

### 6.4 Anomaly bbox visualisation

`_draw_anomaly_image(img_np_rgb, anomaly_results_raw)` draws:
- Red bounding box rectangle per detection
- Label chip above box: `{class_name} {score:.2f}`
- "No anomaly detected" text overlay when `detected=False`

Saved to `RESULT/api_results/anomaly/{safe_name}_anomaly.jpg`.

### 6.5 Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANOMALY_ENABLED` | `true` | Kill-switch — set `false` to skip all anomaly inference |
| `ANOMALY_MODEL_PATH` | `Model/Anomaly/yolox_m_washanomaly20MAY.onnx` | Path to anomaly ONNX checkpoint |
| `ANOMALY_MODEL_CLASSES` | `waste` | Comma-separated class names (must match training order) |
| `ANOMALY_CONF_THR` | `0.30` | Detection confidence threshold |

### 6.6 Adding a second anomaly classifier

Only two lines needed in `fill_estimator_api.py` lifespan:

```python
clf2 = YoloxAnomalyClassifier(name="my_model", onnx_path="...", classes=["classA"], conf_thr=0.5)
anomaly_registry.register(clf2)
```

No pipeline or schema code changes required.

---

## 7. Phase 4 — Enriched API Input and Output Naming (Implemented)

### 7.1 New form fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `barcode` | `Optional[str]` | `"unknown"` | Container barcode |
| `container_type` | `Optional[str]` | `"unknown"` | Container type identifier |

Both are sanitised with `re.sub(r"[^A-Za-z0-9_\-]", "_", value.strip())` before use in filenames.

### 7.2 Output filename format

```
{YYYYMMDD}_{HHMMSS}_{barcode}_{container_type}_{8-char-uuid}_{suffix}

Examples:
  20260520_101108_BC111_64L_a3f2b1c0_mask.png
  20260520_101108_BC111_64L_a3f2b1c0_overlay.jpg
  20260520_101108_BC111_64L_a3f2b1c0_anomaly.jpg
  20260520_101108_BC111_64L_a3f2b1c0_summary.json
```

### 7.3 Summary JSON structure

```json
{
  "image": "frame.jpg",
  "barcode": "BC111",
  "container_type": "64L",
  "num_hits": 100,
  "largest_area": 290775,
  "avg_fill": 78.44,
  "forced_zero": false,
  "rep_point": [844.0, 942.0],
  "anomaly_results": [
    {
      "name": "yolox_washanomaly",
      "detected": true,
      "score": 0.873,
      "class_name": "waste",
      "detection_count": 2,
      "detections": [
        { "bbox": [x1,y1,x2,y2], "score": 0.873, "class_id": 0, "class_name": "waste" },
        ...
      ]
    }
  ],
  "mask_path": "...",
  "overlay_path": "...",
  "anomaly_path": "..."
}
```

### 7.4 PredictionResponse schema (final)

```python
class PredictionResponse(BaseModel):
    success:         bool
    fill_level:      Optional[float]
    num_hits:        int
    largest_area:    int
    forced_zero:     bool
    rep_point:       Optional[List[float]]
    anomaly_results: List[AnomalyResultSummary] = []
    barcode:         Optional[str] = None
    container_type:  Optional[str] = None
    output_path:     Optional[str] = None
    error:           Optional[str] = None
```

---

## 8. Non-Functional Requirements (cross-cutting)

| Requirement | Target | Status |
|---|---|---|
| Backward compatibility | `/predict` response valid for all existing callers; new fields additive | Met |
| Latency — warm path | p95 ≤ 2 s (no cache hit) on single CUDA GPU | Implemented; benchmark on target hardware |
| Latency — cache hit | p95 ≤ 300 ms (repeated identical frame) | Implemented; validate on target hardware |
| Anomaly overhead | ≤ 200 ms additional latency | ONNX CUDA session, ORT_ENABLE_ALL |
| Extensibility | New anomaly classifier = config + one `register()` call | Met |
| Filename safety | Filesystem-safe across Linux and Windows | Met via `_sanitise_label()` |
| Startup time | Server ready within 60 s of process start | Warm-up is one forward pass (~3–5 s) |

---

## 9. Out of Scope

- Changes to the C# WPF client application.
- Model retraining or fine-tuning of YOLO, SAM, or the anomaly model.
- Multi-camera or multi-bin batching within a single request.
- Authentication or rate-limiting on the API.
- Persistent database storage of results.
- Horizontal scaling or containerisation.
- The `run_mask_inference.py` batch script (cleanup limited to not breaking it).

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SAM FP16 reduces mask accuracy | Medium | High | `SAM_FP16` env var defaults to `false`; validate before enabling in production |
| Embedding cache stale result for similar frames | Low | Medium | Cache keyed on full MD5 of image bytes; `SAM_CACHE_SIZE=0` kill-switch available |
| Async file I/O errors are silent | Low | Low | Errors logged to stdout from background thread; `_summary.json` paths captured |
| Filename sanitisation too aggressive for some barcode formats | Low | Low | Allowlist `[A-Za-z0-9_-]`; documented in CLAUDE.md |
| GPU VRAM: SAM ViT-H + 2× YOLOX models | Medium | Medium | Validate total VRAM on target hardware; disable SAM FP16 first if OOM |

---

## 11. Implementation Order and Milestones

| Milestone | Phase | Status |
|---|---|---|
| 1 | Phase 1: Code Cleanup | ✅ Complete |
| 2 | Phase 2: Inference Speed Optimisation | ✅ Complete |
| 3 | Phase 3: Anomaly Detection Integration | ✅ Complete |
| 4 | Phase 4: Enriched API Input/Output | ✅ Complete |

---

*Document version: 2.0 — 2026-05-20 (updated to reflect as-built state)*
