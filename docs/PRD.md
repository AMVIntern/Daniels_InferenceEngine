# PRD.md — Inference Engine: Optimization & Enhancement Plan

---

## 1. Overview and Problem Statement

The DanielsHealth Washline Inference Engine is a Python/FastAPI service that estimates how full laundry bins are from camera images. It runs on a GPU PC (Linux), is exposed over ngrok, and is called by a C# WPF client application on a separate machine.

The current system is functional but has accumulated dead code from earlier development iterations, has not been profiled for latency, is missing a secondary anomaly detection capability that a trained ONNX model already exists for, and does not capture per-request metadata (barcode, container type) that the client already knows at call time. These gaps limit operational usefulness, make the codebase harder to maintain, and leave performance headroom unrealised.

This PRD describes four sequential phases of work to address these problems.

---

## 2. Current Architecture

### 2.1 End-to-end data flow

```
C# WPF Client (camera PC)
  │
  ├─ Captures frame from camera
  ├─ Reads container barcode
  ├─ Looks up container type → derives wall lines + fill-level reference lines
  └─ POST /predict (multipart/form-data)
         file          : JPEG/PNG, BGR-encoded
         walls_json    : {"bottom right": {"start":[x,y], "end":[x,y]}, "top right": {...}}
         fill_lines_json: [{"fill": 95.0, "p1":[x1,y1], "p2":[x2,y2]}, ...]

FastAPI Server (GPU PC, Linux, exposed via ngrok)
  fill_estimator_api.py
  │
  ├─ 1. Parse walls_json + fill_lines_json from form data
  ├─ 2. Decode image bytes → PIL Image → numpy RGB array
  ├─ 3. YOLO detect (yolo_detector.YoloDetector)
  │      └─ letterbox → ONNX session.run → decode + NMS → best bbox
  │      └─ If no detection → return fill=0, forced_zero=true
  ├─ 4. SAM segment (sam_segmenter_large.SamSegmenter)
  │      └─ image_encoder (ViT-H, 1024×1024) ← heaviest step
  │      └─ Pass 1: multi-mask decode with bbox + center-point prompts
  │      └─ Pass 2: single-mask decode using Pass-1 union as mask prompt
  │      └─ Sigmoid threshold (default 0.30) → binary mask → clean_mask
  ├─ 5. generate_parallel_lines (wall_utils) → 100 scan lines across bin
  ├─ 6. collect_hit_points: march each scan line right→left to find waste surface
  ├─ 7. compute_fill_level: median hit-x → linear interpolation on ref lines
  ├─ 8. build_overlay_image: mask + fill lines + hit points rendered on frame
  ├─ 9. Save to RESULT/api_results/
  │      ├─ masks/{ts}_{uuid}_mask.png
  │      ├─ overlays/{ts}_{uuid}_overlay.jpg
  │      └─ {ts}_{uuid}_summary.json
  └─ 10. Return PredictionResponse JSON
          { success, fill_level, num_hits, largest_area, forced_zero, rep_point, output_path, error }
```

### 2.2 Module inventory

| File | Status | Notes |
|---|---|---|
| `fill_estimator_api.py` | Active | Main FastAPI app and pipeline orchestrator |
| `sam_segmenter_large.py` | Active | SAM ViT-H via `segment_anything` library |
| `yolo_detector.py` | Active | YOLOX ONNX, single class `waste` |
| `wall_utils.py` | Partially active | `generate_parallel_lines` + `march_mask` used; `parse_walls_csv` imported but never called |
| `fill_interpolator.py` | Partially active | `estimate_fill_from_hit` used; `point_line_signed_distance` defined but never called |
| `visualizer.py` | Active | Overlay drawing |
| `fill_line_loader.py` | Dead import | Imported in API but `load_fill_level_lines` is never called (fill lines parsed inline from JSON) |
| `sam_segmenter.py` | Dead file | SAM ViT-B via HuggingFace transformers; entirely superseded by `sam_segmenter_large.py` |
| `run_mask_inference.py` | Standalone utility | Batch offline script; not part of the API runtime |
| `Model/Anomaly/*.onnx` | Dormant | Anomaly detection ONNX model exists but has zero code wired to it |
| `launch.sh` | Utility | Linux gnome-terminal launcher for server + ngrok |
| `images/*/bigbin*.json`, `medium bin.js`, `small bin.js` | Reference only | COCO annotation exports from labelling tool; not used at runtime |

### 2.3 Global state problems

`fill_estimator_api.py` declares four globals (`sam`, `yolo`, `walls`, `ref_lines`) at module level. `sam` and `yolo` are correctly initialised at startup and reused across requests. `walls` and `ref_lines` were originally loaded from CSVs at startup but that code was removed; they are now shadowed by local variables inside `predict_fill_level`, making the globals misleading dead weight.

### 2.4 Output filename format (current)

```
{YYYYMMDD}_{HHMMSS}_{microseconds}_{8-char-uuid}_{mask|overlay|summary}
e.g. 20260520_103036_010260_48cb12b2_overlay.jpg
```

Barcode and container type are not captured anywhere in the saved outputs or response payload.

---

## 3. Goals and Success Metrics

### Phase 1 — Code Cleanup

| Goal | Success Metric |
|---|---|
| Remove all dead code | Zero unused imports, functions, or module-level variables remaining |
| No regression | All existing API behaviours preserved; `/predict` response schema unchanged |
| Codebase clarity | Each file has a single clear responsibility |

### Phase 2 — Inference Speed Optimisation

| Goal | Success Metric |
|---|---|
| Reduce end-to-end latency | p95 latency ≤ 2 s per request on a single-GPU machine (from current ~4–6 s estimate) |
| Consistent response time | First-request latency within 10% of steady-state (warm-up done at startup) |
| No accuracy regression | fill_level output unchanged for identical inputs before/after optimisation |

### Phase 3 — Anomaly Detection Integration

| Goal | Success Metric |
|---|---|
| Anomaly result in response | `anomaly_detected` (bool) and `anomaly_score` (float) present in every `/predict` response |
| Pluggable architecture | Adding a second anomaly classifier requires touching only a config/registry, not pipeline logic |
| Latency budget respected | Anomaly inference adds ≤ 200 ms to p95 latency (runs on YOLO-cropped region, ONNX session) |

### Phase 4 — Enriched Input/Output

| Goal | Success Metric |
|---|---|
| Barcode + container type accepted | `/predict` accepts both fields without breaking existing callers that omit them |
| Output filenames include metadata | Saved files follow `{ts}_{barcode}_{container_type}_{mask|overlay|summary}` naming |
| Metadata in summary JSON | `barcode` and `container_type` present in every `*_summary.json` |

---

## 4. Phase 1 — Code Cleanup

### 4.1 Items to remove or fix

**Dead file: `sam_segmenter.py`**
The original SAM ViT-B wrapper using HuggingFace `transformers` and grid-point prompting. It has been entirely superseded by `sam_segmenter_large.py` (ViT-H, `segment_anything`, bbox prompting). No file in the project imports it. Remove the file and remove `transformers` and `torchvision` from `requirements.txt` if they are only used here (verify first).

**Dead import: `fill_line_loader.py` / `load_fill_level_lines`**
`fill_estimator_api.py` imports `load_fill_level_lines` at the top but never calls it — fill lines are parsed inline from `fill_lines_json`. Remove the import. `fill_line_loader.py` itself can be kept as a utility (it is used by `run_mask_inference.py` indirectly and may be useful for tooling), but the dead import in the API must be removed.

**Dead import: `wall_utils.parse_walls_csv`**
Imported in `fill_estimator_api.py` but never called — the CSV loading path was removed when the API was refactored to accept JSON. Remove from the import line.

**Dead function: `point_line_signed_distance` in `fill_interpolator.py`**
Defined but never called anywhere in the project. The active `estimate_fill_from_hit` function uses only x-midpoint interpolation. Remove.

**Dead global state: `walls` and `ref_lines` module-level variables in `fill_estimator_api.py`**
Declared as `None` globals alongside `sam` and `yolo`, giving the false impression they are loaded at startup. They are immediately shadowed by local variables inside `predict_fill_level`. Remove the globals; keep only `sam` and `yolo` as module-level state.

**Dead env vars: `WALLS_CSV` and `FILL_LINES_CSV`**
Referenced in `.env.example` but commented out in the API code. Remove from `.env.example` to avoid confusion.

**Misleading print statement formatting**
`fill_estimator_api.py` has `print("\n[2/3]...")` and `print("\n[3/3]...")` inside `predict_fill_level` — these are numbered as if there are only 3 steps but the full pipeline has 7. Renumber or remove.

### 4.2 Items to retain as-is

`run_mask_inference.py` — standalone batch utility, useful for offline testing. Keep. Its own inline `parse_walls_csv` is intentional (different CSV label convention) and separate from `wall_utils`.

`images/` annotation JSONs and `Refrence_fill_levels/` CSVs — calibration reference data, not dead code.

`launch.sh` — operational utility for the Linux deployment machine.

### 4.3 Functional requirements

- FR1.1: Delete `sam_segmenter.py`.
- FR1.2: Remove dead import `load_fill_level_lines` from `fill_estimator_api.py`.
- FR1.3: Remove dead import `parse_walls_csv` from `fill_estimator_api.py`.
- FR1.4: Delete `point_line_signed_distance` from `fill_interpolator.py`.
- FR1.5: Remove `walls` and `ref_lines` module-level globals from `fill_estimator_api.py`.
- FR1.6: Remove `WALLS_CSV` and `FILL_LINES_CSV` from `.env.example`.
- FR1.7: After all removals, verify `/predict` returns identical output for a reference image.

---

## 5. Phase 2 — Inference Speed Optimisation

### 5.1 Current bottleneck analysis

**SAM image encoder (dominant cost, ~80% of latency)**
`SamSegmenter.predict_mask` calls `self.sam.image_encoder(t)` on every request. The ViT-H encoder processes a 1024×1024 tensor — this is a full transformer forward pass and is by far the most expensive operation. There is no caching or reuse across requests.

**No warm-up**
At server startup SAM and YOLO are loaded but no inference is run. The first production request pays JIT compilation cost (PyTorch) and CUDA lazy initialisation overhead.

**Redundant image colour conversions**
The pipeline converts BGR→RGB (PIL), then RGB→BGR for YOLO, then BGR→RGB again for the final overlay. These are cheap but unnecessary roundtrips.

**Synchronous ONNX session for YOLO**
`YoloDetector` uses a single `ort.InferenceSession`. The session itself is correctly reused, but it is configured with both `CUDAExecutionProvider` and `CPUExecutionProvider` without explicit device configuration or session options.

**No async I/O overlap**
Image file saving (mask PNG, overlay JPG, summary JSON) is done synchronously inside the request handler, adding latency that the caller does not benefit from.

### 5.2 Proposed optimisations

**O1 — Warm-up inference at startup**
During the `lifespan` startup handler, after models are loaded, run one dummy inference pass through both YOLO and SAM on a synthetic input of representative size (e.g. 1368×1368 zeros). This ensures CUDA kernels are compiled and cached before the first real request arrives.

**O2 — SAM encoder result caching (per-image hash)**
The image encoder output (embeddings) is the same for any two identical images. Compute a fast hash (xxhash or image shape + first/last 64 bytes) of the incoming image bytes. Cache the last N encoder embeddings (LRU, N=4) keyed by hash. On a cache hit, skip `image_encoder` and go directly to `prompt_encoder` + `mask_decoder`. This yields near-zero latency for repeated frames from a polling client.

**O3 — Offload file I/O to a background thread**
Move saving of mask PNG, overlay JPG, and summary JSON to a `ThreadPoolExecutor` using `asyncio.get_event_loop().run_in_executor(...)`. The API response is returned to the caller as soon as fill level is computed; file writing completes asynchronously. This alone will cut caller-perceived latency by the time it takes to write ~2–3 image files.

**O4 — Eliminate redundant colour space conversions**
Restructure the pipeline to maintain a single canonical colour space (RGB) and convert to BGR only once, immediately before YOLO inference. Remove the second BGR→RGB conversion before overlay building.

**O5 — ONNX session optimisation for YOLO**
Configure `ort.InferenceSession` with `SessionOptions` setting `graph_optimization_level = ORT_ENABLE_ALL` and `intra_op_num_threads` appropriate for the GPU machine. Confirm CUDA execution provider is actually selected (log provider used at startup).

**O6 — Half-precision SAM inference (optional, evaluate accuracy impact)**
Run SAM encoder in `torch.float16` on CUDA. This halves memory bandwidth for the ViT-H encoder and typically improves throughput by ~1.5–2×. Requires validation that mask quality is equivalent within acceptable tolerance. Implement as an opt-in env var `SAM_FP16=true`.

### 5.3 Functional requirements

- FR2.1: Warm-up inference must complete during `lifespan` startup before the server begins accepting requests.
- FR2.2: Encoder embedding cache must be bounded (LRU, configurable via env var `SAM_CACHE_SIZE`, default 4).
- FR2.3: File I/O must be non-blocking from the perspective of the HTTP response.
- FR2.4: `output_path` in the response must still be returned (it can be the intended path even if writing is still in progress).
- FR2.5: `/predict` response schema must not change.
- FR2.6: `SAM_FP16` env var controls half-precision mode (default `false`).

### 5.4 Non-functional requirements

- p95 end-to-end latency ≤ 2 s (warm path, no cache hit) on a single CUDA GPU.
- p95 latency ≤ 300 ms on cache hit (repeated identical frame).
- No change to fill level output values for identical inputs.

---

## 6. Phase 3 — YOLOX Anomaly Model Integration

### 6.1 Context

A trained YOLOX anomaly detection ONNX model already exists at `Model/Anomaly/yolox_m_washanomaly20MAY.onnx`. It is intended to run on the YOLO-cropped bin region (not the full image) and classify whether anomalous content is present. The architecture must accommodate future additional classifiers (e.g. a different anomaly type, a contamination detector) without restructuring the pipeline.

### 6.2 Proposed architecture

Introduce an `AnomalyClassifier` abstraction and a runtime registry. Each classifier is a lightweight wrapper around a YOLOX ONNX session (or any future model type) that accepts a cropped BGR numpy array and returns a typed result.

```
AnomalyResult
  name: str           # classifier identifier
  detected: bool
  score: float        # highest detection confidence, 0.0 if no detection
  class_name: str     # detected class label or "none"

AnomalyClassifierBase (protocol/ABC)
  name: str
  def classify(img_bgr: np.ndarray) -> AnomalyResult

YoloxAnomalyClassifier(AnomalyClassifierBase)
  Wraps an ort.InferenceSession (reuses YoloDetector decode logic)
  Loaded from a given ONNX path + class list

AnomalyRegistry
  register(classifier: AnomalyClassifierBase)
  run_all(img_bgr: np.ndarray) -> List[AnomalyResult]
```

At startup, the registry is populated from a configuration block (env vars or a small JSON config file). Initially one classifier is registered: `yolox_washanomaly` from `Model/Anomaly/yolox_m_washanomaly20MAY.onnx`. Adding a second classifier later means adding one entry to config and one `register()` call — no pipeline code changes.

In the pipeline, anomaly classification runs immediately after the YOLO crop step (step 3), on the cropped region `img_bgr[y1:y2, x1:x2]`, in parallel with SAM segmentation if GPU memory permits (otherwise sequentially before SAM).

### 6.3 Pipeline change

```
Step 3a (existing): YOLO detect → bbox
Step 3b (new):      AnomalyRegistry.run_all(cropped region) → List[AnomalyResult]
Step 4  (existing): SAM segment full image with bbox prompt
...
Step 10 (updated):  Return PredictionResponse including anomaly_results
```

### 6.4 Response schema additions

```python
class AnomalyResult(BaseModel):
    name: str
    detected: bool
    score: float
    class_name: str

class PredictionResponse(BaseModel):
    # existing fields unchanged
    success: bool
    fill_level: Optional[float]
    num_hits: int
    largest_area: int
    forced_zero: bool
    rep_point: Optional[list[float]]
    output_path: Optional[str]
    error: Optional[str]
    # new
    anomaly_results: List[AnomalyResult] = []
```

`anomaly_results` defaults to empty list so existing callers that ignore it are unaffected.

### 6.5 Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANOMALY_MODEL_PATH` | `Model/Anomaly/yolox_m_washanomaly20MAY.onnx` | Path to first anomaly classifier |
| `ANOMALY_MODEL_CLASSES` | `"anomaly"` | Comma-separated class names for anomaly model |
| `ANOMALY_CONF_THR` | `0.5` | Confidence threshold for anomaly detection |
| `ANOMALY_ENABLED` | `true` | Kill-switch to disable all anomaly inference |

### 6.6 Functional requirements

- FR3.1: `AnomalyClassifierBase` protocol defined; `YoloxAnomalyClassifier` implements it.
- FR3.2: `AnomalyRegistry` supports registering multiple classifiers and running all of them.
- FR3.3: Anomaly inference runs on the YOLO-cropped region only.
- FR3.4: If `ANOMALY_ENABLED=false` or the model file is missing, `anomaly_results` is returned as `[]` with no error.
- FR3.5: If YOLO finds no detection (forced_zero path), anomaly classification is skipped and `anomaly_results` is `[]`.
- FR3.6: `PredictionResponse` includes `anomaly_results: List[AnomalyResult]`.
- FR3.7: Summary JSON saved to disk includes anomaly results.
- FR3.8: Existing `/predict` callers that do not read `anomaly_results` are unaffected (additive schema change only).

### 6.7 Non-functional requirements

- Anomaly inference must add ≤ 200 ms to p95 latency.
- The registry pattern must allow a second classifier to be added with changes confined to startup/config code.

---

## 7. Phase 4 — Enriched API Input and Output Naming

### 7.1 Context

The C# client already knows the container barcode and container type at the time it calls `/predict`. Currently this context is discarded — it is not passed to the server, not stored in output filenames, and not included in summary JSON. This makes it impossible to correlate saved overlays and summaries with specific containers after the fact.

### 7.2 Input changes

Add two optional form fields to `/predict`:

| Field | Type | Description |
|---|---|---|
| `barcode` | `Optional[str]` | Container barcode (e.g. `"BC111"`) |
| `container_type` | `Optional[str]` | Container type identifier (e.g. `"64L"`, `"32L"`) |

Both fields are optional to maintain backward compatibility with existing callers. When absent, they default to `"unknown"`.

### 7.3 Output filename format

Replace the current `{ts}_{uuid}` naming with a human-readable convention that includes operational context:

```
{YYYYMMDD}_{HHMMSS}_{barcode}_{container_type}_{suffix}

Examples:
  20260520_101108_BC111_64L_mask.png
  20260520_101108_BC111_64L_overlay.jpg
  20260520_101108_BC111_64L_summary.json
```

If barcode or container_type is `"unknown"` (not supplied), that segment still appears literally as `"unknown"` so filenames remain parseable.

Both fields must be sanitised before use in filenames (strip or replace characters outside `[A-Za-z0-9_-]`) to prevent path traversal or filesystem errors.

The UUID component is retained as a suffix on the stem to guarantee uniqueness when the same barcode is scanned multiple times within the same second:

```
20260520_101108_BC111_64L_{8-char-uuid}_overlay.jpg
```

### 7.4 Summary JSON additions

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
  "anomaly_results": [...],
  "mask_path": "...",
  "overlay_path": "..."
}
```

### 7.5 Response schema additions

```python
class PredictionResponse(BaseModel):
    # existing + phase 3 fields
    ...
    # new
    barcode: Optional[str] = None
    container_type: Optional[str] = None
```

These are echoed back in the response so the caller can confirm the values were received correctly.

### 7.6 Functional requirements

- FR4.1: `/predict` accepts `barcode` and `container_type` as optional form fields.
- FR4.2: Both fields default to `"unknown"` when absent.
- FR4.3: Both fields are sanitised (non-alphanumeric/hyphen/underscore chars replaced with `_`) before use in filenames.
- FR4.4: Output filenames include sanitised barcode and container_type.
- FR4.5: Summary JSON includes `barcode` and `container_type`.
- FR4.6: `PredictionResponse` echoes `barcode` and `container_type`.
- FR4.7: Existing callers that omit both fields receive valid responses and valid output files (with `"unknown"` in the filename).

---

## 8. Non-Functional Requirements (cross-cutting)

| Requirement | Target |
|---|---|
| Backward compatibility | `/predict` response must remain valid for all existing callers throughout all phases; new fields are additive |
| Latency (Phase 2 complete) | p95 ≤ 2 s warm path; ≤ 300 ms on embedding cache hit |
| Anomaly overhead | ≤ 200 ms additional latency |
| GPU memory | Total model memory (SAM ViT-H + 2× YOLOX + anomaly) must fit in GPU VRAM; validate on target hardware |
| Extensibility | Adding a new anomaly classifier = config change + one `register()` call only |
| Filename safety | Output filenames are always filesystem-safe across Linux and Windows |
| Startup time | Warm-up adds to startup time but server must be ready within 60 s of process start |

---

## 9. Out of Scope

- Changes to the C# WPF client application (this PRD covers server-side only).
- Model retraining or fine-tuning of YOLO, SAM, or the anomaly model.
- Multi-camera or multi-bin batching within a single request.
- Authentication or rate-limiting on the API (handled externally via ngrok).
- Persistent database storage of results (file-based output is sufficient).
- Horizontal scaling or containerisation (single-GPU deployment assumed).
- The `run_mask_inference.py` batch script — cleanup limited to not breaking it.

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SAM FP16 reduces mask accuracy | Medium | High | Gate behind `SAM_FP16` env var (default off); validate against reference images before enabling in production |
| Embedding cache returns stale result for visually similar but distinct frames | Low | Medium | Key cache on full image bytes hash (not shape), ensure cache is per-server-instance only, provide `SAM_CACHE_SIZE=0` kill-switch |
| Anomaly ONNX model input size differs from YOLO crop dimensions | Medium | Medium | Letterbox the crop to the anomaly model's declared input shape (same letterbox function already exists) |
| Async file I/O errors are silent (response already sent) | Low | Low | Log errors from background write tasks; expose a `/health` endpoint count of failed writes |
| Filename sanitisation strips too aggressively for some barcode formats | Low | Low | Use allowlist `[A-Za-z0-9_-]`, replace others with `_`; document the convention |
| Phase 2 warm-up increases cold-start time unacceptably | Low | Low | Warm-up is bounded by one forward pass (~3–5 s); acceptable for a persistent server process |

---

## 11. Implementation Order and Milestones

### Milestone 1 — Phase 1: Code Cleanup
**Deliverables:**
- `sam_segmenter.py` deleted
- Dead imports and globals removed from `fill_estimator_api.py`
- `point_line_signed_distance` removed from `fill_interpolator.py`
- `.env.example` updated
- Regression test: `/predict` response identical before/after for a reference image

### Milestone 2 — Phase 2: Inference Speed
**Deliverables:**
- Warm-up inference in `lifespan` startup
- SAM embedding LRU cache
- Async file I/O via `run_in_executor`
- Colour conversion refactor
- ONNX session options tuned
- `SAM_FP16` env var (default off)
- Latency benchmarked and documented

### Milestone 3 — Phase 3: Anomaly Integration
**Deliverables:**
- `AnomalyClassifierBase` protocol + `YoloxAnomalyClassifier` implementation
- `AnomalyRegistry` with startup registration
- Pipeline wired: anomaly runs post-YOLO on cropped region
- `anomaly_results` in `PredictionResponse` and summary JSON
- `ANOMALY_ENABLED`, `ANOMALY_MODEL_PATH`, `ANOMALY_CONF_THR` env vars
- Integration test: response contains `anomaly_results` for a reference image

### Milestone 4 — Phase 4: Enriched Input/Output
**Deliverables:**
- `barcode` and `container_type` optional form fields on `/predict`
- Filename sanitisation utility
- Output files use new naming convention
- Summary JSON includes barcode and container_type
- `PredictionResponse` echoes both fields
- Backward compatibility test: existing caller without new fields receives valid response

---

*Document version: 1.0 — 2026-05-20*
