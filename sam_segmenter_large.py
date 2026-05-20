# sam_segmenter_large.py

import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# Read once at import time so SamSegmenter picks them up on construction.
# SAM_CACHE_SIZE : number of image embeddings to cache (0 = disabled).
# SAM_FP16       : run encoder in half precision on CUDA (default off).
_CACHE_SIZE = int(os.getenv("SAM_CACHE_SIZE", "4"))
_FP16       = os.getenv("SAM_FP16", "false").lower() == "true"


class _EmbedCache:
    """Bounded LRU cache for SAM image embeddings, keyed by image hash."""

    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: str):
        if self._maxsize == 0 or key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value) -> None:
        if self._maxsize == 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


class SamSegmenter:
    """
    SAM ViT-H segmenter using segment_anything library.
    Loads fine-tuned weights directly from checkpoint.
    Uses bbox + center point prompting with two-pass refinement
    (mirrors the training inference approach).

    Env vars:
      SAM_CACHE_SIZE  — LRU embedding cache size, default 4 (0 = disabled).
      SAM_FP16        — run in half precision on CUDA, default false.
    """

    def __init__(self, checkpoint_path, model_type="vit_h", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.fp16   = _FP16 and device == "cuda"

        print(f"Loading SAM {model_type} from: {checkpoint_path}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        if self.fp16:
            self.sam = self.sam.half()
            print("  SAM running in FP16 mode")

        self.sam.to(self.device).eval()
        self.transform  = ResizeLongestSide(1024)
        self._emb_cache = _EmbedCache(_CACHE_SIZE)

        print(f"SAM {model_type} loaded on {self.device}  "
              f"(embed cache size={_CACHE_SIZE})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_mask(self, image, bbox, threshold=0.3, image_hash: str = None):
        """
        Predict segmentation mask using bbox + center point prompts
        with two-pass refinement.

        image       : PIL Image (RGB)
        bbox        : [x1, y1, x2, y2]  — bin region bounding box
        threshold   : mask confidence threshold (default 0.3, matches training)
        image_hash  : hex string (e.g. MD5 of raw image bytes); enables the
                      embedding cache so identical frames skip the encoder.
        """
        image_rgb = np.array(image)
        img_h, img_w = image_rgb.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # --- Image encoder (with optional LRU cache) ---
        emb = self._emb_cache.get(image_hash) if image_hash else None

        if emb is None:
            inp   = self.transform.apply_image(image_rgb)
            dtype = torch.float16 if self.fp16 else torch.float32
            t = (torch.as_tensor(inp, dtype=dtype)
                      .permute(2, 0, 1)
                      .unsqueeze(0)
                      .to(self.device))
            t = self.sam.preprocess(t)
            with torch.no_grad():
                emb = self.sam.image_encoder(t)
            if image_hash:
                self._emb_cache.put(image_hash, emb)
        else:
            print(f"  [SAM] Embedding cache hit ({image_hash[:8]}…, "
                  f"cache size={len(self._emb_cache)})")

        # --- Prompts ---
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pts_np = self.transform.apply_coords(
            np.array([[cx, cy]], dtype=np.float32), (img_h, img_w)
        )
        pts_t = torch.tensor(pts_np[None]).float().to(self.device)
        pl_t  = torch.ones(1, 1, dtype=torch.long).to(self.device)

        box = self.transform.apply_boxes(
            np.array([x1, y1, x2, y2], dtype=np.float32)[None, :], (img_h, img_w)
        )
        bt = torch.tensor(box).float().to(self.device)

        with torch.no_grad():
            # Pass 1 — multi-mask output, take union as refined mask prompt
            se, de = self.sam.prompt_encoder(points=(pts_t, pl_t), boxes=bt, masks=None)
            lrm, _ = self.sam.mask_decoder(
                image_embeddings=emb,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=True,
            )
            union_lrm = lrm.max(dim=1, keepdim=True).values

            # Pass 2 — single-mask output using union as mask prompt
            se2, de2 = self.sam.prompt_encoder(points=(pts_t, pl_t), boxes=bt, masks=union_lrm)
            lrm2, _ = self.sam.mask_decoder(
                image_embeddings=emb,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se2,
                dense_prompt_embeddings=de2,
                multimask_output=False,
            )

            m    = torch.sigmoid(F.interpolate(lrm2, size=(img_h, img_w),
                                               mode="bilinear", align_corners=False))
            mask = (m[0, 0].cpu().numpy() > threshold).astype(np.uint8)

        # Clip to bbox region only
        clipped = np.zeros_like(mask)
        clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        return self._clean_mask(clipped)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_mask(self, mask):
        """Remove noise, keep largest component, fill holes — matches training clean_mask."""
        noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_k)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return mask

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = (labels == largest).astype(np.uint8)

        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        return mask
