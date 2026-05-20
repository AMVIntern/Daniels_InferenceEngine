# sam_segmenter_large.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class SamSegmenter:
    """
    SAM ViT-H segmenter using segment_anything library.
    Loads fine-tuned weights directly from checkpoint.
    Uses bbox + center point prompting with two-pass refinement
    (mirrors the training inference approach).
    """

    def __init__(self, checkpoint_path, model_type="vit_h", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Loading SAM {model_type} from: {checkpoint_path}")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device).eval()
        self.transform = ResizeLongestSide(1024)

        print(f"SAM {model_type} loaded on {self.device}")

    def predict_mask(self, image, bbox, threshold=0.3):
        """
        Predict segmentation mask using bbox + center point prompts
        with two-pass refinement.

        image   : PIL Image (RGB)
        bbox    : [x1, y1, x2, y2]  — bin region bounding box
        threshold: mask confidence threshold (default 0.3, matches training)
        """
        image_rgb = np.array(image)
        img_h, img_w = image_rgb.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Prepare image tensor
        inp = self.transform.apply_image(image_rgb)
        t = torch.as_tensor(inp, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = self.sam.preprocess(t)

        # Center point of bbox as point prompt
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pts_np = self.transform.apply_coords(
            np.array([[cx, cy]], dtype=np.float32), (img_h, img_w)
        )
        pts_t = torch.tensor(pts_np[None]).float().to(self.device)
        pl_t  = torch.ones(1, 1, dtype=torch.long).to(self.device)

        # Transform bbox
        box = self.transform.apply_boxes(
            np.array([x1, y1, x2, y2], dtype=np.float32)[None, :], (img_h, img_w)
        )
        bt = torch.tensor(box).float().to(self.device)

        with torch.no_grad():
            emb = self.sam.image_encoder(t)

            # Pass 1 — multi-mask output, take union as refined prompt
            se, de = self.sam.prompt_encoder(points=(pts_t, pl_t), boxes=bt, masks=None)
            lrm, _ = self.sam.mask_decoder(
                image_embeddings=emb,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=True
            )
            union_lrm = lrm.max(dim=1, keepdim=True).values

            # Pass 2 — single-mask output using union as mask prompt
            se2, de2 = self.sam.prompt_encoder(points=(pts_t, pl_t), boxes=bt, masks=union_lrm)
            lrm2, _ = self.sam.mask_decoder(
                image_embeddings=emb,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se2,
                dense_prompt_embeddings=de2,
                multimask_output=False
            )

            m    = torch.sigmoid(F.interpolate(lrm2, size=(img_h, img_w),
                                               mode='bilinear', align_corners=False))
            mask = (m[0, 0].cpu().numpy() > threshold).astype(np.uint8)

        # Clip to bbox region only
        clipped = np.zeros_like(mask)
        clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        return self._clean_mask(clipped)

    def _clean_mask(self, mask):
        """Remove noise, keep largest component, fill holes — matches training clean_mask."""
        noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_k)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return mask

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        return mask
