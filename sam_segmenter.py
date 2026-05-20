# sam_segmenter.py

import torch
import numpy as np
from transformers import SamProcessor, SamModel
import torchvision.transforms.functional as TF
from PIL import Image

class SamSegmenter:
    def __init__(self, model_path, device=None):
        print("Loading SAM model...")

        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.model = SamModel.from_pretrained("facebook/sam-vit-base")

        # Try to load fine-tuned weights
        try:
            state = torch.load(model_path, map_location="cpu")
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                if "model_state_dict" in state:
                    state = state["model_state_dict"]

            self.model.load_state_dict(state, strict=False)
            print(f"Loaded fine-tuned SAM weights: {model_path}")
        except Exception as e:
            print(f"WARNING: Using base SAM model (weights load failed): {e}")

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model.to(device).eval()

    def predict_mask(self, image: Image.Image, grid_size=14, threshold=0.5):
        w, h = image.size

        # Generate evenly spaced grid points
        xs = np.linspace(0, w - 1, grid_size, dtype=int)
        ys = np.linspace(0, h - 1, grid_size, dtype=int)
        points = [[int(x), int(y)] for y in ys for x in xs]
        input_points = [points]

        with torch.no_grad():
            inputs = self.processor(image, input_points=input_points,
                                    return_tensors="pt").to(self.device)

            outputs = self.model(**inputs, multimask_output=False)

            pm = outputs.pred_masks.squeeze()

            # Handle different shape conventions
            if pm.ndim == 3:
                prob = torch.sigmoid(pm[0]).cpu().numpy()
            else:
                prob = torch.sigmoid(pm).cpu().numpy()

        # Resize to original resolution
        prob_resized = TF.resize(
            torch.from_numpy(prob)[None, None, ...],
            (h, w),
            interpolation=TF.InterpolationMode.BILINEAR
        ).squeeze().numpy()

        mask = (prob_resized > threshold).astype(np.uint8)
        return mask
