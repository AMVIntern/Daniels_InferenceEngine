# visualizer.py

import cv2
import numpy as np


# ============================================================
# 1.  OVERLAY SAM SEGMENTATION MASK
# ============================================================

def overlay_mask(base_img, mask, color=(0,255,0), alpha=0.35):
    """
    Draw SAM mask as a translucent overlay.
    base_img : HxWx3 RGB image
    mask     : HxW uint8 binary mask (0 or 1 or 255)
    """
    out = base_img.copy()

    # Normalize: convert 255 mask to 1
    mask_bin = (mask > 0).astype(np.uint8)

    mask_color = np.zeros_like(out)
    mask_color[mask_bin == 1] = color

    cv2.addWeighted(mask_color, alpha, out, 1 - alpha, 0, out)
    return out



# ============================================================
# 2.  VERTICAL TEXT RENDERER
# ============================================================

def put_text_vertical(img, text, org, font_scale=0.55, thickness=2,
                      color=(255,255,255)):
    """
    Draw vertical text (rotated 90° CCW).
    org = (x, y) top-left corner of placement.
    """
    if org is None:
        return

    x, y = org

    (w, h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # Protect against negative dimensions
    if w <= 0 or h <= 0:
        return

    canvas = np.zeros((h + baseline + 5, w + 5, 3), dtype=np.uint8)

    # Draw horizontal text first
    cv2.putText(
        canvas, text, (0, h),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA
    )

    # Rotate to vertical
    canvas_rot = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
    H, W = canvas_rot.shape[:2]

    # Ensure placement inside image
    if y < 0: 
        y = 0
    if x < 0:
        x = 0
    if y + H > img.shape[0] or x + W > img.shape[1]:
        return  # skip out-of-bounds labels

    img[y:y+H, x:x+W] = canvas_rot



# ============================================================
# 3.  DRAW FILL LEVEL REFERENCE LINES
# ============================================================

def draw_fill_lines(img, ref_lines, min_label_gap=20, label_mode="vertical"):
    """
    Draw fill-reference lines.
    Label is placed ABOVE the top endpoint of the line.
    """
    out = img.copy()
    last_label_x = None

    for r in ref_lines:
        (x1, y1) = r["p1"]
        (x2, y2) = r["p2"]
        fill     = r["fill"]

        # Draw line
        cv2.line(out, (x1, y1), (x2, y2), (0,255,255), 2)

        # Determine top endpoint
        if y1 < y2:
            x_top, y_top = x1, y1
        else:
            x_top, y_top = x2, y2

        # Declutter horizontally
        if last_label_x is not None and abs(x_top - last_label_x) < min_label_gap:
            continue

        last_label_x = x_top

        # Offset label higher
        vertical_offset = 60
        label_x = x_top - 18
        label_y = y_top - vertical_offset

        txt = f"{fill:.1f}%"

        # Ensure label placement stays inside image
        if label_y < 0:
            label_y = 0

        if label_mode == "vertical":
            put_text_vertical(out, txt, (label_x, label_y))
        else:
            cv2.putText(
                out, txt, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255,255,255), 2, cv2.LINE_AA
            )

    return out



# ============================================================
# 4.  HIT POINT DRAWING (WHITE + REPRESENTATIVE)
# ============================================================

def draw_hit_points(img, hit_points, rep_point=None):
    """
    Draw raw hit points + representative point.
    Handles None values safely.
    """
    out = img.copy()

    # Raw hits
    for pt in hit_points:
        if pt is None:
            continue
        x, y = pt
        if x is None or y is None:
            continue

        cv2.circle(out, (int(x), int(y)), 4, (255,255,255), -1)

    # Representative point
    if rep_point is not None:
        hx, hy = rep_point
        if hx is not None and hy is not None:
            cv2.circle(out, (int(hx), int(hy)), 7, (0,0,255), -1)
            cv2.putText(
                out, "REP",
                (int(hx) + 5, int(hy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,0,255), 2, cv2.LINE_AA
            )

    return out
