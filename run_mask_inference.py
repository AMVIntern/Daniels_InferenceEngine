#!/usr/bin/env python3
"""
Batch fill-level inference directly from pre-generated mask images.

This script skips SAM model inference and treats each input image as a mask-like
image where non-zero pixels are considered foreground.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from fill_interpolator import estimate_fill_from_hit
from visualizer import draw_fill_lines, draw_hit_points, overlay_mask
from wall_utils import generate_parallel_lines, march_mask


def normalize_left_to_right(
    start: Tuple[int, int], end: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return endpoints ordered from smaller x to larger x."""
    if start[0] <= end[0]:
        return start, end
    return end, start


def parse_walls_csv(csv_path: str) -> Dict[str, Dict[str, Tuple[int, int]]]:
    walls: Dict[str, Dict[str, Tuple[int, int]]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            label = row[0].strip().lower()
            normalized = label.replace("_", " ").replace("-", " ")
            normalized = " ".join(normalized.split())

            # New expected labels in CSV:
            # - Bin_Top_Right
            # - Bin_Bottom_Right
            if normalized == "bin top right":
                canonical = "top left"
            elif normalized == "bin bottom right":
                canonical = "bottom left"
            else:
                continue
            try:
                x1, y1, x2, y2 = map(float, row[1:5])
            except ValueError:
                continue
            start_pt, end_pt = normalize_left_to_right(
                (int(x1), int(y1)), (int(x2), int(y2))
            )
            walls[canonical] = {
                "start": start_pt,
                "end": end_pt,
            }
    if "bottom left" not in walls or "top left" not in walls:
        raise ValueError(
            "WALL CSV must include 'Bin_Top_Right' and 'Bin_Bottom_Right' rows."
        )
    return walls


def load_fill_lines_csv(csv_path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
        has_shifted_header = "co2" in fieldnames and "col2" not in fieldnames
        has_xyxy_header = fieldnames[:5] == ["Fill", "col1", "col2", "row1", "row2"]
        for row in reader:
            if has_shifted_header or has_xyxy_header:
                # Observed format:
                # Fill,col1,co2,row1,row2  OR  Fill,col1,col2,row1,row2
                # where values map to:
                # col1, row1, col2, row2
                x1 = int(float(row["col1"]))
                y1 = int(float(row["co2"])) if has_shifted_header else int(float(row["col2"]))
                x2 = int(float(row["row1"]))
                y2 = int(float(row["row2"]))
            else:
                # Normal format:
                # Fill,col1,row1,col2,row2
                col2_key = "col2" if "col2" in row else "co2"
                if col2_key not in row:
                    raise KeyError("Fill CSV must contain either 'col2' or 'co2'.")
                x1 = int(float(row["col1"]))
                y1 = int(float(row["row1"]))
                x2 = int(float(row[col2_key]))
                y2 = int(float(row["row2"]))

            rows.append({"fill": float(row["Fill"]), "p1": (x1, y1), "p2": (x2, y2)})
    rows.sort(key=lambda r: r["fill"], reverse=True)
    return rows


def mask_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask from either:
    1) true binary mask image, or
    2) SAM overlay-style image (green tint over foreground).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Path A: image already looks like a binary mask.
    near_binary = np.mean((gray < 15) | (gray > 240)) > 0.92
    if near_binary:
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        # Path B: overlay image - extract strong green foreground.
        b = img_bgr[:, :, 0].astype(np.int16)
        g = img_bgr[:, :, 1].astype(np.int16)
        r = img_bgr[:, :, 2].astype(np.int16)

        # Green dominance test.
        green_excess = g - np.maximum(r, b)
        green_dom = (green_excess > 22).astype(np.uint8) * 255

        # HSV green range test.
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hsv_green = cv2.inRange(hsv, (35, 40, 25), (95, 255, 255))

        mask = cv2.bitwise_or(green_dom, hsv_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_component(mask: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        empty = np.zeros_like(mask)
        return empty, 0, (empty > 0).astype(np.uint8)

    areas = stats[:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas[1:]))
    largest_area = int(areas[largest_label])
    mask_clean = (labels == largest_label).astype(np.uint8) * 255
    mask_bin = (mask_clean > 0).astype(np.uint8)
    return mask_clean, largest_area, mask_bin


def collect_hit_points(mask_bin, parallel_lines, step_pix, min_run):
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
    if not hit_points:
        return None, None, None
    xs = [p[0] for p in hit_points]
    ys = [p[1] for p in hit_points]
    hx_rep = float(np.median(xs))
    hy_rep = float(np.median(ys))
    avg_fill = estimate_fill_from_hit(hx_rep, hy_rep, ref_lines)
    return hx_rep, hy_rep, avg_fill


def build_scanline_debug_overlay(
    base_rgb: np.ndarray, walls: Dict[str, Dict[str, Tuple[int, int]]], num_lines: int
) -> np.ndarray:
    """
    Draw wall boundaries + generated parallel scan lines + direction arrows.
    Arrow direction indicates marching direction: right -> left.
    """
    out = base_rgb.copy()
    parallel_lines = list(generate_parallel_lines(walls, num_lines))

    # Draw top/bottom wall lines in magenta for quick visual validation.
    for key in ("top left", "bottom left"):
        line = walls.get(key)
        if line:
            cv2.line(out, line["start"], line["end"], (255, 0, 255), 3)

    # Draw all generated scan lines + sparse arrows.
    for idx, pair in enumerate(parallel_lines):
        p_left, p_right = pair
        cv2.line(out, p_left, p_right, (0, 200, 255), 1)
        if idx % 8 == 0:
            cv2.arrowedLine(
                out,
                p_right,
                p_left,
                (255, 255, 255),
                2,
                tipLength=0.15,
            )

    cv2.putText(
        out,
        "DEBUG: scan lines and march direction (right->left)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return out


def process_one_image(
    img_path: str,
    walls,
    ref_lines,
    output_dir: str,
    num_lines: int,
    step_pix: int,
    min_run: int,
    area_threshold: int,
    save_debug: bool,
):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")

    mask = mask_from_image(img_bgr)
    mask_clean, largest_area, mask_bin = largest_component(mask)

    parallel_lines = list(generate_parallel_lines(walls, num_lines))
    hit_points = collect_hit_points(mask_bin, parallel_lines, step_pix, min_run)

    if largest_area < area_threshold:
        avg_fill = 0.0
        forced_zero = True
        hx_rep, hy_rep = None, None
    else:
        hx_rep, hy_rep, avg_fill = compute_fill_level(hit_points, ref_lines)
        forced_zero = False

    # Visual overlay on original image for debugging/inspection
    base_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = overlay_mask(base_rgb, mask_clean)
    overlay = draw_fill_lines(overlay, ref_lines, label_mode="vertical")
    overlay = draw_hit_points(
        overlay, hit_points, rep_point=(hx_rep, hy_rep) if hx_rep is not None else None
    )

    if hx_rep is not None and hy_rep is not None and avg_fill is not None:
        cv2.circle(overlay, (int(hx_rep), int(hy_rep)), 6, (255, 0, 0), -1)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    label = f"Fill Level: {avg_fill:.2f}%" if avg_fill is not None else "No valid hits"
    cv2.putText(
        overlay_bgr,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "summaries"), exist_ok=True)

    stem = os.path.splitext(os.path.basename(img_path))[0]
    mask_out = os.path.join(output_dir, "masks", f"{stem}_mask.png")
    overlay_out = os.path.join(output_dir, "overlays", f"{stem}_overlay.jpg")
    summary_out = os.path.join(output_dir, "summaries", f"{stem}_summary.json")

    Image.fromarray(mask_clean).save(mask_out)
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(overlay_out, quality=95)

    if save_debug:
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)
        debug_img = build_scanline_debug_overlay(base_rgb, walls, num_lines)
        debug_out = os.path.join(output_dir, "debug", f"{stem}_scanlines_debug.jpg")
        Image.fromarray(debug_img).save(debug_out, quality=95)

    summary = {
        "image": os.path.basename(img_path),
        "num_hits": len(hit_points),
        "largest_area": int(largest_area),
        "avg_fill": float(avg_fill) if avg_fill is not None else None,
        "forced_zero": forced_zero,
        "rep_point": [hx_rep, hy_rep] if hx_rep is not None else None,
        "mask_path": mask_out,
        "overlay_path": overlay_out,
    }
    with open(summary_out, "w") as jf:
        json.dump(summary, jf, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run fill-level inference on mask images.")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(os.path.dirname(__file__), "input_images"),
        help="Directory containing input mask images.",
    )
    parser.add_argument(
        "--walls-csv",
        default=os.path.join(os.path.dirname(__file__), "images", "Biggest", "64_wall.csv"),
        help="Path to wall lines CSV.",
    )
    parser.add_argument(
        "--fill-lines-csv",
        default=os.path.join(
            os.path.dirname(__file__),
            "Refrence_fill_levels",
            "fill_level_lines_bin_64_series_new_27MAR26.csv",
        ),
        help="Path to fill reference lines CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "RESULT", "mask_inference_results"),
        help="Directory for results.",
    )
    parser.add_argument("--num-lines", type=int, default=100)
    parser.add_argument("--step-pix", type=int, default=1)
    parser.add_argument("--min-run", type=int, default=3)
    parser.add_argument("--area-threshold", type=int, default=30000)
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug overlays with generated scan lines and direction arrows.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    walls = parse_walls_csv(args.walls_csv)
    ref_lines = load_fill_lines_csv(args.fill_lines_csv)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [
        os.path.join(args.input_dir, n)
        for n in sorted(os.listdir(args.input_dir))
        if os.path.splitext(n.lower())[1] in image_exts
    ]
    if not files:
        raise ValueError(f"No image files found in: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(files)} image(s) in {args.input_dir}")

    rows = []
    for i, img_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing {os.path.basename(img_path)}")
        try:
            summary = process_one_image(
                img_path=img_path,
                walls=walls,
                ref_lines=ref_lines,
                output_dir=args.output_dir,
                num_lines=args.num_lines,
                step_pix=args.step_pix,
                min_run=args.min_run,
                area_threshold=args.area_threshold,
                save_debug=args.save_debug,
            )
            rows.append(
                {
                    "image": summary["image"],
                    "avg_fill": summary["avg_fill"],
                    "num_hits": summary["num_hits"],
                    "largest_area": summary["largest_area"],
                    "forced_zero": summary["forced_zero"],
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append(
                {
                    "image": os.path.basename(img_path),
                    "avg_fill": None,
                    "num_hits": 0,
                    "largest_area": 0,
                    "forced_zero": False,
                    "error": str(e),
                }
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_csv = os.path.join(args.output_dir, f"batch_results_{ts}.csv")
    with open(batch_csv, "w", newline="") as f:
        fieldnames = ["image", "avg_fill", "num_hits", "largest_area", "forced_zero", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if "error" not in r:
                r["error"] = ""
            writer.writerow(r)

    print(f"\nDone. Results saved in: {args.output_dir}")
    print(f"Batch CSV: {batch_csv}")


if __name__ == "__main__":
    main()

