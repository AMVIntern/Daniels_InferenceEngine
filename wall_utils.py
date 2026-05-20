# wall_utils.py

import csv
import math
import numpy as np

def parse_walls_csv(csv_path):
    """
    Expects CSV that contains at least the two wall lines:
    'bottom left' and 'top left'.
    """
    walls = {}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue

            label = row[0].strip().lower()
            try:
                col1, col2, row1, row2 = map(float, row[1:5])
                walls[label] = {
                    "start": (int(col1), int(row1)),   # correct (x, y)
                    "end":   (int(col2), int(row2))    # correct (x, y)
                }
            except:
                continue

    return walls


def generate_parallel_lines(walls, num_lines):
    """
    Returns list of (left_point, right_point)
    using interpolation between top/bottom wall lines.
    """
    bottom = walls.get("bottom right")
    top    = walls.get("top right")

    if bottom is None or top is None:
        raise ValueError("Wall lines 'bottom right' and 'top right' missing.")

    left_top  = top["start"]
    left_bot  = bottom["start"]
    right_top = top["end"]
    right_bot = bottom["end"]

    lines = []

    for i in range(1, num_lines + 1):
        t = i / (num_lines + 1)

        lx = int(left_top[0]  + t*(left_bot[0]  - left_top[0]))
        ly = int(left_top[1]  + t*(left_bot[1]  - left_top[1]))

        rx = int(right_top[0] + t*(right_bot[0] - right_top[0]))
        ry = int(right_top[1] + t*(right_bot[1] - right_top[1]))

        lines.append(((lx, ly), (rx, ry)))

    return lines


def march_mask(mask, p0, p1, step=1, min_run=3):
    """
    March along line p0->p1 to find first hit in mask==1.
    Returns (x,y) or None.
    """
    x0,y0 = p0
    x1,y1 = p1

    dx = x1 - x0
    dy = y1 - y0
    L  = math.hypot(dx, dy)
    steps = int(L / step)

    run = 0
    for i in range(steps+1):
        t = i / steps if steps > 0 else 0
        x = int(x0 + t*dx)
        y = int(y0 + t*dy)

        # Out of bounds check
        if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
            run = 0
            continue

        if mask[y, x] == 1:
            run += 1
            if run >= min_run:
                return (x, y)
        else:
            run = 0

    return None
