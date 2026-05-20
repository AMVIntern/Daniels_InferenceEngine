# fill_interpolator.py

import numpy as np

def estimate_fill_from_hit(hx, hy, ref_lines):

    # Step 1: compute x_mid for each fill line
    x_positions = []
    for r in ref_lines:
        (x1, y1) = r["p1"]
        (x2, y2) = r["p2"]
        x_mid = (x1 + x2) / 2.0
        x_positions.append({"fill": r["fill"], "x": x_mid})

    # Step 2: sort left → right
    x_positions.sort(key=lambda a: a["x"])

    # Step 3: find bracket
    for i in range(len(x_positions)-1):
        left = x_positions[i]
        right = x_positions[i+1]

        if left["x"] <= hx <= right["x"]:
            t = (hx - left["x"]) / (right["x"] - left["x"])
            return left["fill"] + t * (right["fill"] - left["fill"])

    # Outside ranges → clamp
    if hx < x_positions[0]["x"]:
        return x_positions[0]["fill"]

    if hx > x_positions[-1]["x"]:
        return x_positions[-1]["fill"]

    return None


