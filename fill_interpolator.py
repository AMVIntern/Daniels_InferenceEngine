# fill_interpolator.py

import numpy as np

def point_line_signed_distance(px, py, x1, y1, x2, y2):
    """Signed perpendicular distance from point P to line AB."""
    A = np.array([x1, y1], float)
    B = np.array([x2, y2], float)
    P = np.array([px, py], float)

    AB = B - A
    AP = P - A

    cross = AB[0]*AP[1] - AB[1]*AP[0]
    length = np.linalg.norm(AB)

    if length < 1e-6:
        return None

    return cross / length   # signed distance

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


