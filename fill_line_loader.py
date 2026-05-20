# fill_line_loader.py

import csv

def load_fill_level_lines(csv_path):
    rows = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:

            fill = float(row["Fill"])
            x1= int(row["col1"]);  y1 = int(row["row1"])
            x2 = int(row["col2"]);  y2 = int(row["row2"])

            rows.append({
                "fill": fill,
                "p1": (x1, y1),
                "p2": (x2, y2)
            })

    # SORT BY FILL PERCENTAGE — THIS IS THE CORRECT TRUE ORDER
    rows.sort(key=lambda r: r["fill"], reverse=True)
    return rows
