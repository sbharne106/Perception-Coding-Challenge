"""Create a tiny deterministic dataset for end-to-end smoke testing."""

import csv
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path("synthetic_dataset")
    (root / "xyz").mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, 31):
        # A fixed light appears closer as the synthetic ego car moves forward.
        x = 15.0 - 0.25 * index
        y = 0.5 * np.sin(index / 8)
        points = np.full((40, 60, 3), [x, y, 4.5], dtype=np.float32)
        points[20, 30] = np.nan
        points[19, 29] = [500, 500, 500]
        np.savez_compressed(root / "xyz" / f"frame_{index:04d}.npz", points=points)
        rows.append([index, 27, 17, 33, 23])
    with (root / "bboxes_light.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
