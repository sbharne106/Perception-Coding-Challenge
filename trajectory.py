"""Estimate an ego-vehicle trajectory from a fixed traffic-light landmark.

Camera coordinates are (+X forward, +Y right, +Z up).  The output world frame
uses (+X from the initial car position toward the light, +Y left, +Z up), with
the ground point below the traffic light at the origin.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


@dataclass(frozen=True)
class LightBox:
    frame_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def is_valid(self) -> bool:
        return self.x_max > self.x_min and self.y_max > self.y_min


def read_bboxes(path: Path) -> list[LightBox]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        aliases = {
            "frame_id": ("frame_id", "frame"),
            "x_min": ("x_min", "x1"),
            "y_min": ("y_min", "y1"),
            "x_max": ("x_max", "x2"),
            "y_max": ("y_max", "y2"),
        }
        fields = set(reader.fieldnames or [])
        columns = {
            canonical: next((candidate for candidate in choices if candidate in fields), None)
            for canonical, choices in aliases.items()
        }
        if any(value is None for value in columns.values()):
            raise ValueError(
                f"{path} must use either frame_id,x_min,y_min,x_max,y_max "
                "or frame,x1,y1,x2,y2"
            )
        boxes = [
            LightBox(
                str(row[columns["frame_id"]]).strip(),
                float(row[columns["x_min"]]),
                float(row[columns["y_min"]]),
                float(row[columns["x_max"]]),
                float(row[columns["y_max"]]),
            )
            for row in reader
        ]
    if not boxes:
        raise ValueError(f"No bounding boxes found in {path}")
    return boxes


def frame_number(frame_id: str) -> int:
    matches = re.findall(r"\d+", str(frame_id))
    if not matches:
        raise ValueError(f"Could not find a frame number in {frame_id!r}")
    return int(matches[-1])


def xyz_path(xyz_dir: Path, frame_id: str) -> Path:
    number = frame_number(frame_id)
    candidates = [
        xyz_dir / f"frame_{number:04d}.npz",
        xyz_dir / f"frame_{number:05d}.npz",
        xyz_dir / f"frame_{number:06d}.npz",
        xyz_dir / f"{frame_id}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Last resort: match by numeric value, independent of zero padding.
    for candidate in xyz_dir.glob("*.npz"):
        try:
            if frame_number(candidate.stem) == number:
                return candidate
        except ValueError:
            pass
    raise FileNotFoundError(f"No XYZ file found for frame {frame_id!r} in {xyz_dir}")


def robust_point(points: np.ndarray, u: float, v: float, radius: int = 4) -> np.ndarray:
    """Return a robust 3-D estimate from a square patch around (u, v)."""
    if points.ndim != 3 or points.shape[2] < 3:
        raise ValueError(f"Expected XYZ shape (H, W, >=3), got {points.shape}")
    # The released files have a fourth, zero-filled channel. Only XYZ is used.
    points = points[:, :, :3]
    height, width, _ = points.shape
    uc, vc = int(round(u)), int(round(v))
    x0, x1 = max(0, uc - radius), min(width, uc + radius + 1)
    y0, y1 = max(0, vc - radius), min(height, vc + radius + 1)
    patch = points[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    valid = np.isfinite(patch).all(axis=1) & (np.linalg.norm(patch, axis=1) > 1e-6)
    patch = patch[valid]
    if len(patch) == 0:
        return np.full(3, np.nan)

    # Reject points far from the patch's dominant depth cluster.
    ranges = np.linalg.norm(patch, axis=1)
    median_range = np.median(ranges)
    mad = np.median(np.abs(ranges - median_range))
    tolerance = max(0.35, 3.5 * 1.4826 * mad)
    inliers = patch[np.abs(ranges - median_range) <= tolerance]
    return np.median(inliers if len(inliers) else patch, axis=0)


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    result = values.astype(float).copy()
    indices = np.arange(len(result))
    for column in range(result.shape[1]):
        valid = np.isfinite(result[:, column])
        if valid.sum() == 0:
            raise ValueError("All traffic-light depth estimates are invalid")
        result[:, column] = np.interp(indices, indices[valid], result[valid, column])
    return result


def hampel_filter(values: np.ndarray, half_window: int = 3, threshold: float = 3.5) -> np.ndarray:
    result = values.copy()
    for column in range(values.shape[1]):
        source = values[:, column]
        for i in range(len(source)):
            lo, hi = max(0, i - half_window), min(len(source), i + half_window + 1)
            window = source[lo:hi]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            scale = 1.4826 * mad
            if scale > 1e-9 and abs(source[i] - median) > threshold * scale:
                result[i, column] = median
    return result


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= 1:
        return values.copy()
    pad = window // 2
    kernel = np.ones(window) / window
    return np.column_stack(
        [np.convolve(np.pad(values[:, c], pad, mode="edge"), kernel, mode="valid") for c in range(values.shape[1])]
    )


def camera_landmarks_to_world_car(camera_points: np.ndarray) -> np.ndarray:
    """Convert fixed-landmark observations into car positions in the world XY plane.

    A single landmark does not reveal camera yaw independently, so this solution
    assumes camera heading is approximately constant over the short sequence.
    """
    # Camera +Y points right, while world +Y points left.
    landmark_xy = np.column_stack((camera_points[:, 0], -camera_points[:, 1]))
    initial_angle = np.arctan2(landmark_xy[0, 1], landmark_xy[0, 0])
    c, s = np.cos(-initial_angle), np.sin(-initial_angle)
    rotation = np.array([[c, -s], [s, c]])
    landmark_world = landmark_xy @ rotation.T
    return -landmark_world  # the light is at world origin


def normalize_camera_lateral_axis(
    camera_points: np.ndarray, boxes: list[LightBox], image_width: int
) -> tuple[np.ndarray, str]:
    """Normalize raw lateral coordinates to camera +Y right.

    The written challenge specifies +Y right, but the released depth maps use
    positive Y for pixels left of image center. The sign is inferred from image
    geometry so either convention is handled without a dataset-specific flag.
    """
    pixel_offsets, lateral_values = [], []
    for box, point in zip(boxes, camera_points):
        if box.is_valid and np.isfinite(point[1]):
            pixel_offsets.append(box.center[0] - image_width / 2.0)
            lateral_values.append(point[1])
    agreement = np.median(np.asarray(pixel_offsets) * np.asarray(lateral_values))
    result = camera_points.copy()
    if agreement < 0:
        result[:, 1] *= -1.0
        return result, "raw +Y left (automatically converted to +Y right)"
    return result, "raw +Y right"


def extract_trajectory(dataset: Path, patch_radius: int = 4, smooth_window: int = 15):
    csv_candidates = [dataset / "bboxes_light.csv", dataset / "bbox_light.csv"]
    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Expected bboxes_light.csv or bbox_light.csv in {dataset}"
        )
    boxes = read_bboxes(csv_path)
    estimates, valid_depth = [], []
    image_width = None
    for box in boxes:
        if not box.is_valid:
            estimates.append(np.full(3, np.nan))
            valid_depth.append(False)
            continue
        with np.load(xyz_path(dataset / "xyz", box.frame_id)) as archive:
            key = "points" if "points" in archive else "xyz" if "xyz" in archive else None
            if key is None:
                raise KeyError(
                    f"XYZ archive for frame {box.frame_id} has neither 'points' nor 'xyz' key"
                )
            points = archive[key]
            image_width = points.shape[1]
            estimate = robust_point(points, *box.center, radius=patch_radius)
        estimates.append(estimate)
        valid_depth.append(bool(np.isfinite(estimate).all()))

    camera_points = interpolate_missing(np.asarray(estimates))
    camera_points = hampel_filter(camera_points)
    camera_points = moving_average(camera_points, smooth_window)
    camera_points, lateral_convention = normalize_camera_lateral_axis(
        camera_points, boxes, image_width
    )
    trajectory = camera_landmarks_to_world_car(camera_points)
    print(f"Lateral-axis check: {lateral_convention}")
    return boxes, camera_points, trajectory, np.asarray(valid_depth)


def plot_trajectory(trajectory: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#1565c0", linewidth=1.5, alpha=0.75)
    scatter = ax.scatter(
        trajectory[:, 0], trajectory[:, 1], c=np.arange(len(trajectory)),
        cmap="viridis", s=18, label="Ego vehicle"
    )
    ax.scatter(0, 0, marker="*", s=220, color="#ef6c00", edgecolor="black", label="Traffic light")
    ax.scatter(*trajectory[0], marker="s", s=70, color="#2e7d32", label="Start")
    ax.scatter(*trajectory[-1], marker="X", s=90, color="#c62828", label="End")
    ax.set(title="Ego-Vehicle Trajectory in Ground Frame", xlabel="X forward (m)", ylabel="Y left (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Frame index")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def animate_trajectory(trajectory: np.ndarray, output: Path, fps: int = 20) -> None:
    margin = max(1.0, 0.08 * np.ptp(trajectory, axis=0).max())
    xlim = (min(trajectory[:, 0].min(), 0) - margin, max(trajectory[:, 0].max(), 0) + margin)
    ylim = (min(trajectory[:, 1].min(), 0) - margin, max(trajectory[:, 1].max(), 0) + margin)
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.set(title="Ego-Vehicle Trajectory in Ground Frame", xlabel="X forward (m)", ylabel="Y left (m)")
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.scatter(0, 0, marker="*", s=220, color="#ef6c00", edgecolor="black", label="Traffic light")
    line, = ax.plot([], [], color="#1565c0", linewidth=2, label="Trajectory")
    car = ax.scatter([], [], marker="^", s=100, color="#c62828", label="Ego vehicle")
    ax.legend(loc="best")

    def update(index: int):
        path = trajectory[: index + 1]
        line.set_data(path[:, 0], path[:, 1])
        car.set_offsets(path[-1:])
        return line, car

    animation = FuncAnimation(fig, update, frames=len(trajectory), interval=1000 / fps, blit=True)
    animation.save(output, writer=FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)


def write_csv(boxes, camera_points, trajectory, valid_depth, output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "light_x_cam_m", "light_y_cam_m", "light_z_cam_m", "x_m", "y_m", "original_depth_valid"])
        for box, point, position, valid in zip(boxes, camera_points, trajectory, valid_depth):
            writer.writerow([box.frame_id, *point, *position, int(valid)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    boxes, camera_points, trajectory, valid = extract_trajectory(
        args.dataset, patch_radius=args.patch_radius, smooth_window=args.smooth_window
    )
    plot_trajectory(trajectory, args.output / "trajectory.png")
    animate_trajectory(trajectory, args.output / "trajectory.mp4", fps=args.fps)
    write_csv(boxes, camera_points, trajectory, valid, args.output / "trajectory.csv")
    print(f"Processed {len(boxes)} frames; outputs written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
