"""Part B: track the golf cart and render an enhanced ground-frame BEV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from trajectory import (
    LightBox,
    extract_trajectory,
    hampel_filter,
    interpolate_missing,
    moving_average,
    normalize_camera_lateral_axis,
    robust_point,
)


def track_cart(rgb_dir: Path, initial_box: tuple[int, int, int, int]):
    frames = sorted(rgb_dir.glob("left*.png"))
    if not frames:
        raise FileNotFoundError(f"No left*.png frames found in {rgb_dir}")
    tracker = cv2.TrackerCSRT_create()
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Could not read {frames[0]}")
    tracker.init(first, initial_box)
    boxes = [tuple(float(x) for x in initial_box)]
    valid = [True]
    for frame_path in frames[1:]:
        image = cv2.imread(str(frame_path))
        ok, box = tracker.update(image)
        boxes.append(tuple(float(x) for x in box) if ok else (np.nan,) * 4)
        valid.append(bool(ok))
    return frames, np.asarray(boxes), np.asarray(valid)


def sample_cart_depth(xyz_dir: Path, boxes: np.ndarray) -> np.ndarray:
    points = []
    for index, (x, y, width, height) in enumerate(boxes):
        if not np.isfinite([x, y, width, height]).all():
            points.append(np.full(3, np.nan))
            continue
        with np.load(xyz_dir / f"depth{index:06d}.npz") as archive:
            key = "xyz" if "xyz" in archive else "points"
            # The box center stays on the cart's rear body/seat across the sequence.
            points.append(robust_point(archive[key], x + width / 2, y + height / 2, radius=8))
    result = interpolate_missing(np.asarray(points))
    return moving_average(hampel_filter(result), window=15)


def traffic_light_colors(frames: list[Path], light_boxes: list[LightBox]) -> list[str]:
    colors = []
    previous = "red"
    for path, box in zip(frames, light_boxes):
        if not box.is_valid:
            colors.append(previous)
            continue
        image = cv2.imread(str(path))
        crop = image[
            int(box.y_min): int(box.y_max) + 1,
            int(box.x_min): int(box.x_max) + 1,
        ]
        if crop.size == 0:
            colors.append(previous)
            continue
        b, g, r = cv2.split(crop.astype(np.int16))
        red_score = int(((r > g + 35) & (r > b + 25) & (r > 120)).sum())
        green_score = int(((g > r + 25) & (g > 110)).sum())
        previous = "#e53935" if red_score >= green_score else "#00bfa5"
        colors.append(previous)
    return colors


def ground_rotation(light_points: np.ndarray) -> np.ndarray:
    light_local = np.column_stack((light_points[:, 0], -light_points[:, 1]))
    angle = np.arctan2(light_local[0, 1], light_local[0, 0])
    c, s = np.cos(-angle), np.sin(-angle)
    return np.array([[c, -s], [s, c]])


def cart_ground_positions(
    cart_points: np.ndarray, cart_boxes: np.ndarray, light_points: np.ndarray,
    light_boxes: list[LightBox], ego_positions: np.ndarray, image_width: int,
) -> np.ndarray:
    pseudo_boxes = [
        LightBox(str(i), x, y, x + w, y + h)
        for i, (x, y, w, h) in enumerate(cart_boxes)
    ]
    cart_normalized, _ = normalize_camera_lateral_axis(cart_points, pseudo_boxes, image_width)
    rotation = ground_rotation(light_points)
    cart_relative = np.column_stack((cart_normalized[:, 0], -cart_normalized[:, 1]))
    return ego_positions + cart_relative @ rotation.T


def save_tracking_csv(boxes: np.ndarray, valid: np.ndarray, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "width", "height", "tracking_valid"])
        for i, (box, ok) in enumerate(zip(boxes, valid)):
            writer.writerow([i, *box, int(ok)])


def save_quality_sheet(frames: list[Path], boxes: np.ndarray, output: Path) -> None:
    indices = np.linspace(0, len(frames) - 1, 9, dtype=int)
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
    for ax, index in zip(axes.flat, indices):
        image = cv2.cvtColor(cv2.imread(str(frames[index])), cv2.COLOR_BGR2RGB)
        x, y, width, height = boxes[index]
        ax.imshow(image)
        ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, color="red", linewidth=2))
        ax.set_title(f"Frame {index}")
        ax.axis("off")
    fig.suptitle("CSRT Golf-Cart Tracking Quality Check")
    fig.savefig(output, dpi=130)
    plt.close(fig)


def save_bev_plot(ego: np.ndarray, cart: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.plot(ego[:, 0], ego[:, 1], color="#1565c0", linewidth=2.5, label="Ego path")
    ax.plot(cart[:, 0], cart[:, 1], color="#ef6c00", linewidth=2.5, linestyle="--", label="Golf-cart path")
    ax.scatter(0, 0, marker="*", s=240, color="#fdd835", edgecolor="black", label="Traffic light")
    ax.scatter(*ego[0], marker="s", s=80, color="#2e7d32", label="Ego start")
    ax.scatter(*cart[0], marker="^", s=90, color="#c62828", label="Cart start")
    ax.scatter(*ego[-1], marker="o", s=75, color="#7e57c2", label="Ego end")
    ax.scatter(*cart[-1], marker="D", s=75, color="#6d4c41", label="Cart end")
    ax.set(title="Part B: Ego and Golf-Cart Ground-Frame Trajectories", xlabel="X forward (m)", ylabel="Y left (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    # Write atomically so an interrupted run cannot leave a zero-byte deliverable.
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary, dpi=180)
    os.replace(temporary, output)
    plt.close(fig)


def save_world_trajectories(ego: np.ndarray, cart: np.ndarray, path: Path, fps: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "ego_x_m", "ego_y_m", "golf_cart_x_m", "golf_cart_y_m"])
        for i, (ego_xy, cart_xy) in enumerate(zip(ego, cart)):
            writer.writerow([i, i / fps, *ego_xy, *cart_xy])


def save_bev_video(
    ego: np.ndarray, cart: np.ndarray, light_colors: list[str], output: Path, fps: int
) -> None:
    all_points = np.vstack((ego, cart, np.array([[0.0, 0.0]])))
    margin = 2.0
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.set(
        title="Part B: Dynamic Ground-Frame BEV",
        xlabel="X forward (m)", ylabel="Y left (m)",
        xlim=(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin),
        ylim=(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ego_line, = ax.plot([], [], color="#1565c0", linewidth=2.5, label="Ego path")
    cart_line, = ax.plot([], [], color="#ef6c00", linewidth=2.5, linestyle="--", label="Golf-cart path")
    ego_dot = ax.scatter([], [], marker="o", s=90, color="#7e57c2", label="Ego")
    cart_dot = ax.scatter([], [], marker="D", s=90, color="#6d4c41", label="Golf cart")
    light = ax.scatter([0], [0], marker="*", s=260, color=light_colors[0], edgecolor="black", label="Traffic light")
    timestamp = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    ax.legend(loc="best")

    def update(index: int):
        ego_line.set_data(ego[: index + 1, 0], ego[: index + 1, 1])
        cart_line.set_data(cart[: index + 1, 0], cart[: index + 1, 1])
        ego_dot.set_offsets(ego[index:index + 1])
        cart_dot.set_offsets(cart[index:index + 1])
        light.set_facecolor(light_colors[index])
        timestamp.set_text(f"t = {index / fps:.2f} s")
        return ego_line, cart_line, ego_dot, cart_dot, light, timestamp

    animation = FuncAnimation(fig, update, frames=len(ego), interval=1000 / fps, blit=True)
    animation.save(output, writer=FFMpegWriter(fps=fps, bitrate=2000))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--initial-box", type=int, nargs=4, default=(635, 585, 160, 180), metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames, cart_boxes, tracking_valid = track_cart(args.dataset / "rgb", tuple(args.initial_box))
    light_boxes, light_points, ego, _ = extract_trajectory(args.dataset, smooth_window=15)
    cart_points = sample_cart_depth(args.dataset / "xyz", cart_boxes)
    cart = cart_ground_positions(cart_points, cart_boxes, light_points, light_boxes, ego, 1920)
    colors = traffic_light_colors(frames, light_boxes)

    save_tracking_csv(cart_boxes, tracking_valid, args.output / "golf_cart_bboxes.csv")
    save_world_trajectories(ego, cart, args.output / "bev_part_b.csv", args.fps)
    save_quality_sheet(frames, cart_boxes, args.output / "tracking_quality.png")
    save_bev_plot(ego, cart, args.output / "bev_part_b.png")
    save_bev_video(ego, cart, colors, args.output / "bev_part_b.mp4", args.fps)
    print(f"Tracked golf cart in {tracking_valid.sum()}/{len(frames)} frames")
    print(f"Part B outputs written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
