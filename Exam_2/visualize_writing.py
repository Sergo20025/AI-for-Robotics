from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "exam_traj_2.mp4"
OUT_DIR = ROOT / "analysis"
FRAMES_DIR = OUT_DIR / "_gif_frames"
PNG_PATH = OUT_DIR / "result.png"
GIF_PATH = OUT_DIR / "result.gif"

FX = 605.2803955078125
FY = 604.8619384765625
CX = 435.7801208496094
CY = 248.21617126464844
WIDTH = 848
HEIGHT = 480

ARUCO_DICT = cv2.aruco.DICT_6X6_250
ARUCO_ID = 0
MARKER_SIZE_M = 0.015


def intrinsics_matrix() -> np.ndarray:
    return np.array(
        [
            [FX, 0.0, CX],
            [0.0, FY, CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def marker_object_points() -> np.ndarray:
    half = MARKER_SIZE_M / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def detect_marker_track() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    camera_matrix = intrinsics_matrix()
    dist_coeffs = np.zeros(5, dtype=np.float64)
    object_points = marker_object_points()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    centers_2d: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if marker_id != ARUCO_ID:
                continue

            ok_pose, rvec, tvec = cv2.solvePnP(
                object_points,
                marker_corners[0].astype(np.float64),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok_pose:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                rotations.append(rotation_matrix)
                translations.append(tvec.reshape(3))
                centers_2d.append(marker_corners[0].mean(axis=0))
            break

    cap.release()

    if not rotations:
        raise RuntimeError("Could not detect the required ArUco marker in the video")

    return np.array(rotations), np.array(translations), np.array(centers_2d)


def smooth_points(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) < 5:
        return points_xy.copy()

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    x = np.convolve(points_xy[:, 0], kernel, mode="same")
    y = np.convolve(points_xy[:, 1], kernel, mode="same")
    return np.column_stack([x, y])


def orientation_variants(points_xy: np.ndarray) -> dict[str, np.ndarray]:
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    return {
        "orig": centered,
        "mirror_x": np.c_[-centered[:, 0], centered[:, 1]],
        "mirror_y": np.c_[centered[:, 0], -centered[:, 1]],
        "mirror_xy": -centered,
        "rot90": np.c_[-centered[:, 1], centered[:, 0]],
        "rot90_mirror_x": np.c_[centered[:, 1], centered[:, 0]],
        "rot90_mirror_y": np.c_[-centered[:, 1], -centered[:, 0]],
        "rot90_mirror_xy": np.c_[centered[:, 1], -centered[:, 0]],
    }


def choose_readable_orientation(points_xy: np.ndarray) -> tuple[str, np.ndarray]:
    best_name = "orig"
    best_points = points_xy
    best_score = -np.inf

    for name, candidate in orientation_variants(points_xy).items():
        width = float(np.ptp(candidate[:, 0]))
        height = float(np.ptp(candidate[:, 1]))
        if height < 1e-9:
            continue

        steps = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
        connect_threshold = max(float(np.percentile(steps, 70)), 0.01) if len(steps) else 0.01
        continuity = float((steps < connect_threshold).sum()) - float((steps >= connect_threshold).sum())
        score = 2.0 * (width / height) + 0.2 * continuity

        if score > best_score:
            best_score = score
            best_name = name
            best_points = candidate

    return best_name, best_points


def split_into_strokes(points_xy: np.ndarray) -> list[np.ndarray]:
    if len(points_xy) < 2:
        return [points_xy]

    steps = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    jump_threshold = max(float(np.percentile(steps, 85)), 25.0)

    strokes: list[np.ndarray] = []
    start = 0
    for idx, step in enumerate(steps, start=1):
        if step > jump_threshold:
            if idx - start >= 2:
                strokes.append(points_xy[start:idx])
            start = idx
    if len(points_xy) - start >= 2:
        strokes.append(points_xy[start:])

    return strokes if strokes else [points_xy]


def build_center_trajectory(centers_2d: np.ndarray) -> tuple[np.ndarray, str]:
    centered = centers_2d.copy()
    centered[:, 0] = WIDTH - centered[:, 0]
    centered = smooth_points(centered)
    orientation_name, final_xy = choose_readable_orientation(centered)
    return final_xy, orientation_name


def plot_frame(points_xy: np.ndarray, upto: int, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    current = points_xy[:upto]
    strokes = split_into_strokes(current)

    for stroke_idx, stroke in enumerate(strokes):
        ax.plot(
            stroke[:, 0],
            stroke[:, 1],
            color="#1f77b4",
            linewidth=2.8,
            label="trajectory" if stroke_idx == 0 else None,
        )
    ax.scatter(current[0, 0], current[0, 1], color="#2ca02c", s=90, zorder=3, label="start point")
    ax.scatter(current[-1, 0], current[-1, 1], color="#3366dd", s=90, zorder=3, label="end point")

    x_pad = max(float(np.ptp(points_xy[:, 0])) * 0.08, 10.0)
    y_pad = max(float(np.ptp(points_xy[:, 1])) * 0.08, 10.0)
    ax.set_xlim(points_xy[:, 0].min() - x_pad, points_xy[:, 0].max() + x_pad)
    ax.set_ylim(points_xy[:, 1].max() + y_pad, points_xy[:, 1].min() - y_pad)
    ax.set_title("Projection of the tracked marker trajectory", fontsize=22)
    ax.set_xlabel("u, px", fontsize=14)
    ax.set_ylabel("v, px", fontsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=12)

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def save_png(points_xy: np.ndarray) -> None:
    plot_frame(points_xy, len(points_xy), PNG_PATH)


def save_gif(points_xy: np.ndarray) -> None:
    FRAMES_DIR.mkdir(exist_ok=True)
    for old_frame in FRAMES_DIR.glob("frame_*.png"):
        old_frame.unlink()

    frame_indices = np.unique(np.linspace(2, len(points_xy), 90, dtype=int))
    frame_paths: list[Path] = []

    for frame_no, idx in enumerate(frame_indices):
        frame_path = FRAMES_DIR / f"frame_{frame_no:03d}.png"
        plot_frame(points_xy, int(idx), frame_path)
        frame_paths.append(frame_path)

    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    frames.extend([frames[-1]] * 10)
    imageio.mimsave(GIF_PATH, frames, duration=0.07, loop=0)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    _rotations, _translations, centers_2d = detect_marker_track()
    points_xy, orientation_name = build_center_trajectory(centers_2d)

    save_png(points_xy)
    save_gif(points_xy)

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved GIF: {GIF_PATH}")
    print(f"Detected marker: DICT_6X6_250, ID {ARUCO_ID}, size {MARKER_SIZE_M * 100:.1f} cm")
    print(f"Detections: {len(centers_2d)}")
    print(f"Chosen orientation: {orientation_name}")
    print("Visualization mode: mirrored ArUco center trajectory in image space")


if __name__ == "__main__":
    main()
