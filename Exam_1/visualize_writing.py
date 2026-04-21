from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
URDF_PATH = ROOT / "robot.urdf"
TRAJ_PATH = ROOT / "exam_traj.json"
OUT_DIR = ROOT / "analysis"
FRAMES_DIR = OUT_DIR / "_gif_frames"
PNG_PATH = OUT_DIR / "result.png"
GIF_PATH = OUT_DIR / "result.gif"


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def parse_urdf(path: Path) -> tuple[dict[str, Joint], dict[str, list[str]]]:
    root = ET.parse(path).getroot()
    joints: dict[str, Joint] = {}
    children: dict[str, list[str]] = {}

    for joint_node in root.findall("joint"):
        origin = joint_node.find("origin")
        axis = joint_node.find("axis")
        joint = Joint(
            name=joint_node.attrib["name"],
            joint_type=joint_node.attrib["type"],
            parent=joint_node.find("parent").attrib["link"],
            child=joint_node.find("child").attrib["link"],
            xyz=np.array([float(x) for x in origin.attrib.get("xyz", "0 0 0").split()]),
            rpy=np.array([float(x) for x in origin.attrib.get("rpy", "0 0 0").split()]),
            axis=np.array(
                [float(x) for x in (axis.attrib.get("xyz", "0 0 0") if axis is not None else "0 0 0").split()]
            ),
        )
        joints[joint.name] = joint
        children.setdefault(joint.parent, []).append(joint.name)

    return joints, children


def find_chain(
    joints: dict[str, Joint],
    children: dict[str, list[str]],
    base_link: str = "base_link",
    target_link: str = "marker_holder_link",
) -> list[str]:
    chain: list[str] = []

    def dfs(link: str) -> bool:
        if link == target_link:
            return True
        for joint_name in children.get(link, []):
            chain.append(joint_name)
            if dfs(joints[joint_name].child):
                return True
            chain.pop()
        return False

    if not dfs(base_link):
        raise RuntimeError(f"Could not find chain from {base_link} to {target_link}")
    return chain


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def transform_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-12:
        transform[:3, :3] = R.from_rotvec(axis / axis_norm * angle).as_matrix()
    return transform


def compute_holder_fk() -> tuple[np.ndarray, np.ndarray]:
    joints, children = parse_urdf(URDF_PATH)
    chain = find_chain(joints, children)
    traj = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))

    positions = []
    rotations = []
    for sample in traj["joints"]:
        transform = np.eye(4)
        for joint_name in chain:
            joint = joints[joint_name]
            transform = transform @ transform_from_xyz_rpy(joint.xyz, joint.rpy)
            if joint.joint_type != "fixed":
                transform = transform @ transform_from_axis_angle(joint.axis, sample[joint_name])
        positions.append(transform[:3, 3].copy())
        rotations.append(transform[:3, :3].copy())

    return np.array(positions), np.array(rotations)


def estimate_tip_offset(holder_positions: np.ndarray, holder_rotations: np.ndarray) -> np.ndarray:
    z_projection = np.stack(
        [holder_rotations[:, 2, 0], holder_rotations[:, 2, 1], holder_rotations[:, 2, 2]],
        axis=1,
    )
    base_z = holder_positions[:, 2]
    best_score: tuple[int, float] | None = None
    best_offset: np.ndarray | None = None

    def evaluate(offset: np.ndarray, eps: float) -> tuple[int, float]:
        tip_z = base_z + z_projection @ offset
        paper_z = float(np.quantile(tip_z, 0.15))
        near_paper = np.abs(tip_z - paper_z) < eps
        if not np.any(near_paper):
            return 0, float("inf")
        return int(near_paper.sum()), float(np.std(tip_z[near_paper]))

    for ox in np.linspace(-0.04, 0.04, 17):
        for oy in np.linspace(-0.04, 0.04, 17):
            for oz in np.linspace(0.02, 0.10, 17):
                offset = np.array([ox, oy, oz])
                score = evaluate(offset, eps=0.0015)
                ranking = (score[0], -score[1])
                if best_score is None or ranking > best_score:
                    best_score = ranking
                    best_offset = offset

    assert best_offset is not None

    for ox in np.linspace(best_offset[0] - 0.01, best_offset[0] + 0.01, 21):
        for oy in np.linspace(best_offset[1] - 0.01, best_offset[1] + 0.01, 21):
            for oz in np.linspace(max(0.0, best_offset[2] - 0.02), best_offset[2] + 0.02, 21):
                offset = np.array([ox, oy, oz])
                score = evaluate(offset, eps=0.001)
                ranking = (score[0], -score[1])
                if ranking > best_score:
                    best_score = ranking
                    best_offset = offset

    return best_offset


def project_tip_path_to_writing_plane(tip_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    paper_z = float(np.quantile(tip_positions[:, 2], 0.15))
    pen_down_mask = np.abs(tip_positions[:, 2] - paper_z) < 0.003

    contact_points = tip_positions[pen_down_mask]
    center = contact_points.mean(axis=0)
    _, _, basis = np.linalg.svd(contact_points - center, full_matrices=False)

    projected = (tip_positions - center) @ basis[:2].T
    projected = orient_for_reading(projected)
    return projected, pen_down_mask


def orient_for_reading(points_2d: np.ndarray) -> np.ndarray:
    oriented = points_2d.copy()
    span = oriented.max(axis=0) - oriented.min(axis=0)

    if span[1] > span[0]:
        oriented = oriented @ np.array([[0.0, -1.0], [1.0, 0.0]])

    if oriented[0, 0] > oriented[-1, 0]:
        oriented[:, 0] = oriented[:, 0].min() + oriented[:, 0].max() - oriented[:, 0]

    return oriented


def plot_frame(points_2d: np.ndarray, pen_down_mask: np.ndarray, upto: int, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

    if upto > 0:
        pen_down_idx = np.where(pen_down_mask[:upto])[0]
        pen_up_idx = np.where(~pen_down_mask[:upto])[0]

        if len(pen_up_idx) > 0:
            ax.scatter(
                points_2d[pen_up_idx, 0],
                points_2d[pen_up_idx, 1],
                color="lightgray",
                s=8,
                linewidths=0,
                alpha=0.9,
            )

        if len(pen_down_idx) > 0:
            ax.scatter(
                points_2d[pen_down_idx, 0],
                points_2d[pen_down_idx, 1],
                c=np.arange(len(pen_down_idx)),
                cmap="viridis",
                s=12,
                linewidths=0,
            )

        if upto > 1:
            ax.plot(points_2d[:upto, 0], points_2d[:upto, 1], color="black", alpha=0.10, linewidth=1.0)

        ax.scatter(points_2d[upto - 1, 0], points_2d[upto - 1, 1], color="#1f77b4", s=26, zorder=3)

    x_pad = 0.015
    y_pad = 0.015
    ax.set_xlim(points_2d[:, 0].min() - x_pad, points_2d[:, 0].max() + x_pad)
    ax.set_ylim(points_2d[:, 1].min() - y_pad, points_2d[:, 1].max() + y_pad)
    ax.set_title("Marker Tip Trajectory")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def save_png(points_2d: np.ndarray, pen_down_mask: np.ndarray) -> None:
    plot_frame(points_2d, pen_down_mask, len(points_2d), PNG_PATH)


def save_gif(points_2d: np.ndarray, pen_down_mask: np.ndarray) -> None:
    frame_indices = np.unique(np.linspace(1, len(points_2d), 90, dtype=int))
    FRAMES_DIR.mkdir(exist_ok=True)
    for old_frame in FRAMES_DIR.glob("frame_*.png"):
        old_frame.unlink()

    frame_paths = []
    for frame_no, idx in enumerate(frame_indices):
        frame_path = FRAMES_DIR / f"frame_{frame_no:03d}.png"
        plot_frame(points_2d, pen_down_mask, int(idx), frame_path)
        frame_paths.append(frame_path)

    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    frames.extend([frames[-1]] * 10)
    imageio.mimsave(GIF_PATH, frames, duration=0.07, loop=0)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    holder_positions, holder_rotations = compute_holder_fk()
    tip_offset = estimate_tip_offset(holder_positions, holder_rotations)
    tip_positions = holder_positions + np.einsum("nij,j->ni", holder_rotations, tip_offset)
    points_2d, pen_down_mask = project_tip_path_to_writing_plane(tip_positions)

    save_png(points_2d, pen_down_mask)
    save_gif(points_2d, pen_down_mask)

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved GIF: {GIF_PATH}")
    print(f"Tip offset estimate: {tip_offset.tolist()}")
    print(f"Pen-down samples: {int(pen_down_mask.sum())} / {len(pen_down_mask)}")
    print("Visualization: estimated marker tip projected onto writing plane")


if __name__ == "__main__":
    main()
