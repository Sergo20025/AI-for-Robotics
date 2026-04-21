from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")


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
    path: list[str] = []

    def dfs(link: str) -> bool:
        if link == target_link:
            return True
        for joint_name in children.get(link, []):
            path.append(joint_name)
            if dfs(joints[joint_name].child):
                return True
            path.pop()
        return False

    if not dfs(base_link):
        raise RuntimeError(f"Could not find chain from {base_link} to {target_link}")
    return path


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def transform_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4)
    norm = np.linalg.norm(axis)
    if norm > 1e-12:
        transform[:3, :3] = R.from_rotvec(axis / norm * angle).as_matrix()
    return transform


def compute_marker_holder_xy() -> tuple[np.ndarray, np.ndarray]:
    joints, children = parse_urdf(URDF_PATH)
    chain = find_chain(joints, children)
    traj = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))

    xy_points = []
    xyz_points = []
    for sample in traj["joints"]:
        transform = np.eye(4)
        for joint_name in chain:
            joint = joints[joint_name]
            transform = transform @ transform_from_xyz_rpy(joint.xyz, joint.rpy)
            if joint.joint_type != "fixed":
                transform = transform @ transform_from_axis_angle(joint.axis, sample[joint_name])
        point = transform[:3, 3].copy()
        xyz_points.append(point)
        xy_points.append(point[:2])

    xy_points = np.array(xy_points)
    xyz_points = np.array(xyz_points)
    return xy_points, xyz_points


def orient_horizontally(points: np.ndarray) -> np.ndarray:
    oriented = points.copy()
    oriented[:, 0] = oriented[:, 0].min() + oriented[:, 0].max() - oriented[:, 0]
    oriented = oriented @ np.array([[0.0, -1.0], [1.0, 0.0]])
    oriented[:, 0] = oriented[:, 0].min() + oriented[:, 0].max() - oriented[:, 0]
    return oriented


def plot_frame(
    points_xy: np.ndarray,
    upto: int,
    png_path: Path,
    current_color: str = "#1f77b4",
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

    ax.scatter(
        points_xy[:upto, 0],
        points_xy[:upto, 1],
        c=np.arange(upto),
        cmap="viridis",
        s=12,
        linewidths=0,
    )
    if upto > 1:
        ax.plot(points_xy[:upto, 0], points_xy[:upto, 1], color="black", alpha=0.12, linewidth=1.0)
    if upto > 0:
        ax.scatter(points_xy[upto - 1, 0], points_xy[upto - 1, 1], color=current_color, s=26, zorder=3)

    x_pad = 0.015
    y_pad = 0.015
    ax.set_xlim(points_xy[:, 0].min() - x_pad, points_xy[:, 0].max() + x_pad)
    ax.set_ylim(points_xy[:, 1].min() - y_pad, points_xy[:, 1].max() + y_pad)
    ax.set_title("Marker Holder XY")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def save_png(points_xy: np.ndarray) -> None:
    plot_frame(points_xy, len(points_xy), PNG_PATH)


def save_gif(points_xy: np.ndarray) -> None:
    frame_indices = np.unique(np.linspace(1, len(points_xy), 90, dtype=int))
    FRAMES_DIR.mkdir(exist_ok=True)
    for old_frame in FRAMES_DIR.glob("frame_*.png"):
        old_frame.unlink()

    frame_paths = []
    for frame_no, idx in enumerate(frame_indices):
        frame_path = FRAMES_DIR / f"frame_{frame_no:03d}.png"
        plot_frame(points_xy, int(idx), frame_path)
        frame_paths.append(frame_path)

    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    frames.extend([frames[-1]] * 10)
    imageio.mimsave(GIF_PATH, frames, duration=0.07, loop=0)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    points_xy, xyz_points = compute_marker_holder_xy()
    points_xy = orient_horizontally(points_xy)

    save_png(points_xy)
    save_gif(points_xy)

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved GIF: {GIF_PATH}")
    print(f"Marker holder XY samples: {len(points_xy)}")
    print(f"XYZ bounds min: {xyz_points.min(axis=0).tolist()}")
    print(f"XYZ bounds max: {xyz_points.max(axis=0).tolist()}")
    print("Orientation: mirrored and rotated to horizontal reading view")


if __name__ == "__main__":
    main()
