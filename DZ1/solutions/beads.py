from __future__ import annotations

import math

import numpy as np

from lib.beads import build_necklace, bounding_sphere_radius

JOINT_LIMIT_RADIUS = np.pi / 3
_SAFE_COS = 0.5002
_SAFE_DIST = 2.0001
_A_GRID = np.linspace(0.60, 1.04, 23)
_B_GRID = np.linspace(0.00, 0.35, 15)
_REFINE_STEPS = (0.05, 0.02, 0.01)


def _constant_angles(link_count: int, theta1: float, theta2: float) -> np.ndarray:
    return np.tile(np.array([[theta1, theta2]], dtype=np.float64), (link_count - 1, 1))


def _is_feasible(link_lengths: np.ndarray, angles: np.ndarray) -> bool:
    vertices = build_necklace(link_lengths, angles)
    for i in range(len(vertices)):
        for j in range(i + 2, len(vertices)):
            if float(np.linalg.norm(vertices[i] - vertices[j])) < _SAFE_DIST:
                return False
    return True


def _radius_if_feasible(link_lengths: np.ndarray, theta1: float, theta2: float) -> float | None:
    if math.cos(theta1) * math.cos(theta2) < _SAFE_COS:
        return None
    angles = _constant_angles(len(link_lengths), theta1, theta2)
    if not _is_feasible(link_lengths, angles):
        return None
    return bounding_sphere_radius(link_lengths, angles)


def optimal_bead_config(
    link_lengths: np.ndarray
    ) -> np.ndarray:
    """
    Compute joint angles (in radians) that minimize the bounding sphere radius of the articulated chain.

    Parameters
    ----------
    link_lengths : array-like of shape (N,)
        Lengths of each link in the chain.

    Returns
    -------
    ndarray of shape (N-1, 2)
        Joint angles [theta1, theta2] at each joint, satisfying:
        - Spherical cap: great-circle angle from identity ≤ π/3, i.e. cos(θ₁)cos(θ₂) ≥ cos(π/3) (joint limits)
        - No overlap: Spheres of radius 1 at each joint do not intersect.

    Notes
    -----
    The returned configuration achieves maximal compactness (minimal enclosing sphere), respects joint limits,
    and maintains collision-free geometry between beads.
    """
    link_lengths = np.asarray(link_lengths, dtype=np.float64)
    if link_lengths.ndim != 1 or len(link_lengths) == 0:
        raise ValueError("link_lengths must be a 1D array with at least one link")
    if len(link_lengths) == 1:
        return np.zeros((0, 2), dtype=np.float64)

    best_radius = float("inf")
    best_theta1 = 0.85
    best_theta2 = 0.25

    for theta1 in _A_GRID:
        cos_theta1 = math.cos(float(theta1))
        for theta2 in _B_GRID:
            if cos_theta1 * math.cos(float(theta2)) < _SAFE_COS:
                continue
            radius = _radius_if_feasible(link_lengths, float(theta1), float(theta2))
            if radius is not None and radius < best_radius:
                best_radius = radius
                best_theta1 = float(theta1)
                best_theta2 = float(theta2)

    for step in _REFINE_STEPS:
        theta1_values = np.linspace(max(0.0, best_theta1 - step), min(float(JOINT_LIMIT_RADIUS), best_theta1 + step), 11)
        theta2_values = np.linspace(max(0.0, best_theta2 - step), min(float(JOINT_LIMIT_RADIUS), best_theta2 + step), 11)
        for theta1 in theta1_values:
            cos_theta1 = math.cos(float(theta1))
            for theta2 in theta2_values:
                if cos_theta1 * math.cos(float(theta2)) < _SAFE_COS:
                    continue
                radius = _radius_if_feasible(link_lengths, float(theta1), float(theta2))
                if radius is not None and radius < best_radius:
                    best_radius = radius
                    best_theta1 = float(theta1)
                    best_theta2 = float(theta2)

    return _constant_angles(len(link_lengths), best_theta1, best_theta2)
