from __future__ import annotations

from functools import lru_cache
from typing import Callable
import math

import numpy as np
from scipy.optimize import minimize

from lib.broom_racing import Configuration, PHI_MAX, PHI_MIN, XYZConfiguration, curve_length

_EPS = 1e-9


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clip_phi(phi: float) -> float:
    return float(np.clip(phi, PHI_MIN, PHI_MAX))


def _direction(theta: float, phi: float) -> np.ndarray:
    cp = math.cos(phi)
    return np.array(
        [math.cos(theta) * cp, math.sin(theta) * cp, math.sin(phi)],
        dtype=np.float64,
    )


def _angles_from_vector(vec: np.ndarray) -> tuple[float, float]:
    vec = np.asarray(vec, dtype=np.float64)
    norm = max(float(np.linalg.norm(vec)), _EPS)
    vec = vec / norm
    return math.atan2(vec[1], vec[0]), math.asin(float(np.clip(vec[2], -1.0, 1.0)))


def _pitch_displacement(theta: float, phi0: float, phi1: float, length: float | None = None) -> np.ndarray:
    delta = phi1 - phi0
    if abs(delta) < _EPS:
        return np.zeros(3, dtype=np.float64)
    sign = 1.0 if delta >= 0.0 else -1.0
    if length is None:
        length = abs(delta)
    phi = phi0 + sign * length
    return np.array(
        [
            math.cos(theta) * (math.sin(phi) - math.sin(phi0)) / sign,
            math.sin(theta) * (math.sin(phi) - math.sin(phi0)) / sign,
            (math.cos(phi0) - math.cos(phi)) / sign,
        ],
        dtype=np.float64,
    )


def _heading_displacement(
    theta0: float,
    delta_theta: float,
    phi: float,
    length: float | None = None,
) -> np.ndarray:
    if abs(delta_theta) < _EPS:
        return np.zeros(3, dtype=np.float64)
    sign = 1.0 if delta_theta >= 0.0 else -1.0
    cp = max(math.cos(phi), _EPS)
    total_length = abs(delta_theta) * cp
    if length is None:
        length = total_length
    theta = theta0 + sign * length / cp
    cp_sq = math.cos(phi) ** 2
    return np.array(
        [
            cp_sq * (math.sin(theta) - math.sin(theta0)) / sign,
            cp_sq * (-math.cos(theta) + math.cos(theta0)) / sign,
            length * math.sin(phi),
        ],
        dtype=np.float64,
    )


def _line_of_sight_angles(start_pos: np.ndarray, goal_pos: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(goal_pos, dtype=np.float64) - np.asarray(start_pos, dtype=np.float64)
    theta = math.atan2(delta[1], delta[0])
    phi = math.atan2(delta[2], max(float(np.hypot(delta[0], delta[1])), _EPS))
    return theta, _clip_phi(phi)


def _segment_endpoints(segment: tuple, start_pos: np.ndarray) -> np.ndarray:
    seg_type = segment[0]
    if seg_type == "pitch":
        _, theta, phi0, phi1 = segment
        return start_pos + _pitch_displacement(theta, phi0, phi1)
    if seg_type == "heading":
        _, theta0, delta_theta, phi = segment
        return start_pos + _heading_displacement(theta0, delta_theta, phi)
    _, theta, phi, length = segment
    return start_pos + length * _direction(theta, phi)


def _build_curve_from_segments(
    start: Configuration,
    segments: tuple[tuple, ...],
) -> Callable[[np.ndarray], Configuration]:
    start_pos = np.array(start.position(), dtype=np.float64)
    start_positions = [start_pos]
    lengths = []
    for segment in segments:
        if segment[0] == "pitch":
            length = abs(segment[3] - segment[2])
        elif segment[0] == "heading":
            length = abs(segment[2]) * math.cos(segment[3])
        else:
            length = segment[3]
        lengths.append(float(length))
        start_positions.append(_segment_endpoints(segment, start_positions[-1]))
    cumulative = np.cumsum([0.0, *lengths], dtype=np.float64)
    total_length = float(cumulative[-1])

    def curve(s: np.ndarray) -> Configuration:
        if total_length < _EPS:
            return start
        s_value = float(np.asarray(s, dtype=np.float64).reshape(-1)[0])
        s_value = float(np.clip(s_value, 0.0, 1.0))
        arc = s_value * total_length
        idx = min(len(segments) - 1, int(np.searchsorted(cumulative[1:], arc, side="right")))
        local_arc = arc - cumulative[idx]
        segment = segments[idx]
        seg_type = segment[0]
        origin = start_positions[idx]

        if seg_type == "pitch":
            _, theta, phi0, phi1 = segment
            sign = 1.0 if phi1 >= phi0 else -1.0
            pos = origin + _pitch_displacement(theta, phi0, phi1, local_arc)
            return Configuration(
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(theta),
                float(phi0 + sign * local_arc),
            )

        if seg_type == "heading":
            _, theta0, delta_theta, phi = segment
            sign = 1.0 if delta_theta >= 0.0 else -1.0
            cp = max(math.cos(phi), _EPS)
            pos = origin + _heading_displacement(theta0, delta_theta, phi, local_arc)
            return Configuration(
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(theta0 + sign * local_arc / cp),
                float(phi),
            )

        _, theta, phi, _ = segment
        pos = origin + local_arc * _direction(theta, phi)
        return Configuration(float(pos[0]), float(pos[1]), float(pos[2]), float(theta), float(phi))

    return curve


def _compose_curves(
    curve_a: Callable[[np.ndarray], Configuration],
    curve_b: Callable[[np.ndarray], Configuration],
) -> Callable[[np.ndarray], Configuration]:
    len_a = curve_length(curve_a)
    len_b = curve_length(curve_b)
    total = len_a + len_b

    def curve(s: np.ndarray) -> Configuration:
        if total < _EPS:
            return curve_b(np.array([1.0]))
        s_value = float(np.asarray(s, dtype=np.float64).reshape(-1)[0])
        s_value = float(np.clip(s_value, 0.0, 1.0))
        arc = s_value * total
        if arc <= len_a:
            return curve_a(np.array([arc / max(len_a, _EPS)]))
        return curve_b(np.array([(arc - len_a) / max(len_b, _EPS)]))

    return curve


def _solve_gate_plan(
    start: Configuration,
    goal: Configuration,
) -> tuple[tuple[tuple, ...], float]:
    start_pos = np.array(start.position(), dtype=np.float64)
    goal_pos = np.array(goal.position(), dtype=np.float64)
    displacement = goal_pos - start_pos
    los_theta, los_phi = _line_of_sight_angles(start_pos, goal_pos)

    def objective(theta_c: float, phi_c: float, loop_1: int, loop_2: int) -> tuple[float, float, float, float, float]:
        delta_1 = theta_c - start.theta + 2.0 * math.pi * loop_1
        delta_2 = goal.theta - theta_c + 2.0 * math.pi * loop_2
        remainder = (
            displacement
            - _pitch_displacement(start.theta, start.phi, phi_c)
            - _heading_displacement(start.theta, delta_1, phi_c)
            - _heading_displacement(theta_c, delta_2, phi_c)
            - _pitch_displacement(goal.theta, phi_c, goal.phi)
        )
        cruise_dir = _direction(theta_c, phi_c)
        straight_length = float(np.dot(remainder, cruise_dir))
        perp_error = float(np.linalg.norm(remainder - straight_length * cruise_dir))
        total_length = (
            abs(phi_c - start.phi)
            + abs(delta_1) * math.cos(phi_c)
            + max(straight_length, 0.0)
            + abs(delta_2) * math.cos(phi_c)
            + abs(goal.phi - phi_c)
        )
        penalty = 1000.0 * perp_error**2 + 1000.0 * max(-straight_length, 0.0) ** 2
        return total_length + penalty, total_length, straight_length, delta_1, delta_2

    best: tuple[float, float, float, float, float, float] | None = None
    seeds = [(los_theta, los_phi), (start.theta, start.phi), (goal.theta, goal.phi)]
    for theta_c in np.linspace(-math.pi, math.pi, 13):
        for phi_c in np.linspace(PHI_MIN, PHI_MAX, 7):
            seeds.append((theta_c, phi_c))

    for loop_1 in (-1, 0, 1):
        for loop_2 in (-1, 0, 1):
            scored = []
            for theta_seed, phi_seed in seeds:
                score = objective(theta_seed, phi_seed, loop_1, loop_2)[0]
                scored.append((score, theta_seed, phi_seed))
            scored.sort(key=lambda item: item[0])
            for _, theta_seed, phi_seed in scored[:8]:
                res = minimize(
                    lambda x: objective(_wrap_angle(float(x[0])), _clip_phi(float(x[1])), loop_1, loop_2)[0],
                    np.array([theta_seed, phi_seed], dtype=np.float64),
                    method="Nelder-Mead",
                    options={"maxiter": 800, "xatol": 1e-6, "fatol": 1e-6},
                )
                theta_c = _wrap_angle(float(res.x[0]))
                phi_c = _clip_phi(float(res.x[1]))
                score, total_length, straight_length, delta_1, delta_2 = objective(theta_c, phi_c, loop_1, loop_2)
                candidate = (
                    score,
                    total_length,
                    theta_c,
                    phi_c,
                    straight_length,
                    delta_1,
                    delta_2,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate

    assert best is not None
    _, _, theta_c, phi_c, straight_length, delta_1, delta_2 = best
    segments = []
    if abs(phi_c - start.phi) > _EPS:
        segments.append(("pitch", start.theta, start.phi, phi_c))
    if abs(delta_1) > _EPS:
        segments.append(("heading", start.theta, delta_1, phi_c))
    if straight_length > _EPS:
        segments.append(("straight", theta_c, phi_c, straight_length))
    if abs(delta_2) > _EPS:
        segments.append(("heading", theta_c, delta_2, phi_c))
    if abs(goal.phi - phi_c) > _EPS:
        segments.append(("pitch", goal.theta, phi_c, goal.phi))
    return tuple(segments), float(sum(
        abs(seg[3] - seg[2]) if seg[0] == "pitch"
        else abs(seg[2]) * math.cos(seg[3]) if seg[0] == "heading"
        else seg[3]
        for seg in segments
    ))


def _solve_snitch_plan(
    start: Configuration,
    goal_xyz: XYZConfiguration,
) -> tuple[tuple[tuple, ...], float]:
    start_pos = np.array(start.position(), dtype=np.float64)
    goal_pos = np.array(goal_xyz.position(), dtype=np.float64)
    displacement = goal_pos - start_pos
    los_theta, los_phi = _line_of_sight_angles(start_pos, goal_pos)

    def objective(theta_c: float, phi_c: float, loop_1: int) -> tuple[float, float, float, float]:
        delta_1 = theta_c - start.theta + 2.0 * math.pi * loop_1
        remainder = (
            displacement
            - _pitch_displacement(start.theta, start.phi, phi_c)
            - _heading_displacement(start.theta, delta_1, phi_c)
        )
        cruise_dir = _direction(theta_c, phi_c)
        straight_length = float(np.dot(remainder, cruise_dir))
        perp_error = float(np.linalg.norm(remainder - straight_length * cruise_dir))
        total_length = abs(phi_c - start.phi) + abs(delta_1) * math.cos(phi_c) + max(straight_length, 0.0)
        penalty = 1000.0 * perp_error**2 + 1000.0 * max(-straight_length, 0.0) ** 2
        return total_length + penalty, total_length, straight_length, delta_1

    best: tuple[float, float, float, float, float, float] | None = None
    seeds = [(los_theta, los_phi), (start.theta, start.phi)]
    for theta_c in np.linspace(-math.pi, math.pi, 13):
        for phi_c in np.linspace(PHI_MIN, PHI_MAX, 7):
            seeds.append((theta_c, phi_c))

    for loop_1 in (-1, 0, 1):
        scored = []
        for theta_seed, phi_seed in seeds:
            score = objective(theta_seed, phi_seed, loop_1)[0]
            scored.append((score, theta_seed, phi_seed))
        scored.sort(key=lambda item: item[0])
        for _, theta_seed, phi_seed in scored[:8]:
            res = minimize(
                lambda x: objective(_wrap_angle(float(x[0])), _clip_phi(float(x[1])), loop_1)[0],
                np.array([theta_seed, phi_seed], dtype=np.float64),
                method="Nelder-Mead",
                options={"maxiter": 800, "xatol": 1e-6, "fatol": 1e-6},
            )
            theta_c = _wrap_angle(float(res.x[0]))
            phi_c = _clip_phi(float(res.x[1]))
            score, total_length, straight_length, delta_1 = objective(theta_c, phi_c, loop_1)
            candidate = (score, total_length, theta_c, phi_c, straight_length, delta_1)
            if best is None or candidate[0] < best[0]:
                best = candidate

    assert best is not None
    _, _, theta_c, phi_c, straight_length, delta_1 = best
    segments = []
    if abs(phi_c - start.phi) > _EPS:
        segments.append(("pitch", start.theta, start.phi, phi_c))
    if abs(delta_1) > _EPS:
        segments.append(("heading", start.theta, delta_1, phi_c))
    if straight_length > _EPS:
        segments.append(("straight", theta_c, phi_c, straight_length))
    return tuple(segments), float(sum(
        abs(seg[3] - seg[2]) if seg[0] == "pitch"
        else abs(seg[2]) * math.cos(seg[3]) if seg[0] == "heading"
        else seg[3]
        for seg in segments
    ))


@lru_cache(maxsize=None)
def _cached_gate_curve(
    start_x: float,
    start_y: float,
    start_z: float,
    start_theta: float,
    start_phi: float,
    goal_x: float,
    goal_y: float,
    goal_z: float,
    goal_theta: float,
    goal_phi: float,
) -> Callable[[np.ndarray], Configuration]:
    start = Configuration(start_x, start_y, start_z, start_theta, start_phi)
    goal = Configuration(goal_x, goal_y, goal_z, goal_theta, goal_phi)
    segments, _ = _solve_gate_plan(start, goal)
    return _build_curve_from_segments(start, segments)


@lru_cache(maxsize=None)
def _cached_snitch_curve(
    start_x: float,
    start_y: float,
    start_z: float,
    start_theta: float,
    start_phi: float,
    goal_x: float,
    goal_y: float,
    goal_z: float,
) -> Callable[[np.ndarray], Configuration]:
    start = Configuration(start_x, start_y, start_z, start_theta, start_phi)
    goal_xyz = XYZConfiguration(goal_x, goal_y, goal_z)
    segments, _ = _solve_snitch_plan(start, goal_xyz)
    return _build_curve_from_segments(start, segments)


def gate_pass(
    start: Configuration,
    goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    """
    Fly from start to goal through the Quidditch gate (contest 1).

    Parameters
    ----------
    start : Configuration
        Initial pose (x, y, z, θ, φ). Speed v=1; θ heading, φ pitch.
    goal : Configuration
        Gate pose to pass through (position and orientation).

    Returns
    -------
    callable
        curve(s) for s ∈ [0, 1], returning Configuration. Must satisfy discretized EOM
        (ẋ=v cos θ cos φ, ẏ=v sin θ cos φ, ż=v sin φ, θ̇=u₁/cos φ, φ̇=u₂), pitch φ ∈ [φ_min, φ_max],
        and curvature κ = √(u₁² + u₂²) ≤ κ_max. Path length ≤ reference implementation.
    """
    return _cached_gate_curve(
        float(start.x),
        float(start.y),
        float(start.z),
        float(start.theta),
        float(start.phi),
        float(goal.x),
        float(goal.y),
        float(goal.z),
        float(goal.theta),
        float(goal.phi),
    )


def catch_snitch(
    start: Configuration,
    goal_xyz: XYZConfiguration,
) -> Callable[[np.ndarray], Configuration]:
    """
    Reach the Snitch at goal_xyz (contest 2); only position matters at goal.

    Parameters
    ----------
    start : Configuration
        Initial pose (x, y, z, θ, φ).
    goal_xyz : XYZConfiguration
        Target position (x, y, z) to reach.

    Returns
    -------
    callable
        curve(s) for s ∈ [0, 1], returning Configuration. Same EOM and constraint
        requirements as gate_pass; path length ≤ reference.
    """
    return _cached_snitch_curve(
        float(start.x),
        float(start.y),
        float(start.z),
        float(start.theta),
        float(start.phi),
        float(goal_xyz.x),
        float(goal_xyz.y),
        float(goal_xyz.z),
    )


def catch_ball_and_gate(
    start: Configuration,
    intermediate_goal_xyz: XYZConfiguration,
    final_goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    """
    Catch the ball at intermediate_goal_xyz then fly through the gate at final_goal (contest 3).

    Parameters
    ----------
    start : Configuration
        Initial pose.
    intermediate_goal_xyz : XYZConfiguration
        Ball position (x, y, z) to visit first.
    final_goal : Configuration
        Gate pose to pass through after catching the ball.

    Returns
    -------
    callable
        curve(s) for s ∈ [0, 1], returning Configuration. Curve visits ball then gate;
        same EOM and constraint requirements; path length ≤ reference.
    """
    greedy_first = catch_snitch(start, intermediate_goal_xyz)
    greedy_mid = greedy_first(np.array([1.0]))
    greedy_second = gate_pass(greedy_mid, final_goal)
    best_curve = _compose_curves(greedy_first, greedy_second)
    best_length = curve_length(best_curve)

    ball_pos = intermediate_goal_xyz.position()
    to_final_theta, to_final_phi = _line_of_sight_angles(ball_pos, final_goal.position())
    avg_theta, avg_phi = _angles_from_vector(
        _direction(*_line_of_sight_angles(start.position(), ball_pos))
        + _direction(to_final_theta, to_final_phi)
    )
    for theta_mid, phi_mid in (
        (to_final_theta, to_final_phi),
        (avg_theta, _clip_phi(avg_phi)),
    ):
        mid_cfg = Configuration(
            float(intermediate_goal_xyz.x),
            float(intermediate_goal_xyz.y),
            float(intermediate_goal_xyz.z),
            float(theta_mid),
            float(phi_mid),
        )
        first_curve = gate_pass(start, mid_cfg)
        second_curve = gate_pass(mid_cfg, final_goal)
        candidate = _compose_curves(first_curve, second_curve)
        candidate_length = curve_length(candidate)
        if candidate_length < best_length:
            best_curve = candidate
            best_length = candidate_length

    return best_curve
