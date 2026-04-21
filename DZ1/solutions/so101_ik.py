from pathlib import Path
from functools import lru_cache

import numpy as np
import sympy
import pytorch_kinematics as pk
import torch

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "so101" / "robot.urdf"

_PAN_X = sympy.Float("0.0388353")
_PAN_Z = sympy.Float("0.0624")
_SHOULDER_DX = sympy.Float("0.0303992")
_SHOULDER_DZ = sympy.Float("0.0542")
_ELBOW_DX = sympy.Float("0.028")
_ELBOW_DZ = sympy.Float("0.11257")
_WRIST_LINK = sympy.Float("0.1349")
_WRIST_DROP = sympy.Float("0.0611") + sympy.Float("0.1034")
_ROLL_OFFSET = sympy.Float("0.0486795")
_L1 = sympy.sqrt(_ELBOW_DX**2 + _ELBOW_DZ**2)
_ALPHA = sympy.atan2(_ELBOW_DX, _ELBOW_DZ)


@lru_cache(maxsize=1)
def _get_serial_chain() -> pk.SerialChain:
    chain = pk.build_chain_from_urdf(open(URDF_PATH, mode="rb").read())
    return pk.SerialChain(chain, "gripper_frame_link", "base_link")


@lru_cache(maxsize=1)
def _get_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    low, high = _get_serial_chain().get_joint_limits()
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


@lru_cache(maxsize=1)
def _get_base_retry_configs() -> torch.Tensor:
    low, high = _get_joint_limits()
    retries = [np.zeros(5, dtype=np.float64)]
    pan_grid = np.linspace(low[0], high[0], 9)
    shoulder_grid = np.linspace(low[1], high[1], 7)
    elbow_grid = np.linspace(low[2], high[2], 7)
    for shoulder_pan in pan_grid:
        for shoulder_lift in shoulder_grid:
            for elbow_flex in elbow_grid:
                wrist_flex = np.clip(np.pi / 2 - shoulder_lift - elbow_flex, low[3], high[3])
                wrist_roll = np.clip(shoulder_pan + float(_ROLL_OFFSET), low[4], high[4])
                retries.append(
                    np.array(
                        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll],
                        dtype=np.float64,
                    )
                )
    retries = np.unique(np.asarray(retries, dtype=np.float64), axis=0)
    return torch.tensor(retries, dtype=torch.float32)


def _build_target_matrix(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    target = np.eye(4, dtype=np.float64)
    # This orientation yields the same yaw convention as the tests' YZX extraction.
    target[:3, :3] = np.array(
        [[-c, s, 0.0], [s, c, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    target[:3, 3] = (x, y, z)
    return target


def so101_downturned_ik_symbolic(
    x: sympy.Symbol,
    y: sympy.Symbol,
    z: sympy.Symbol,
    yaw: sympy.Symbol,
) -> dict[str, sympy.Expr]:
    """
    Return a dict mapping each joint name to a sympy expression in (x, y, z, yaw).

    Parameters
    ----------
    x, y, z, yaw : sympy.Symbol
        Symbols for end-effector position and yaw.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a sympy expression (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    q1 = -sympy.atan2(y, x - _PAN_X)

    wrist_x = sympy.sqrt((x - _PAN_X) ** 2 + y**2) - _SHOULDER_DX
    wrist_z = z + _WRIST_DROP - (_PAN_Z + _SHOULDER_DZ)
    cos_phi = (wrist_x**2 + wrist_z**2 - _L1**2 - _WRIST_LINK**2) / (2 * _L1 * _WRIST_LINK)
    phi = sympy.acos(cos_phi)
    theta1 = sympy.atan2(wrist_z, wrist_x) - sympy.atan2(
        -_WRIST_LINK * sympy.sin(phi),
        _L1 + _WRIST_LINK * sympy.cos(phi),
    )

    q2 = sympy.pi / 2 - _ALPHA - theta1
    q3 = phi + _ALPHA - sympy.pi / 2
    q4 = sympy.pi / 2 - q2 - q3
    q5 = q1 - yaw + _ROLL_OFFSET

    return {
        "shoulder_pan": sympy.simplify(q1),
        "shoulder_lift": sympy.simplify(q2),
        "elbow_flex": sympy.simplify(q3),
        "wrist_flex": sympy.simplify(q4),
        "wrist_roll": sympy.simplify(q5),
    }


def analytical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float]:
    """
    Evaluate the analytical IK formulas numerically and check joint limits.

    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw angle (radians) in the downturned end-effector plane.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    x_sym, y_sym, z_sym, yaw_sym = sympy.symbols("x y z yaw", real=True)
    formulas = so101_downturned_ik_symbolic(x_sym, y_sym, z_sym, yaw_sym)
    func = sympy.lambdify(
        (x_sym, y_sym, z_sym, yaw_sym),
        [formulas[k] for k in SO101_JOINT_NAMES],
        "numpy",
    )
    q = dict(zip(SO101_JOINT_NAMES, func(x, y, z, yaw)))

    if any(not np.isfinite(value) for value in q.values()):
        return None

    low, high = _get_joint_limits()
    for idx, joint_name in enumerate(SO101_JOINT_NAMES):
        joint_angle = float(q[joint_name])
        if joint_angle < low[idx] or joint_angle > high[idx]:
            return None
        q[joint_name] = joint_angle
    return q


def numerical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float] | None:
    """
    Numerical IK for a downturned SO101 pose.

    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw (radians) in the downturned end-effector plane.

    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    serial_chain = _get_serial_chain()
    low, high = _get_joint_limits()

    retry_configs = [_get_base_retry_configs().numpy()]
    q_analytical = analytical_ik_so101_downturned(x, y, z, yaw)
    if q_analytical is not None:
        retry_configs.append(
            np.array(
                [[q_analytical[name] for name in SO101_JOINT_NAMES]],
                dtype=np.float32,
            )
        )
    retry_configs = torch.tensor(
        np.unique(np.vstack(retry_configs).astype(np.float32), axis=0),
        dtype=torch.float32,
    )

    solver = pk.PseudoInverseIK(
        serial_chain,
        retry_configs=retry_configs,
        pos_tolerance=1e-4,
        rot_tolerance=1e-3,
        max_iterations=160,
        lr=0.25,
        regularlization=1e-6,
        lm_damping=0.5,
        early_stopping_any_converged=False,
        early_stopping_no_improvement="any",
        enforce_joint_limits=True,
    )
    target = torch.tensor(_build_target_matrix(x, y, z, yaw), dtype=torch.float32).unsqueeze(0)
    solution = solver.solve(pk.Transform3d(matrix=target))
    if not bool(solution.converged_any[0]):
        return None

    converged = solution.converged[0]
    scores = torch.where(
        converged,
        solution.err_pos[0] + solution.err_rot[0],
        torch.full_like(solution.err_pos[0], 1e9),
    )
    best_idx = int(torch.argmin(scores))
    best_q = solution.solutions[0, best_idx].detach().cpu().numpy().astype(np.float64)
    if np.any(best_q < low) or np.any(best_q > high):
        return None
    return {name: float(best_q[idx]) for idx, name in enumerate(SO101_JOINT_NAMES)}
