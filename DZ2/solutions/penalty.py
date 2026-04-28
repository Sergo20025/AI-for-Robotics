"""Student solution: penalty contact forces. Task 3."""
from dataclasses import dataclass
from typing import List

import numpy as np

from lib.phys.phys_objects import RigidBody, Particle
from lib.phys.contact_forces.handler import ContactHandlerBase
from lib.phys.collisions.collision_detector import Collision


@dataclass
class PenaltyMethod(ContactHandlerBase):
    k_s: float = 10000.0
    k_d: float = 20.0
    mu: float = 0.5
    k_drag: float = 10.0

    def step(self, collisions: List[Collision], t_0: float, t_1: float):
        for collision in collisions:
            obj_a = collision.obj_a
            obj_b = collision.obj_b
            normal = np.asarray(collision.normal, dtype=np.float64)
            point_a = np.asarray(collision.point_a, dtype=np.float64)
            point_b = np.asarray(collision.point_b, dtype=np.float64)

            vel_a = np.asarray(obj_a.body_velocity, dtype=np.float64)
            vel_b = np.asarray(obj_b.body_velocity, dtype=np.float64)
            if isinstance(obj_a, RigidBody):
                vel_a = vel_a + np.cross(np.asarray(obj_a.angular_velocity, dtype=np.float64), point_a)
            if isinstance(obj_b, RigidBody):
                vel_b = vel_b + np.cross(np.asarray(obj_b.angular_velocity, dtype=np.float64), point_b)

            v_rel = vel_a - vel_b
            v_rel_n = float(v_rel @ normal)
            force_n_mag = max(self.k_s * collision.depth + self.k_d * v_rel_n, 0.0)
            force_n = -force_n_mag * normal

            v_rel_t = v_rel - v_rel_n * normal
            force_t = np.zeros(3, dtype=np.float64)
            v_rel_t_norm = float(np.linalg.norm(v_rel_t))
            max_friction = self.mu * force_n_mag
            if v_rel_t_norm > 1e-12 and max_friction > 0.0:
                friction_mag = min(self.k_drag * v_rel_t_norm, max_friction)
                force_t = -friction_mag * (v_rel_t / v_rel_t_norm)

            total_force = force_n + force_t
            obj_a.force_accumulator += total_force
            if isinstance(obj_a, RigidBody):
                obj_a.torque_accumulator += np.cross(point_a, total_force)
