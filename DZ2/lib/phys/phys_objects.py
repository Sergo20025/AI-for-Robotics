import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional
from pyquaternion import Quaternion
from lib.phys.collisions.colliders import ColliderBase
from lib.basic_structs import Vec3, Pose

MAX_LINEAR_VELOCITY = 1e2
MAX_ANGULAR_VELOCITY = 1e2
EPS = 1e-6


@dataclass
class PhysObject(ABC):
    name: str

    @property
    @abstractmethod
    def state_x_size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def state_p_size(self):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        raise NotImplementedError


@dataclass
class Particle(PhysObject):
    inv_mass: float
    collider: Optional[ColliderBase] = None
    pos: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    linear_momentum: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    @property
    def body_velocity(self):
        return self.calc_velocity(self.linear_momentum)

    def calc_velocity(self, linear_momentum):
        return linear_momentum * self.inv_mass

    force_accumulator: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    @property
    def state_x_size(self):
        return len(self.get_state()['x'])

    @property
    def state_p_size(self):
        return len(self.get_state()['p'])

    def get_state(self):
        state = dict(
            x=self.pos,
            p=self.linear_momentum
        )
        return state

    def set_state(self, new_state):
        assert len(new_state['x']) == self.state_x_size
        assert len(new_state['p']) == self.state_p_size
        self.pos = new_state['x']
        self.linear_momentum = Vec3(new_state['p'])

    def dx_dt_func(self, x_state):
        linear_momentum = x_state['p']
        body_velocity = self.calc_velocity(linear_momentum)
        return {
            'x_dot': body_velocity,
            'p_dot': self.force_accumulator
        }


@dataclass
class RigidBody(PhysObject):
    inv_mass: float
    body_inertia_tensor_inv: np.array
    collider: Optional[ColliderBase] = None
    pose: Pose = field(default_factory=Pose)
    linear_momentum: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    angular_momentum: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    @property
    def body_velocity(self):
        return self.calc_velocity(self.linear_momentum)

    def calc_velocity(self, linear_momentum):
        return self.inv_mass * linear_momentum

    @property
    def rotation_matrix(self):
        return self.pose.q.rotation_matrix

    @property
    def inertia_tensor_inv(self):
        return self.calc_inertia_tensor_inv(self.pose.q)

    def calc_inertia_tensor_inv(self, q: Quaternion):
        rotation_matrix = q.rotation_matrix
        return rotation_matrix @ self.body_inertia_tensor_inv @ rotation_matrix.T

    @property
    def angular_velocity(self):
        return self.calc_angular_velocity(self.pose.q, self.angular_momentum)

    def calc_angular_velocity(self, q: Quaternion, angular_momentum: Vec3):
        inertia_tensor_inv = self.calc_inertia_tensor_inv(q)
        return inertia_tensor_inv @ angular_momentum

    force_accumulator: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    torque_accumulator: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    def get_state(self):
        state = dict(
            x=self.pose.raw,
            p=np.concatenate((self.linear_momentum, self.angular_momentum))
        )
        return state

    @property
    def state_x_size(self):
        return len(self.get_state()['x'])

    @property
    def state_p_size(self):
        return len(self.get_state()['p'])

    @property
    def state_size(self):
        return self.state_x_size + self.state_p_size

    def set_state(self, new_state):
        assert len(new_state['x']) == self.state_x_size
        assert len(new_state['p']) == self.state_p_size
        self.pose = Pose.from_raw(new_state['x'])
        self.pose.q._normalise()
        self.linear_momentum = Vec3(new_state['p'][:3])

        if np.linalg.norm(self.body_velocity) > MAX_LINEAR_VELOCITY:
            print(f"Body {self.name} reached maximum linear velocity")
            max_momentum = MAX_LINEAR_VELOCITY / self.inv_mass if self.inv_mass > 0 else EPS
            self.linear_momentum = self.linear_momentum / np.linalg.norm(self.linear_momentum) * max_momentum

        self.angular_momentum = Vec3(new_state['p'][3:])

        if np.linalg.norm(self.angular_velocity) > MAX_LINEAR_VELOCITY:
            print(f"Body {self.name} reached maximum angular velocity")
            if np.linalg.norm(self.inertia_tensor_inv, ord=2) > 0:
                max_angular_momentum = 1 / np.linalg.norm(self.inertia_tensor_inv, ord=2) * MAX_ANGULAR_VELOCITY
            else:
                max_angular_momentum = EPS
            self.angular_momentum = self.angular_momentum / np.linalg.norm(self.angular_momentum) * max_angular_momentum

    @property
    def kinetic_energy(self):
        energy = 0
        if self.inv_mass != 0:
            linear_k = 1 / 2 / self.inv_mass * self.body_velocity @ self.body_velocity
            energy += linear_k.tolist()
        if np.linalg.norm(self.inertia_tensor_inv, ord=2) > 0:
            angular_k = 1 / 2 * self.angular_velocity.T @ self.angular_momentum
            energy += angular_k.tolist()

        return energy

    def dx_dt_func(self, x_state):
        q = Quaternion(x_state['x'][3:])
        linear_momentum = x_state['p'][:3]
        angular_momentum = x_state['p'][3:]

        body_velocity = self.calc_velocity(linear_momentum)
        angular_velocity = self.calc_angular_velocity(q, angular_momentum)
        q_dot = 0.5 * Quaternion(vector=angular_velocity) * q

        P_dot = self.force_accumulator
        L_dot = self.torque_accumulator

        dx_dt = {}
        dx_dt['x_dot'] = np.concatenate((body_velocity, q_dot.q))
        dx_dt['p_dot'] = np.concatenate((P_dot, L_dot))
        return dx_dt
