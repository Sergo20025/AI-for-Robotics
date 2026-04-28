"""Student solution: kinetic energy callback. Task 1.1."""
from typing import List

from lib.phys.physics_world import PhysicsEngine
from lib.phys.phys_objects import RigidBody


class KinEnergyCallback:
    def __init__(self, draw_graph=True):
        if draw_graph:
            import vpython as vp
            self.energy = vp.gcurve(color=vp.color.black, label="E<sub>kin</sub>")
        else:
            self.energy = None
        self.kin_e_history = []

    def __call__(self, phys_engine: PhysicsEngine):
        kin_e = self.calc_kinetic_energy(phys_engine.bodies)
        self.kin_e_history.append(kin_e)
        if self.energy is not None:
            self.energy.plot(phys_engine.time, kin_e)

    def calc_kinetic_energy(self, bodies: List[RigidBody]):
        total_energy = 0.0
        for body in bodies:
            if body.inv_mass == 0:
                continue
            mass = 1.0 / body.inv_mass
            total_energy += 0.5 * mass * float(body.body_velocity @ body.body_velocity)
            total_energy += 0.5 * float(body.angular_velocity @ body.angular_momentum)
        return total_energy
