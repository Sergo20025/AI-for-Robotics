"""Student solution: impulse-based constraint manager. Task 2.1."""
from dataclasses import dataclass

from scipy.sparse.linalg import cg

from lib.phys.constraints.manager import ConstraintsManager


@dataclass
class ConstraintsManagerImpulseBased(ConstraintsManager):
    beta_baumgarte: float = 0.5

    def calc_forces(self, t_0, t_1):
        if len(self.constraints) == 0:
            return None

        dt = t_1 - t_0
        self.update_constraints(t_0, t_1)
        self.update_W()
        self.update_J()

        W = self.W
        J = self.J
        C = self.get_C()
        V_0 = self.get_V()
        F = self.get_F()

        V_1 = V_0 + W @ F * dt
        K = J @ W @ J.T
        rhs = -J @ V_1 - (self.beta_baumgarte / dt) * C
        l, exit_code = cg(K, rhs, atol=1e-5)
        assert exit_code == 0
        F_C = J.T @ l / dt
        return F_C
