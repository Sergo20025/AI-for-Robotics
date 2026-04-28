"""Student solution: RK4 and re-exports from lib. Task 1.2."""
from typing import Callable

import numpy as np

from lib.phys.ode_solvers import (
    EulerMethod,
    ODESolverBase,
    SemiImplicitEulerMethod,
)


class RK4Method(ODESolverBase):
    """Classic 4th-order Runge-Kutta. Implement in solutions."""

    def ode(self, x_0, t_0, t_1, dx_dt_func: Callable):
        dt = t_1 - t_0

        def add_scaled(state, derivative, scale):
            return {
                "x": state["x"] + scale * derivative["x_dot"],
                "p": state["p"] + scale * derivative["p_dot"],
            }

        k1 = dx_dt_func(x_0)
        k2 = dx_dt_func(add_scaled(x_0, k1, 0.5 * dt))
        k3 = dx_dt_func(add_scaled(x_0, k2, 0.5 * dt))
        k4 = dx_dt_func(add_scaled(x_0, k3, dt))

        x_1 = x_0["x"] + (dt / 6.0) * (
            k1["x_dot"] + 2.0 * k2["x_dot"] + 2.0 * k3["x_dot"] + k4["x_dot"]
        )
        p_1 = x_0["p"] + (dt / 6.0) * (
            k1["p_dot"] + 2.0 * k2["p_dot"] + 2.0 * k3["p_dot"] + k4["p_dot"]
        )
        return {
            "x": np.asarray(x_1),
            "p": np.asarray(p_1),
        }


__all__ = ["EulerMethod", "SemiImplicitEulerMethod", "ODESolverBase", "RK4Method"]
