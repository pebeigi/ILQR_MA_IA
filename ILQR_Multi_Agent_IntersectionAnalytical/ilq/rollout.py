import numpy as np


def integrate(dynamics, t0, dt, x0, us, use_euler=False):
    """
    Integrate xdot = f(t, x, us) forward by dt using Euler or RK4.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    if use_euler:
        xdot = dynamics(t0, x0, us)
        return x0 + dt * np.asarray(xdot, dtype=float).reshape(-1)

    # RK4
    k1 = np.asarray(dynamics(t0, x0, us), dtype=float).reshape(-1)
    k2 = np.asarray(dynamics(t0 + 0.5 * dt, x0 + 0.5 * dt * k1, us), dtype=float).reshape(-1)
    k3 = np.asarray(dynamics(t0 + 0.5 * dt, x0 + 0.5 * dt * k2, us), dtype=float).reshape(-1)
    k4 = np.asarray(dynamics(t0 + dt, x0 + dt * k3, us), dtype=float).reshape(-1)

    return x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
