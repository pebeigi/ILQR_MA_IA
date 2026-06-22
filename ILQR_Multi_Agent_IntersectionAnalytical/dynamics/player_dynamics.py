import numpy as np


def unicycle_4d(x, u):
    """Original template dynamics.

    State:   [px, py, theta, v]
    Control: [omega, a]
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    px, py, theta, v = x
    omega, a = u
    return np.array([v * np.cos(theta), v * np.sin(theta), omega, a], dtype=float)


def unicycle_4d_jacobians(x, u):
    """Exact continuous-time Jacobians for :func:`unicycle_4d`."""
    x = np.asarray(x, dtype=float).reshape(-1)
    _, _, theta, v = x

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    A = np.zeros((4, 4), dtype=float)
    A[0, 2] = -v * sin_theta
    A[0, 3] = cos_theta
    A[1, 2] = v * cos_theta
    A[1, 3] = sin_theta

    B = np.zeros((4, 2), dtype=float)
    B[2, 0] = 1.0
    B[3, 1] = 1.0
    return A, B


def vehicle_4d(x, u):
    """Single-vehicle dynamics matching the LQR Game notebook.

    State:   [px, py, speed, heading]
             heading = 0 → +x, pi/2 → +y.
    Control: [kappa, a]
             kappa = curvature (rad/m) = heading_rate / speed
             a     = longitudinal acceleration (m/s²)

    Equations:
        px_dot      = speed * cos(heading)
        py_dot      = speed * sin(heading)
        speed_dot   = a
        heading_dot = speed * kappa
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    px, py, speed, heading = x
    kappa, a = u

    return np.array([
        speed * np.cos(heading),
        speed * np.sin(heading),
        a,
        speed * kappa,
    ], dtype=float)


def vehicle_4d_jacobians(x, u):
    """Exact continuous-time Jacobians for :func:`vehicle_4d`."""
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    _, _, speed, heading = x
    kappa, _ = u

    sin_heading = np.sin(heading)
    cos_heading = np.cos(heading)
    A = np.zeros((4, 4), dtype=float)
    A[0, 2] = cos_heading
    A[0, 3] = -speed * sin_heading
    A[1, 2] = sin_heading
    A[1, 3] = speed * cos_heading
    A[3, 2] = kappa

    B = np.zeros((4, 2), dtype=float)
    B[2, 1] = 1.0
    B[3, 0] = speed
    return A, B


# ConcatenatedDynamics discovers exact subsystem derivatives through this
# lightweight function attribute while preserving the existing callable API.
unicycle_4d.jacobians = unicycle_4d_jacobians
vehicle_4d.jacobians = vehicle_4d_jacobians
