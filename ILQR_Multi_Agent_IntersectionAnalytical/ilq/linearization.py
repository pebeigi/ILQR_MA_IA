from __future__ import annotations
import numpy as np


def _analytical_model(dynamics):
    """Resolve a dynamics object from either the object or its bound evaluate method."""
    if hasattr(dynamics, "evaluate") and hasattr(dynamics, "jacobians"):
        return dynamics
    owner = getattr(dynamics, "__self__", None)
    if owner is not None and hasattr(owner, "evaluate") and hasattr(owner, "jacobians"):
        return owner
    return None


def analytical_discrete_jacobians(dynamics, t, x, us, dt: float, use_euler=False):
    """Differentiate the actual Euler or RK4 step using exact dynamics Jacobians."""
    model = _analytical_model(dynamics)
    if model is None:
        raise NotImplementedError("dynamics does not provide analytical Jacobians")

    x = np.asarray(x, dtype=float).reshape(-1)
    us = [np.asarray(u, dtype=float).reshape(-1) for u in us]
    nx = x.size
    identity = np.eye(nx, dtype=float)

    if use_euler:
        A, B = model.jacobians(t, x, us)
        return identity + dt * A, [dt * Bi for Bi in B]

    # Propagate sensitivities through the exact RK4 computation used by rollout.
    k1 = model.evaluate(t, x, us)
    A1, B1 = model.jacobians(t, x, us)
    K1x = A1
    K1u = B1

    x2 = x + 0.5 * dt * k1
    X2x = identity + 0.5 * dt * K1x
    X2u = [0.5 * dt * Ki for Ki in K1u]
    k2 = model.evaluate(t + 0.5 * dt, x2, us)
    A2, B2 = model.jacobians(t + 0.5 * dt, x2, us)
    K2x = A2 @ X2x
    K2u = [A2 @ X2u[i] + B2[i] for i in range(len(us))]

    x3 = x + 0.5 * dt * k2
    X3x = identity + 0.5 * dt * K2x
    X3u = [0.5 * dt * Ki for Ki in K2u]
    k3 = model.evaluate(t + 0.5 * dt, x3, us)
    A3, B3 = model.jacobians(t + 0.5 * dt, x3, us)
    K3x = A3 @ X3x
    K3u = [A3 @ X3u[i] + B3[i] for i in range(len(us))]

    x4 = x + dt * k3
    X4x = identity + dt * K3x
    X4u = [dt * Ki for Ki in K3u]
    A4, B4 = model.jacobians(t + dt, x4, us)
    K4x = A4 @ X4x
    K4u = [A4 @ X4u[i] + B4[i] for i in range(len(us))]

    Ad = identity + (dt / 6.0) * (K1x + 2.0 * K2x + 2.0 * K3x + K4x)
    Bd = [
        (dt / 6.0) * (K1u[i] + 2.0 * K2u[i] + 2.0 * K3u[i] + K4u[i])
        for i in range(len(us))
    ]
    return Ad, Bd


def finite_difference_jacobians(dynamics, t, x, us, eps: float = 1e-6):
    """Linearize continuous-time multi-player dynamics by finite differences.

    Returns ``A, B`` such that ``xdot ≈ f(x0,u0) + A dx + Σ_i B[i] du_i``.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    us = [np.asarray(u, dtype=float).reshape(-1) for u in us]
    f0 = np.asarray(dynamics(t, x, us), dtype=float).reshape(-1)
    nx = x.size
    A = np.zeros((nx, nx), dtype=float)
    B = [np.zeros((nx, u.size), dtype=float) for u in us]

    for j in range(nx):
        dx = np.zeros(nx)
        dx[j] = eps
        fp = np.asarray(dynamics(t, x + dx, us), dtype=float).reshape(-1)
        fm = np.asarray(dynamics(t, x - dx, us), dtype=float).reshape(-1)
        A[:, j] = (fp - fm) / (2.0 * eps)

    for i, u in enumerate(us):
        for j in range(u.size):
            dup = [ui.copy() for ui in us]
            dum = [ui.copy() for ui in us]
            dup[i][j] += eps
            dum[i][j] -= eps
            fp = np.asarray(dynamics(t, x, dup), dtype=float).reshape(-1)
            fm = np.asarray(dynamics(t, x, dum), dtype=float).reshape(-1)
            B[i][:, j] = (fp - fm) / (2.0 * eps)

    return A, B, f0

def discretize_linear_system(A, B, dt: float):
    A = np.asarray(A, dtype=float)
    Ad = np.eye(A.shape[0]) + dt * A
    Bd = [dt * np.asarray(Bi, dtype=float) for Bi in B]
    return Ad, Bd

def finite_difference_discrete_jacobians(dynamics, t, x, us, dt: float, use_euler=False, eps: float = 1e-6):
    """Linearize the actual one-step discrete rollout map by finite differences."""
    try:
        from .rollout import integrate
    except ImportError:  # pragma: no cover
        from ilq.rollout import integrate

    x = np.asarray(x, dtype=float).reshape(-1)
    us = [np.asarray(u, dtype=float).reshape(-1) for u in us]
    nx = x.size
    A = np.zeros((nx, nx), dtype=float)
    B = [np.zeros((nx, u.size), dtype=float) for u in us]

    for j in range(nx):
        dx = np.zeros(nx)
        dx[j] = eps
        fp = integrate(dynamics, t, dt, x + dx, us, use_euler=use_euler)
        fm = integrate(dynamics, t, dt, x - dx, us, use_euler=use_euler)
        A[:, j] = (fp - fm) / (2.0 * eps)

    for i, u in enumerate(us):
        for j in range(u.size):
            dup = [ui.copy() for ui in us]
            dum = [ui.copy() for ui in us]
            dup[i][j] += eps
            dum[i][j] -= eps
            fp = integrate(dynamics, t, dt, x, dup, use_euler=use_euler)
            fm = integrate(dynamics, t, dt, x, dum, use_euler=use_euler)
            B[i][:, j] = (fp - fm) / (2.0 * eps)

    return A, B

def linearize_discrete(dynamics, t, x, us, dt: float, eps: float = 1e-6, use_euler=True):
    """Linearize a discrete step analytically.

    Finite-difference helpers remain available for derivative validation, but
    the Analytical solver never silently falls back to them.
    """
    if _analytical_model(dynamics) is None:
        raise NotImplementedError(
            "Analytical linearization requires dynamics with a jacobians(t, x, us) method"
        )
    return analytical_discrete_jacobians(dynamics, t, x, us, dt, use_euler=use_euler)
