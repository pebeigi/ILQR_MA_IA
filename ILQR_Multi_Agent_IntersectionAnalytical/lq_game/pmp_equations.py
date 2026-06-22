from __future__ import annotations
import numpy as np

def costate_update(A, Q, l, lambda_next, x, u=None):
    """One-step discrete PMP-style costate recursion.

    λ_k = Q x_k + l + Aᵀ λ_{k+1}
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    l = np.asarray(l, dtype=float).reshape(-1, 1)
    lam_next = np.asarray(lambda_next, dtype=float).reshape(-1, 1)
    return Q @ x + l + A.T @ lam_next

def stationarity_control(B, R, lambda_next, u_ref=None, r=None):
    """Solve the quadratic PMP stationarity condition for one player.

    0 = R (u-u_ref) + r + Bᵀ λ_{k+1}
    """
    g = B.T @ np.asarray(lambda_next, dtype=float).reshape(-1, 1)
    if r is not None:
        g = g + np.asarray(r, dtype=float).reshape(-1, 1)
    rhs = -g
    if u_ref is not None:
        rhs = rhs + R @ np.asarray(u_ref, dtype=float).reshape(-1, 1)
    return np.linalg.solve(R, rhs)
