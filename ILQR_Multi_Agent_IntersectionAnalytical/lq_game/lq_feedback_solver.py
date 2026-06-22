from __future__ import annotations
import numpy as np


def _as_col(v, n=None):
    """Return v as a float column vector."""
    arr = np.asarray(v, dtype=float).reshape(-1, 1)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"expected vector length {n}, got {arr.shape[0]}")
    return arr


def _regularize_gershgorin(S: np.ndarray, min_eval: float = 1e-3) -> np.ndarray:
    """Apply the Gershgorin-circle diagonal regularization used in ilqgames.

    The feedback Nash solve requires solving a coupled linear system S X = Y.
    Near-singular S matrices make the feedback gains unstable. The C++ solver
    optionally adds just enough diagonal weight so each Gershgorin lower bound
    is at least ``min_eval``.
    """
    S = np.array(S, dtype=float, copy=True)
    diagonal = np.diag(S).copy()
    radii = np.sum(np.abs(S), axis=0) - np.abs(diagonal)
    eval_lows = diagonal - radii
    mask = eval_lows < min_eval
    indices = np.flatnonzero(mask)
    S[indices, indices] += radii[mask] + min_eval - eval_lows[mask]
    return S


def solve_feedback_lq_game(
    As,
    Bs,
    Qs,
    ls,
    Rs,
    rs=None,
    *,
    adaptive_regularization: bool = True,
    regularization_min_eval: float = 1e-3,
    return_value_functions: bool = False,
):
    """Solve a finite-horizon LQ dynamic game for feedback Nash strategies.

    This is a NumPy adaptation of ``ilqgames-master``'s C++
    ``LQFeedbackSolver::Solve``.

    The local LQ dynamics are
        dx[k+1] = A[k] dx[k] + sum_i B[i][k] du_i[k]

    The returned strategies satisfy
        du_i[k] = -P_i[k] dx[k] - alpha_i[k]

    Parameters
    ----------
    As : list[np.ndarray]
        Discrete-time state Jacobians A[k]. Length is the number of trajectory
        samples. The final entry is kept for shape consistency and is not used
        in the backward feedback recursion.
    Bs : list[list[np.ndarray]]
        Bs[i][k] is player i's discrete-time control Jacobian at time k.
    Qs, ls : list[list[np.ndarray]]
        State Hessians and state gradients for each player's local quadratic
        cost at each time sample. The final entry acts as the terminal cost.
    Rs, rs : nested lists
        Rs[i][j][k] and rs[i][j][k] are player i's quadratic and linear cost
        terms with respect to player j's control.
    adaptive_regularization : bool
        Whether to regularize the coupled Nash matrix using the same
        Gershgorin-style rule as the C++ solver.

    Returns
    -------
    Ps, alphas : tuple
        Player-indexed lists of feedback gains and feedforward vectors.
    Optional third return value:
        ``(Zs, zetas)`` value-function arrays if ``return_value_functions`` is
        true.
    """
    num_steps = len(As)
    if num_steps == 0:
        raise ValueError("As must contain at least one time step")
    num_players = len(Bs)
    if num_players == 0:
        raise ValueError("Bs must contain at least one player")

    xdim = int(np.asarray(As[0]).shape[0])
    u_dims = [int(np.asarray(Bs[i][0]).shape[1]) for i in range(num_players)]
    total_udim = int(sum(u_dims))

    if rs is None:
        rs = [
            [[np.zeros((u_dims[j], 1), dtype=float) for _ in range(num_steps)] for j in range(num_players)]
            for _ in range(num_players)
        ]

    # Output strategies. The final time step has no outgoing dynamics, so the
    # final strategy remains zero just like the preallocated C++ Strategy.
    Ps = [[np.zeros((u_dims[i], xdim), dtype=float) for _ in range(num_steps)] for i in range(num_players)]
    alphas = [[np.zeros((u_dims[i], 1), dtype=float) for _ in range(num_steps)] for i in range(num_players)]

    # Value functions V_i(dx) = 0.5 dx.T Z_i dx + zeta_i.T dx + const.
    Zs = [[np.zeros((xdim, xdim), dtype=float) for _ in range(num_players)] for _ in range(num_steps)]
    zetas = [[np.zeros((xdim, 1), dtype=float) for _ in range(num_players)] for _ in range(num_steps)]

    # Terminal value functions come from the final quadraticized state cost.
    for i in range(num_players):
        Zs[-1][i] = np.asarray(Qs[i][-1], dtype=float).reshape(xdim, xdim)
        zetas[-1][i] = _as_col(ls[i][-1], xdim)

    # Work backward from the second-to-last sample, matching the C++ solver's
    # treatment of the last sample as terminal cost.
    for k in range(num_steps - 2, -1, -1):
        A = np.asarray(As[k], dtype=float).reshape(xdim, xdim)
        B = [np.asarray(Bs[i][k], dtype=float).reshape(xdim, u_dims[i]) for i in range(num_players)]

        S = np.zeros((total_udim, total_udim), dtype=float)
        Y = np.zeros((total_udim, xdim + 1), dtype=float)

        row0 = 0
        for i in range(num_players):
            row1 = row0 + u_dims[i]
            BiZi = B[i].T @ Zs[k + 1][i]

            col0 = 0
            for j in range(num_players):
                col1 = col0 + u_dims[j]
                if i == j:
                    Rij = np.asarray(Rs[i][i][k], dtype=float).reshape(u_dims[i], u_dims[i])
                    S[row0:row1, col0:col1] = BiZi @ B[i] + Rij
                else:
                    S[row0:row1, col0:col1] = BiZi @ B[j]
                col0 = col1

            # Solve for both feedback gains P and feedforward terms alpha using
            # one block linear system S X = Y, as in lq_feedback_solver.cpp.
            Y[row0:row1, :xdim] = BiZi @ A
            rii = _as_col(rs[i][i][k], u_dims[i])
            Y[row0:row1, xdim:] = B[i].T @ zetas[k + 1][i] + rii
            row0 = row1

        if adaptive_regularization:
            S = _regularize_gershgorin(S, regularization_min_eval)

        X, *_ = np.linalg.lstsq(S, Y, rcond=None)

        row0 = 0
        Pk = []
        alphak = []
        for i in range(num_players):
            row1 = row0 + u_dims[i]
            Pi = X[row0:row1, :xdim]
            ai = X[row0:row1, xdim:].reshape(u_dims[i], 1)
            Ps[i][k] = Pi
            alphas[i][k] = ai
            Pk.append(Pi)
            alphak.append(ai)
            row0 = row1

        # Closed-loop linearization dx+ = F dx + beta under Nash strategies.
        F = A.copy()
        beta = np.zeros((xdim, 1), dtype=float)
        for i in range(num_players):
            F -= B[i] @ Pk[i]
            beta -= B[i] @ alphak[i]

        # Riccati/value-function recursion for each player's cost-to-go.
        for i in range(num_players):
            Qi = np.asarray(Qs[i][k], dtype=float).reshape(xdim, xdim)
            li = _as_col(ls[i][k], xdim)
            Zi_next = Zs[k + 1][i]
            zeta_next = zetas[k + 1][i]

            Zi = F.T @ Zi_next @ F + Qi
            zeta = F.T @ (zeta_next + Zi_next @ beta) + li

            for j in range(num_players):
                Rij = np.asarray(Rs[i][j][k], dtype=float).reshape(u_dims[j], u_dims[j])
                rij = _as_col(rs[i][j][k], u_dims[j])
                Zi += Pk[j].T @ Rij @ Pk[j]
                zeta += Pk[j].T @ (Rij @ alphak[j] - rij)

            # Numerical symmetry helps keep later solves stable.
            Zs[k][i] = 0.5 * (Zi + Zi.T)
            zetas[k][i] = zeta

    if return_value_functions:
        return Ps, alphas, (Zs, zetas)
    return Ps, alphas


# Backward-compatible name for older scripts. In this feedback package this
# intentionally points to the feedback-Nash solver, not the old open-loop solver.
def solve_lq_game(*args, **kwargs):
    return solve_feedback_lq_game(*args, **kwargs)
