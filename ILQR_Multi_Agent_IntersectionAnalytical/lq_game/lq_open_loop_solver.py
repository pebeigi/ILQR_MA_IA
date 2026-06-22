from __future__ import annotations
import numpy as np
from collections import deque

def solve_lq_game(As, Bs, Qs, ls, Rs, rs=None):
    """Solve the local finite-horizon LQ game.

    For a single player this is the standard local LQR backward pass in
    dx/du coordinates. The optional rs argument contains local linear control
    gradients. Including rs is important because quadratic control costs have
    gradient R @ u_ref around a nonzero nominal control sequence.
    """
    horizon = len(As) - 1
    num_players = len(Bs)
    x_dim = As[0].shape[0]
    u_dims = [Bis[0].shape[1] for Bis in Bs]

    if rs is None:
        rs = [[ [np.zeros((u_dims[j], 1)) for _ in range(len(As))] for j in range(num_players)]
              for _ in range(num_players)]

    # Initialize value functions to zero so each Q[k] is used exactly once
    # as a running cost. The last Q (k=horizon) serves as the effective
    # terminal cost via the Riccati update at k=horizon with Z=0.
    Zs = [deque([np.zeros_like(Qis[-1])]) for Qis in Qs]
    zetas = [deque([np.zeros_like(lis[-1])]) for lis in ls]
    Ps = [deque() for _ in range(num_players)]
    alphas = [deque() for _ in range(num_players)]

    for k in range(horizon, -1, -1):
        A = As[k]
        B = [Bis[k] for Bis in Bs]
        Q = [Qis[k] for Qis in Qs]
        l = [lis[k] for lis in ls]
        R = [[Rijs[k] for Rijs in Ris] for Ris in Rs]
        r = [[rijs[k] for rijs in ris] for ris in rs]
        Z = [Zis[0] for Zis in Zs]
        zeta = [zetais[0] for zetais in zetas]

        S_rows = []
        for ii in range(num_players):
            row_blocks = []
            for jj in range(num_players):
                if jj == ii:
                    row_blocks.append(R[ii][ii] + B[ii].T @ Z[ii] @ B[ii])
                else:
                    row_blocks.append(B[ii].T @ Z[ii] @ B[jj])
            S_rows.append(np.concatenate(row_blocks, axis=1))
        S = np.concatenate(S_rows, axis=0)

        Y = np.concatenate([B[ii].T @ Z[ii] @ A for ii in range(num_players)], axis=0)
        P, *_ = np.linalg.lstsq(S, Y, rcond=None)
        P_split = np.split(P, np.cumsum(u_dims[:-1]), axis=0)
        for ii in range(num_players):
            Ps[ii].appendleft(P_split[ii])

        F = A - sum(B[ii] @ P_split[ii] for ii in range(num_players))
        for ii in range(num_players):
            Zs[ii].appendleft(
                F.T @ Z[ii] @ F
                + Q[ii]
                + sum(P_split[jj].T @ R[ii][jj] @ P_split[jj] for jj in range(num_players))
            )

        # Local affine control term. For a single player this is
        # alpha = (R + B.T Z B)^-1 (B.T zeta + r).
        Y_alpha = np.concatenate([B[ii].T @ zeta[ii] + r[ii][ii] for ii in range(num_players)], axis=0)
        alpha, *_ = np.linalg.lstsq(S, Y_alpha, rcond=None)
        alpha_split = np.split(alpha, np.cumsum(u_dims[:-1]), axis=0)
        beta = -sum(B[ii] @ alpha_split[ii] for ii in range(num_players))
        for ii in range(num_players):
            alphas[ii].appendleft(alpha_split[ii])
            zetas[ii].appendleft(
                F.T @ (zeta[ii] + Z[ii] @ beta)
                + l[ii]
                + sum(P_split[jj].T @ R[ii][jj] @ alpha_split[jj] for jj in range(num_players))
                - sum(P_split[jj].T @ r[ii][jj] for jj in range(num_players))
            )
    return [list(Pis) for Pis in Ps], [list(ais) for ais in alphas]
