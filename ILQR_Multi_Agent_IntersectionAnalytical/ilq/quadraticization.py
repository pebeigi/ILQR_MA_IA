from __future__ import annotations
from typing import Sequence
import numpy as np

try:
    from ..costs.player_cost import PlayerCost
except ImportError:  # pragma: no cover
    from costs.player_cost import PlayerCost

def quadraticize_player_costs(player_costs: Sequence[PlayerCost], x, us, k=None):
    """Quadraticize all players' costs at a single operating point."""
    approximations = [pc.quadraticize(x, us, k, include_const=False) for pc in player_costs]
    return approximations

def pack_for_lq_game(player_costs: Sequence[PlayerCost], x_traj, u_traj_by_player):
    """Convert stage costs to local LQ terms around the nominal trajectory.

    The LQ solver works in local coordinates:
        dx = x - x_ref,    du_i = u_i - u_i_ref

    Therefore the linear terms must be gradients evaluated at the nominal
    trajectory, not the global affine q/r terms from the quadratic cost object.
    This is the key change that makes the ILQR update respond correctly when
    a cost coefficient, such as obstacle_weight, is changed.
    """
    num_players = len(player_costs)
    horizon = len(x_traj)
    Qs = [[] for _ in range(num_players)]
    ls = [[] for _ in range(num_players)]
    Rs = [[[] for _ in range(num_players)] for _ in range(num_players)]
    rs = [[[] for _ in range(num_players)] for _ in range(num_players)]

    for k in range(horizon):
        xk = np.asarray(x_traj[k], dtype=float).reshape(-1)
        us_k = [np.asarray(u_traj_by_player[i][k], dtype=float).reshape(-1) for i in range(num_players)]
        approximations = quadraticize_player_costs(player_costs, xk, us_k, k)
        for i, qa in enumerate(approximations):
            Qs[i].append(qa.Q)
            # local linear state term = gradient wrt x at x_ref
            ls[i].append((qa.Q @ xk + qa.q).reshape(-1, 1))
            for j in range(num_players):
                Rjj = qa.R[j]
                rj = qa.r[j]
                Rs[i][j].append(Rjj)
                # local linear control term = gradient wrt u_j at u_j_ref
                rs[i][j].append((Rjj @ us_k[j] + rj).reshape(-1, 1))
    return Qs, ls, Rs, rs
