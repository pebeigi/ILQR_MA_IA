import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from costs.base_cost import QuadraticStateCost, QuadraticControlCost
from costs.player_cost import PlayerCost
from dynamics.player_dynamics import unicycle_4d
from dynamics.base_dynamics import ConcatenatedDynamics
from game.player import Player
from game.game_definition import GameDefinition
from ilq.ilq_solver import ILQSolver

def main():
    players = [
        Player(index=0, name="p1", state_dim=4, control_dim=2),
        Player(index=1, name="p2", state_dim=4, control_dim=2),
    ]
    dynamics = ConcatenatedDynamics([unicycle_4d, unicycle_4d], [4, 4], [2, 2])
    pc1 = PlayerCost("p1", state_regularization=1e-6, control_regularization=1e-6)
    pc2 = PlayerCost("p2", state_regularization=1e-6, control_regularization=1e-6)
    pc1.add_cost(QuadraticStateCost(np.eye(8), x_ref=np.array([5, 0, 0, 3, 0, 0, 0, 0], dtype=float)))
    pc2.add_cost(QuadraticStateCost(np.eye(8), x_ref=np.array([0, 0, 0, 0, -5, 0, 0, 3], dtype=float)))
    pc1.add_cost(QuadraticControlCost(0, np.eye(2) * 0.1))
    pc2.add_cost(QuadraticControlCost(1, np.eye(2) * 0.1))
    game = GameDefinition(players=players, dynamics=dynamics, player_costs=[pc1, pc2], x0=np.array([0,0,0,2, 0,1,0,2], dtype=float), dt=0.1, horizon_steps=15)
    solver = ILQSolver(game)
    result = solver.solve()
    print("Final combined state:", result["xs"][-1])
    print("Player costs:", [sum(c) for c in result["costs"]])

if __name__ == "__main__":
    main()
