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
    players = [Player(index=0, name="ego", state_dim=4, control_dim=2)]
    dynamics = ConcatenatedDynamics([unicycle_4d], [4], [2])
    pc = PlayerCost("ego", state_regularization=1e-6, control_regularization=1e-6)
    pc.add_cost(QuadraticStateCost(np.diag([1.0, 1.0, 0.1, 0.1]), x_ref=np.array([5.0, 0.0, 0.0, 3.0])))
    pc.add_cost(QuadraticControlCost(0, np.diag([0.2, 0.1])))
    game = GameDefinition(players=players, dynamics=dynamics, player_costs=[pc], x0=np.array([0.0, 0.0, 0.0, 2.0]), dt=0.1, horizon_steps=20)
    solver = ILQSolver(game)
    result = solver.solve()
    print("Final state:", result["xs"][-1])
    print("Total cost:", sum(result["costs"][0]))

if __name__ == "__main__":
    main()
