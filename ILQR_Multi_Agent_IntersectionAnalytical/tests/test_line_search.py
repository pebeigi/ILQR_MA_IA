import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ilq.ilq_solver import ILQSolver


class FakeGame:
    num_players = 1
    horizon_steps = 1
    state_dim = 1
    control_dims = [1]
    x0 = np.array([0.0])


def test_line_search_accepts_cost_improvement_before_delta_contraction():
    solver = ILQSolver(
        FakeGame(),
        alpha_scaling=0.5,
        alpha_line_search=True,
        alpha_line_search_min=0.25,
        alpha_line_search_shrink=0.5,
        alpha_line_search_max_growth=5.0,
    )
    solver.current_operating_point = (
        [np.array([0.0])],
        [[np.array([0.0])]],
        [[100.0]],
    )

    def rollout(alpha_scaling=None, Ps=None, alphas=None):
        if alpha_scaling == 0.5:
            return [np.array([2.0])], [[np.array([0.0])]], [[90.0]]
        if alpha_scaling == 0.25:
            return [np.array([0.2])], [[np.array([0.0])]], [[110.0]]
        raise AssertionError(f"unexpected alpha {alpha_scaling}")

    solver._rollout_with_strategy = rollout

    alpha, info = solver._choose_alpha_scaling(previous_max_delta=1.0, iteration=0)

    assert alpha == 0.5
    assert info["reason"] == "merit_decrease"
    assert info["max_delta_x"] == 2.0
    assert info["merit"] == 90.0
    assert solver._pending_rollout[0][0][0] == 2.0
