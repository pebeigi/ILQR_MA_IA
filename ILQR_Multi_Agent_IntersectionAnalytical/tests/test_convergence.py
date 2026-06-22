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


def test_convergence_metrics_use_max_delta_x_and_relative_total_cost():
    solver = ILQSolver(FakeGame())
    solver.last_operating_point = ([np.array([0.0]), np.array([1.0])], None, None)
    solver.current_operating_point = ([np.array([0.005]), np.array([1.008])], None, None)

    max_delta_x, relative_delta_cost = solver._convergence_metrics(1000.0, 1000.0005)

    np.testing.assert_allclose(max_delta_x, 0.008)
    np.testing.assert_allclose(relative_delta_cost, 5e-7)


def test_convergence_configuration_rejects_invalid_tolerances_and_patience():
    for kwargs in (
        {"convergence_tol": 0.0},
        {"relative_cost_convergence_tol": 0.0},
        {"convergence_patience": 0},
    ):
        try:
            ILQSolver(FakeGame(), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_convergence_requires_three_consecutive_qualifying_iterations():
    solver = ILQSolver(
        FakeGame(),
        convergence_tol=1e-2,
        relative_cost_convergence_tol=1e-6,
        convergence_patience=3,
    )

    streak = 0
    streak = solver._update_convergence_streak(streak, 0.009, 9e-7)
    assert streak == 1
    streak = solver._update_convergence_streak(streak, 0.008, 8e-7)
    assert streak == 2
    assert streak < solver.convergence_patience
    streak = solver._update_convergence_streak(streak, 0.007, 7e-7)
    assert streak == solver.convergence_patience

    assert solver._update_convergence_streak(streak, 0.011, 1e-7) == 0
    assert solver._update_convergence_streak(streak, 0.001, 2e-6) == 0
