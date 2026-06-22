import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from costs.base_cost import (
    AgentLeaderYieldLineCost,
    AgentLaneKeepingCost,
    AgentProximitySpeedCost,
    AgentSpeedDependentControlCost,
    _c2_quadratic_hinge,
    _smooth_floor,
    _smootherstep01,
)
from dynamics.base_dynamics import ConcatenatedDynamics
from dynamics.player_dynamics import vehicle_4d
from ilq.linearization import linearize_discrete
from ilq.rollout import integrate


def _fd_grad_x(cost, x, us, k=None, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x, dtype=float)
        dx[i] = eps
        grad[i] = (cost.evaluate(x + dx, us, k) - cost.evaluate(x - dx, us, k)) / (2.0 * eps)
    return grad


def _fd_grad_u(cost, x, us, player, k=None, eps=1e-6):
    grad = np.zeros_like(us[player], dtype=float)
    for i in range(grad.size):
        up = [np.asarray(u, dtype=float).copy() for u in us]
        um = [np.asarray(u, dtype=float).copy() for u in us]
        up[player][i] += eps
        um[player][i] -= eps
        grad[i] = (cost.evaluate(x, up, k) - cost.evaluate(x, um, k)) / (2.0 * eps)
    return grad


def _fd_discrete_map(dynamics, x, us, dt, use_euler, eps=1e-6):
    nx = x.size
    A = np.zeros((nx, nx), dtype=float)
    B = [np.zeros((nx, u.size), dtype=float) for u in us]
    for i in range(nx):
        dx = np.zeros_like(x, dtype=float)
        dx[i] = eps
        fp = integrate(dynamics, 0.0, dt, x + dx, us, use_euler=use_euler)
        fm = integrate(dynamics, 0.0, dt, x - dx, us, use_euler=use_euler)
        A[:, i] = (fp - fm) / (2.0 * eps)
    for player, u in enumerate(us):
        for i in range(u.size):
            up = [np.asarray(ui, dtype=float).copy() for ui in us]
            um = [np.asarray(ui, dtype=float).copy() for ui in us]
            up[player][i] += eps
            um[player][i] -= eps
            fp = integrate(dynamics, 0.0, dt, x, up, use_euler=use_euler)
            fm = integrate(dynamics, 0.0, dt, x, um, use_euler=use_euler)
            B[player][:, i] = (fp - fm) / (2.0 * eps)
    return A, B


def test_speed_dependent_control_cost_matches_local_gradient():
    cost = AgentSpeedDependentControlCost(
        agent_index=0,
        state_dim_per_agent=4,
        player_index=0,
        beta_2=3.0,
        beta_3=7.0,
        v_min=1.0,
    )
    x = np.array([1.2, -0.4, 5.0, 0.7], dtype=float)
    us = [np.array([0.13, -0.8], dtype=float)]

    qa = cost.quadraticize(x, us, k=4)
    approx_x_grad = qa.Q @ x + qa.q
    approx_u_grad = qa.R[0] @ us[0] + qa.r[0]

    np.testing.assert_allclose(approx_x_grad, _fd_grad_x(cost, x, us, k=4), atol=1e-5, rtol=1e-7)
    np.testing.assert_allclose(approx_u_grad, _fd_grad_u(cost, x, us, 0, k=4), atol=1e-5, rtol=1e-7)


def test_discrete_linearization_matches_rk4_rollout_when_requested():
    dynamics = ConcatenatedDynamics([vehicle_4d], [4], [2]).evaluate
    x = np.array([1.2, -0.7, 5.5, 0.8], dtype=float)
    us = [np.array([0.12, -1.3], dtype=float)]
    dt = 0.1

    A, B = linearize_discrete(dynamics, 0.0, x, us, dt, use_euler=False)
    A_fd, B_fd = _fd_discrete_map(dynamics, x, us, dt, use_euler=False)

    np.testing.assert_allclose(A, A_fd, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(B[0], B_fd[0], atol=1e-8, rtol=1e-8)


def test_analytical_linearization_does_not_silently_fall_back():
    def dynamics_without_jacobians(t, x, us):
        return np.zeros_like(x)

    with pytest.raises(NotImplementedError, match="Analytical linearization requires"):
        linearize_discrete(
            dynamics_without_jacobians,
            0.0,
            np.zeros(4),
            [np.zeros(2)],
            0.1,
            use_euler=True,
        )


def test_lane_keeping_cost_matches_local_gradient():
    cost = AgentLaneKeepingCost(
        agent_index=0,
        state_dim_per_agent=4,
        lane_x=16.5,
        weight=30.0,
    )
    x = np.array([18.0, 6.2, 5.0, 0.7], dtype=float)
    us = [np.array([0.0, 0.0], dtype=float)]

    qa = cost.quadraticize(x, us, k=4)
    approx_x_grad = qa.Q @ x + qa.q

    np.testing.assert_allclose(approx_x_grad, _fd_grad_x(cost, x, us, k=4), atol=1e-5, rtol=1e-7)


def test_leader_yield_line_cost_matches_local_gradient_and_is_psd():
    cost = AgentLeaderYieldLineCost(
        agent_index=1,
        leader_index=0,
        state_dim_per_agent=4,
        hold_axis=1,
        hold_value=5.0,
        hold_direction=1.0,
        clear_axis=0,
        clear_value=8.0,
        clear_direction=1.0,
        weight=30.0,
        transition_width=0.5,
        clearance_width=2.5,
    )
    x = np.array([10.0, 12.0, 2.0, 1.2, 13.5, 6.2, 4.0, 1.4], dtype=float)
    us = [np.array([0.0, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float)]

    qa = cost.quadraticize(x, us, k=4)
    approx_x_grad = qa.Q @ x + qa.q

    np.testing.assert_allclose(approx_x_grad, _fd_grad_x(cost, x, us, k=4), atol=1e-5, rtol=1e-7)
    assert np.linalg.eigvalsh(qa.Q).min() >= -1e-10


def test_proximity_speed_cost_matches_local_gradient_in_smooth_transition():
    cost = AgentProximitySpeedCost(
        agent_index=1,
        other_agent_index=0,
        state_dim_per_agent=4,
        weight=2000.0,
        epsilon=8.0,
        speed_transition_width=0.1,
    )
    # Agent 1 speed is inside the smooth floor transition [0, 0.1], which is
    # the nontrivial splice used instead of max(speed, 0).
    x = np.array([11.8, 14.9, 3.0, 2.5, 12.7, 12.5, 0.04, 2.3], dtype=float)
    us = [np.array([0.0, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float)]

    qa = cost.quadraticize(x, us, k=4)
    approx_x_grad = qa.Q @ x + qa.q

    np.testing.assert_allclose(approx_x_grad, _fd_grad_x(cost, x, us, k=4), atol=1e-4, rtol=1e-6)
    assert np.linalg.eigvalsh(qa.Q).min() >= -1e-10


def test_gated_proximity_speed_cost_matches_local_gradient():
    cost = AgentProximitySpeedCost(
        agent_index=1,
        other_agent_index=0,
        state_dim_per_agent=4,
        weight=1500.0,
        epsilon=8.0,
        activation_distance=8.5,
        activation_width=2.5,
    )
    # The pair distance is inside the gate transition, so this checks the
    # position derivative from the C2 distance gate as well as the inverse term.
    x = np.array([9.0, 10.0, 5.5, 1.2, 16.5, 13.2, 4.0, 1.4], dtype=float)
    us = [np.array([0.0, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float)]

    qa = cost.quadraticize(x, us, k=4)
    approx_x_grad = qa.Q @ x + qa.q

    np.testing.assert_allclose(approx_x_grad, _fd_grad_x(cost, x, us, k=4), atol=1e-4, rtol=1e-6)
    assert np.linalg.eigvalsh(qa.Q).min() >= -1e-10


def test_smooth_hinge_and_gate_are_c2_at_splice_points():
    weight = 7.0
    width = 0.5

    value0, grad0, hess0 = _c2_quadratic_hinge(0.0, weight, width)
    np.testing.assert_allclose([value0, grad0, hess0], [0.0, 0.0, 0.0], atol=1e-12)

    value1, grad1, hess1 = _c2_quadratic_hinge(width, weight, width)
    np.testing.assert_allclose(value1, 0.5 * weight * width ** 2, atol=1e-12)
    np.testing.assert_allclose(grad1, weight * width, atol=1e-12)
    np.testing.assert_allclose(hess1, weight, atol=1e-12)

    gate0, dgate0 = _smootherstep01(0.0)
    gate1, dgate1 = _smootherstep01(1.0)
    np.testing.assert_allclose([gate0, dgate0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose([gate1, dgate1], [1.0, 0.0], atol=1e-12)

    floor0, dfloor0, hfloor0 = _smooth_floor(0.0, 0.0, width)
    floor1, dfloor1, hfloor1 = _smooth_floor(width, 0.0, width)
    np.testing.assert_allclose([floor0, dfloor0, hfloor0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose([floor1, dfloor1, hfloor1], [width, 1.0, 0.0], atol=1e-12)
