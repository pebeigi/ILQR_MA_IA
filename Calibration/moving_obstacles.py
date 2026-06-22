"""Time-varying (moving) obstacle costs for replayed neighbor agents.

In multi-agent calibration the non-ego agents follow their REAL observed
trajectories, so from the ego's point of view they are moving obstacles whose
positions are known constants at each timestep ``k``.  This makes the costs
single-player (they depend only on the ego state and ``k``), which keeps the
ILQR problem a one-player optimization.

Two costs mirror the ILQR package's static analogues:

* :class:`MovingObstacleRepulsionCost`  ~ ``BatchedStaticObstacleRepulsionCost``
  but with per-step obstacle sets.
* :class:`MovingProximitySpeedCost`      ~ ``AgentProximitySpeedCost`` but the
  other position is a constant per step (no second-player block).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ILQR_ROOT = PROJECT_ROOT / "ILQR_Multi_Agent_IntersectionAnalytical"
if str(ILQR_ROOT) not in sys.path:
    sys.path.insert(0, str(ILQR_ROOT))

from costs.base_cost import BaseCost, QuadraticApprox, _smooth_floor, _smootherstep01  # noqa: E402


def _step_positions(positions_per_step, k) -> np.ndarray:
    """Return the (M, 2) obstacle positions active at step ``k`` (possibly empty)."""
    if k is None:
        return np.empty((0, 2))
    if k < 0 or k >= len(positions_per_step):
        return np.empty((0, 2))
    pos = positions_per_step[k]
    if pos is None or len(pos) == 0:
        return np.empty((0, 2))
    return np.asarray(pos, dtype=float).reshape(-1, 2)


class MovingObstacleRepulsionCost(BaseCost):
    """Repulsion from the ego to all neighbor positions present at step ``k``.

    Cost at step k:  sum_j  weight / (||p_ego - p_j(k)||^2 + epsilon)
    """

    def __init__(
        self,
        positions_per_step,
        agent_index: int = 0,
        state_dim_per_agent: int = 4,
        weight: float = 1000.0,
        epsilon: float = 1.0,
    ):
        self.positions_per_step = positions_per_step
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        start = self.agent_index * self.state_dim_per_agent
        self._idx = np.array([start, start + 1], dtype=int)

    def evaluate(self, x, us, k=None):
        if self.weight == 0.0:
            return 0.0
        obstacles = _step_positions(self.positions_per_step, k)
        if obstacles.shape[0] == 0:
            return 0.0
        p = np.asarray(x, dtype=float).reshape(-1)[self._idx]
        z = np.sum((p - obstacles) ** 2, axis=1) + self.epsilon
        return float(np.sum(self.weight / z))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        obstacles = _step_positions(self.positions_per_step, k)
        if self.weight == 0.0 or obstacles.shape[0] == 0:
            return QuadraticApprox(Q=Q, q=q, const=0.0)

        idx = self._idx
        p = x[idx]
        deltas = p - obstacles
        z = np.sum(deltas ** 2, axis=1) + self.epsilon
        f0 = float(np.sum(self.weight / z))
        grad_pos = np.sum(-2.0 * self.weight * deltas / (z ** 2)[:, None], axis=0)
        total_curvature = float(np.sum(np.maximum(1e-8, 2.0 * self.weight / z ** 2)))
        H = total_curvature * np.eye(2)

        sl = slice(idx[0], idx[0] + 2)
        Q[sl, sl] = H
        q[idx] = grad_pos - H @ p
        const = f0 - grad_pos @ p + 0.5 * p @ H @ p
        return QuadraticApprox(Q=Q, q=q, const=float(const))


class MovingProximitySpeedCost(BaseCost):
    """Slow the ego when close to a neighbor (yielding), neighbor fixed per step.

    Cost at step k:  sum_j  weight * gate(dist_j) * smooth_max(v_ego, 0)
                            / (||p_ego - p_j(k)||^2 + epsilon)
    """

    def __init__(
        self,
        positions_per_step,
        agent_index: int = 0,
        state_dim_per_agent: int = 4,
        weight: float = 0.0,
        epsilon: float = 2.0,
        speed_transition_width: float = 0.1,
        activation_distance: float | None = None,
        activation_width: float = 2.0,
    ):
        self.positions_per_step = positions_per_step
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        self.speed_transition_width = float(speed_transition_width)
        self.activation_distance = None if activation_distance is None else float(activation_distance)
        self.activation_width = float(activation_width)
        start = self.agent_index * self.state_dim_per_agent
        self._idx = np.array([start, start + 1], dtype=int)
        self._speed_idx = start + 2

    def _gate_and_grad(self, dist_sq: float):
        if self.activation_distance is None:
            return 1.0, 0.0
        outer_sq = self.activation_distance ** 2
        inner = max(0.0, self.activation_distance - self.activation_width)
        inner_sq = inner ** 2
        width_sq = outer_sq - inner_sq
        if width_sq <= 1e-12 or dist_sq <= inner_sq:
            return 1.0, 0.0
        if dist_sq >= outer_sq:
            return 0.0, 0.0
        s = (outer_sq - dist_sq) / width_sq
        gate, dgate_ds = _smootherstep01(s)
        return gate, -dgate_ds / width_sq

    def evaluate(self, x, us, k=None):
        if self.weight == 0.0:
            return 0.0
        obstacles = _step_positions(self.positions_per_step, k)
        if obstacles.shape[0] == 0:
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)
        v, _, _ = _smooth_floor(float(x[self._speed_idx]), 0.0, self.speed_transition_width)
        total = 0.0
        p = x[self._idx]
        for o in obstacles:
            delta = p - o
            dist_sq = float(delta @ delta)
            gate, _ = self._gate_and_grad(dist_sq)
            total += self.weight * gate * v / (dist_sq + self.epsilon)
        return float(total)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        obstacles = _step_positions(self.positions_per_step, k)
        if self.weight == 0.0 or obstacles.shape[0] == 0:
            return QuadraticApprox(Q=Q, q=q, const=0.0)

        idx = self._idx
        isp = self._speed_idx
        p = x[idx]
        v, speed_grad, _ = _smooth_floor(float(x[isp]), 0.0, self.speed_transition_width)

        f0 = 0.0
        grad = np.zeros(nx, dtype=float)
        curvature = 0.0
        for o in obstacles:
            delta = p - o
            dist_sq = float(delta @ delta)
            z = dist_sq + self.epsilon
            gate, dgate_ddist_sq = self._gate_and_grad(dist_sq)
            f0 += self.weight * gate * v / z
            grad[isp] += self.weight * gate * speed_grad / z
            grad_pi = -2.0 * self.weight * gate * v * delta / (z ** 2)
            grad_pi += self.weight * v * (2.0 * dgate_ddist_sq * delta) / z
            grad[idx] += grad_pi
            curvature += max(1e-8, 2.0 * self.weight * max(gate, 0.0) * v / (z ** 2))

        sl = slice(idx[0], idx[0] + 2)
        Q[sl, sl] = curvature * np.eye(2)
        q[:] = grad - Q @ x
        const = f0 - grad @ x + 0.5 * x @ Q @ x
        return QuadraticApprox(Q=Q, q=q, const=float(const))
