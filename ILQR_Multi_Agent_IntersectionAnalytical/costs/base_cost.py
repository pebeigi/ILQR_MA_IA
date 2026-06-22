from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class QuadraticApprox:
    """Local quadratic approximation of a scalar cost."""
    Q: np.ndarray
    q: np.ndarray
    R: Dict[int, np.ndarray] = field(default_factory=dict)
    r: Dict[int, np.ndarray] = field(default_factory=dict)
    S: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    const: float = 0.0

class BaseCost:
    """Abstract interface for cost terms."""
    def evaluate(self, x: np.ndarray, us: list[np.ndarray], k: Optional[int] = None) -> float:
        raise NotImplementedError

    def quadraticize(self, x: np.ndarray, us: list[np.ndarray], k: Optional[int] = None) -> QuadraticApprox:
        raise NotImplementedError


def _smooth_floor(value: float, floor: float, width: float):
    """C2 floor: exactly floor below it, exactly value after the transition."""
    value = float(value)
    floor = float(floor)
    width = float(width)
    if width <= 0.0:
        raise ValueError("width must be positive")

    offset = value - floor
    if offset <= 0.0:
        return floor, 0.0, 0.0
    if offset >= width:
        return value, 1.0, 0.0

    s = offset / width
    poly = 3.0 * s ** 5 - 8.0 * s ** 4 + 6.0 * s ** 3
    grad = 15.0 * s ** 4 - 32.0 * s ** 3 + 18.0 * s ** 2
    hess = (60.0 * s ** 3 - 96.0 * s ** 2 + 36.0 * s) / width
    smooth = floor + width * poly
    return float(smooth), float(grad), float(hess)


def _smootherstep01(s: float):
    """C2 smoothstep on [0, 1], returning value and derivative wrt s."""
    s = float(s)
    if s <= 0.0:
        return 0.0, 0.0
    if s >= 1.0:
        return 1.0, 0.0
    value = 6.0 * s ** 5 - 15.0 * s ** 4 + 10.0 * s ** 3
    grad = 30.0 * s ** 4 - 60.0 * s ** 3 + 30.0 * s ** 2
    return float(value), float(grad)


def _heading_to_unit(heading: float):
    return np.array([np.cos(float(heading)), np.sin(float(heading))], dtype=float)


class QuadraticStateCost(BaseCost):
    def __init__(self, Q, x_ref=None, q=None, constant=0.0):
        self.Q = np.asarray(Q, dtype=float)
        self.x_ref = None if x_ref is None else np.asarray(x_ref, dtype=float).reshape(-1)
        self.q = np.zeros(self.Q.shape[0], dtype=float) if q is None else np.asarray(q, dtype=float).reshape(-1)
        self.constant = float(constant)

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        dx = x - self.x_ref if self.x_ref is not None else x
        return float(0.5 * dx @ self.Q @ dx + self.q @ x + self.constant)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        q = self.q.copy()
        const = self.constant
        if self.x_ref is not None:
            q = self.q - self.Q @ self.x_ref
            const = self.constant + 0.5 * self.x_ref @ self.Q @ self.x_ref
        return QuadraticApprox(Q=self.Q.copy(), q=q, const=float(const))

class QuadraticControlCost(BaseCost):
    def __init__(self, player_index: int, R, u_ref=None, r=None, constant=0.0):
        self.player_index = int(player_index)
        self.R = np.asarray(R, dtype=float)
        self.u_ref = None if u_ref is None else np.asarray(u_ref, dtype=float).reshape(-1)
        self.r = np.zeros(self.R.shape[0], dtype=float) if r is None else np.asarray(r, dtype=float).reshape(-1)
        self.constant = float(constant)

    def evaluate(self, x, us, k=None):
        u = np.asarray(us[self.player_index], dtype=float).reshape(-1)
        du = u - self.u_ref if self.u_ref is not None else u
        return float(0.5 * du @ self.R @ du + self.r @ u + self.constant)

    def quadraticize(self, x, us, k=None):
        r = self.r.copy()
        const = self.constant
        if self.u_ref is not None:
            r = self.r - self.R @ self.u_ref
            const = self.constant + 0.5 * self.u_ref @ self.R @ self.u_ref
        nx = len(np.asarray(x).reshape(-1))
        return QuadraticApprox(Q=np.zeros((nx, nx)), q=np.zeros(nx, dtype=float), R={self.player_index: self.R.copy()}, r={self.player_index: r}, const=float(const))

class TerminalDestinationCost(BaseCost):
    """Quadratic terminal-only cost on destination error.

    Cost at terminal step only:
        0.5 * weight * || [px, py] - destination ||^2

    State is assumed to be [px, py, speed, heading].
    """
    def __init__(self, destination, terminal_step: int, weight: float = 100.0):
        self.destination = np.asarray(destination, dtype=float).reshape(2)
        self.terminal_step = int(terminal_step)
        self.weight = float(weight)

    def _active(self, k):
        return k is None or int(k) == self.terminal_step

    def evaluate(self, x, us, k=None):
        if not self._active(k):
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)
        d = x[:2] - self.destination
        return float(0.5 * self.weight * d @ d)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        const = 0.0
        if self._active(k):
            Q[0, 0] = self.weight
            Q[1, 1] = self.weight
            q[0:2] = -self.weight * self.destination
            const = 0.5 * self.weight * float(self.destination @ self.destination)
        return QuadraticApprox(Q=Q, q=q, const=float(const))


class InverseSpeedCost(BaseCost):
    """Running state cost that encourages the vehicle not to move too slowly.

    Cost:
        weight / (speed + epsilon)

    State is assumed to be [px, py, speed, heading].  A smooth speed floor is
    used to avoid division by zero without introducing a gradient kink.
    """
    def __init__(
        self,
        weight: float = 0.2,
        epsilon: float = 0.2,
        speed_floor: float = 0.05,
        floor_width: float = 0.05,
    ):
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        self.speed_floor = float(speed_floor)
        self.floor_width = float(floor_width)
        if self.floor_width <= 0.0:
            raise ValueError("floor_width must be positive")

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        speed, _, _ = _smooth_floor(float(x[2]), self.speed_floor, self.floor_width)
        return float(self.weight / (speed + self.epsilon))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)

        speed_idx = 2
        raw_speed = float(x[speed_idx])
        smooth_speed, speed_grad, speed_hess = _smooth_floor(
            raw_speed,
            self.speed_floor,
            self.floor_width,
        )
        denom = smooth_speed + self.epsilon
        f0 = self.weight / denom
        grad = -self.weight * speed_grad / (denom ** 2)
        exact_hess = (
            2.0 * self.weight * speed_grad ** 2 / (denom ** 3)
            - self.weight * speed_hess / (denom ** 2)
        )
        hess = max(0.0, exact_hess)

        Q[speed_idx, speed_idx] = hess
        q[speed_idx] = grad - hess * raw_speed
        const = f0 - grad * raw_speed + 0.5 * hess * raw_speed * raw_speed
        return QuadraticApprox(Q=Q, q=q, const=float(const))



class ObstacleRepulsionCost(BaseCost):
    """Running state cost that repels the vehicle from a stationary obstacle.

    Cost:
        weight / (||[px, py] - obstacle||^2 + epsilon)

    The cost is high close to the obstacle and quickly becomes small farther
    away.  State is assumed to be [px, py, speed, heading].
    """
    def __init__(self, obstacle, weight: float = 3.0, epsilon: float = 0.25):
        self.obstacle = np.asarray(obstacle, dtype=float).reshape(2)
        self.weight = float(weight)
        self.epsilon = float(epsilon)

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        delta = x[:2] - self.obstacle
        dist_sq = float(delta @ delta)
        return float(self.weight / (dist_sq + self.epsilon))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)

        delta = x[:2] - self.obstacle
        z = float(delta @ delta + self.epsilon)
        f0 = self.weight / z

        # Exact gradient with respect to position [px, py].
        grad_pos = -2.0 * self.weight * delta / (z ** 2)

        # The exact Hessian of 1/(distance^2 + epsilon) is indefinite in some
        # directions.  The LQ subproblem is much more stable if the local state
        # curvature is positive semidefinite.  We therefore keep the exact
        # first-order repulsion gradient, but use a PSD curvature matrix.
        # Because both grad_pos and curvature scale with self.weight, changing
        # obstacle_weight changes the optimized trajectory.
        curvature = max(1e-8, 2.0 * self.weight / (z ** 2))
        hess_pos = curvature * np.eye(2)

        Q[:2, :2] = hess_pos
        q[:2] = grad_pos - hess_pos @ x[:2]
        const = f0 - grad_pos @ x[:2] + 0.5 * x[:2] @ hess_pos @ x[:2]
        return QuadraticApprox(Q=Q, q=q, const=float(const))



class AgentTerminalDestinationCost(BaseCost):
    """Terminal-only quadratic destination cost for one agent in a concatenated state.

    Cost at terminal step only:
        0.5 * weight * ||position_i - destination_i||^2
    """
    def __init__(self, agent_index: int, state_dim_per_agent: int, destination, terminal_step: int, weight: float = 100.0):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.destination = np.asarray(destination, dtype=float).reshape(2)
        self.terminal_step = int(terminal_step)
        self.weight = float(weight)

    def _state_slice(self):
        start = self.agent_index * self.state_dim_per_agent
        return slice(start, start + self.state_dim_per_agent)

    def _position_indices(self):
        start = self.agent_index * self.state_dim_per_agent
        return [start, start + 1]

    def _active(self, k):
        return k is None or int(k) == self.terminal_step

    def evaluate(self, x, us, k=None):
        if not self._active(k):
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)
        xi = x[self._state_slice()]
        d = xi[:2] - self.destination
        return float(0.5 * self.weight * d @ d)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        const = 0.0
        if self._active(k):
            ix, iy = self._position_indices()
            Q[ix, ix] = self.weight
            Q[iy, iy] = self.weight
            q[ix] = -self.weight * self.destination[0]
            q[iy] = -self.weight * self.destination[1]
            const = 0.5 * self.weight * float(self.destination @ self.destination)
        return QuadraticApprox(Q=Q, q=q, const=float(const))


class AgentInverseSpeedCost(BaseCost):
    """Inverse-speed running cost for one agent in a concatenated state.

    Cost:
        weight / (speed_i + epsilon)
    """
    def __init__(
        self,
        agent_index: int,
        state_dim_per_agent: int,
        weight: float = 0.2,
        epsilon: float = 0.2,
        speed_floor: float = 0.05,
        floor_width: float = 0.05,
    ):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        self.speed_floor = float(speed_floor)
        self.floor_width = float(floor_width)
        if self.floor_width <= 0.0:
            raise ValueError("floor_width must be positive")

    def _speed_index(self):
        return self.agent_index * self.state_dim_per_agent + 2

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        speed, _, _ = _smooth_floor(
            float(x[self._speed_index()]),
            self.speed_floor,
            self.floor_width,
        )
        return float(self.weight / (speed + self.epsilon))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)

        idx = self._speed_index()
        raw_speed = float(x[idx])
        smooth_speed, speed_grad, speed_hess = _smooth_floor(
            raw_speed,
            self.speed_floor,
            self.floor_width,
        )
        denom = smooth_speed + self.epsilon
        f0 = self.weight / denom
        grad = -self.weight * speed_grad / (denom ** 2)
        exact_hess = (
            2.0 * self.weight * speed_grad ** 2 / (denom ** 3)
            - self.weight * speed_hess / (denom ** 2)
        )
        hess = max(0.0, exact_hess)

        Q[idx, idx] = hess
        q[idx] = grad - hess * raw_speed
        const = f0 - grad * raw_speed + 0.5 * hess * raw_speed * raw_speed
        return QuadraticApprox(Q=Q, q=q, const=float(const))


class StaticObstacleRepulsionCost(BaseCost):
    """Repulsion from a stationary obstacle for one agent.

    Cost:
        weight / (||position_i - obstacle||^2 + epsilon)
    """
    def __init__(self, agent_index: int, state_dim_per_agent: int, obstacle, weight: float = 3.0, epsilon: float = 0.25):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.obstacle = np.asarray(obstacle, dtype=float).reshape(2)
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        start = self.agent_index * self.state_dim_per_agent
        self._position_slice = slice(start, start + 2)

    def _position_indices(self):
        start = self.agent_index * self.state_dim_per_agent
        return [start, start + 1]

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        ix, iy = self._position_indices()
        p = x[[ix, iy]]
        delta = p - self.obstacle
        dist_sq = float(delta @ delta)
        return float(self.weight / (dist_sq + self.epsilon))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        if self.weight == 0.0:
            return QuadraticApprox(Q=Q, q=q, const=0.0)

        ix, iy = self._position_indices()
        idx = np.array([ix, iy])
        p = x[idx]
        delta = p - self.obstacle
        z = float(delta @ delta + self.epsilon)
        f0 = self.weight / z
        grad_pos = -2.0 * self.weight * delta / (z ** 2)

        # Use exact gradient and positive-semidefinite local curvature for stability.
        curvature = max(1e-8, 2.0 * self.weight / (z ** 2))
        H = curvature * np.eye(2)
        Q[self._position_slice, self._position_slice] = H
        q[idx] = grad_pos - H @ p
        const = f0 - grad_pos @ p + 0.5 * p @ H @ p
        return QuadraticApprox(Q=Q, q=q, const=float(const))


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class BatchedStaticObstacleRepulsionCost(BaseCost):
    """Vectorized repulsion from all static obstacles in one pass.

    Mathematically identical to summing N individual StaticObstacleRepulsionCost
    terms but computed with a single vectorized numpy call — avoids the Python
    overhead and repeated zero-matrix allocation of 42 separate quadraticize calls.

    Cost: sum_i  weight / (||p - obs_i||^2 + epsilon)
    """
    def __init__(self, agent_index: int, state_dim_per_agent: int,
                 obstacles, weight: float = 1000.0, epsilon: float = 0.1):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.obstacles = np.asarray(obstacles, dtype=float)   # (N, 2)
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        start = self.agent_index * self.state_dim_per_agent
        self._idx = np.array([start, start + 1], dtype=int)
        self._slice = slice(start, start + 2)

    def evaluate(self, x, us, k=None):
        if self.weight == 0.0:
            return 0.0
        p = np.asarray(x, dtype=float).reshape(-1)[self._idx]
        deltas = p - self.obstacles                          # (N, 2)
        z = np.sum(deltas ** 2, axis=1) + self.epsilon       # (N,)
        return float(np.sum(self.weight / z))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        if self.weight == 0.0:
            return QuadraticApprox(Q=Q, q=q, const=0.0)

        idx = self._idx
        p = x[idx]
        deltas = p - self.obstacles                              # (N, 2)
        z = np.sum(deltas ** 2, axis=1) + self.epsilon          # (N,)

        f0 = float(np.sum(self.weight / z))

        # Sum of exact gradients: -2*w*delta_i/z_i^2
        grad_pos = np.sum(-2.0 * self.weight * deltas / (z ** 2)[:, np.newaxis], axis=0)  # (2,)

        # Sum of PSD per-obstacle Hessians: each H_i = curvature_i * I_2
        curvatures = np.maximum(1e-8, 2.0 * self.weight / z ** 2)   # (N,)
        total_curvature = float(np.sum(curvatures))
        H = total_curvature * np.eye(2)

        Q[self._slice, self._slice] = H
        q[idx] = grad_pos - H @ p
        const = f0 - grad_pos @ p + 0.5 * p @ H @ p
        return QuadraticApprox(Q=Q, q=q, const=float(const))

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        """Accumulate the analytical approximation without full-size temporaries."""
        if self.weight == 0.0:
            return 0.0

        p = x[self._idx]
        deltas = p - self.obstacles
        z = np.sum(deltas ** 2, axis=1) + self.epsilon
        grad_pos = np.sum(
            -2.0 * self.weight * deltas / (z ** 2)[:, np.newaxis],
            axis=0,
        )
        total_curvature = float(np.sum(np.maximum(1e-8, 2.0 * self.weight / z ** 2)))
        approx.Q[self._idx[0], self._idx[0]] += total_curvature
        approx.Q[self._idx[1], self._idx[1]] += total_curvature
        approx.q[self._idx] += grad_pos - total_curvature * p
        if not compute_const:
            return 0.0
        f0 = float(np.sum(self.weight / z))
        return f0 - grad_pos @ p + 0.5 * total_curvature * (p @ p)


class AgentRunningDestinationCost(BaseCost):
    """Weak quadratic running cost toward destination, active every step.

    Prevents an agent from drifting past or wandering away from its destination
    when the planning horizon extends well beyond the natural arrival time.

    Cost: (weight/2) · ||[px, py] - destination||²
    """
    def __init__(self, agent_index: int, state_dim_per_agent: int, destination, weight: float = 5.0):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.destination = np.asarray(destination, dtype=float).reshape(2)
        self.weight = float(weight)
        start = self.agent_index * self.state_dim_per_agent
        self._idx = np.array([start, start + 1], dtype=int)

    def _position_indices(self):
        start = self.agent_index * self.state_dim_per_agent
        return start, start + 1

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        ix, iy = self._position_indices()
        d = x[[ix, iy]] - self.destination
        return float(0.5 * self.weight * d @ d)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        ix, iy = self._position_indices()
        Q[ix, ix] = self.weight
        Q[iy, iy] = self.weight
        q[ix] = -self.weight * self.destination[0]
        q[iy] = -self.weight * self.destination[1]
        const = float(0.5 * self.weight * float(self.destination @ self.destination))
        return QuadraticApprox(Q=Q, q=q, const=const)

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        approx.Q[self._idx[0], self._idx[0]] += self.weight
        approx.Q[self._idx[1], self._idx[1]] += self.weight
        approx.q[self._idx] -= self.weight * self.destination
        if compute_const:
            return 0.5 * self.weight * float(self.destination @ self.destination)
        return 0.0


class AgentArrivalHoldCost(BaseCost):
    """Post-arrival running cost that keeps an agent parked at its destination.

    The activation gate is based on signed remaining distance along stop_heading.
    It is zero before the final approach, ramps on smoothly, and remains one
    after pass-through.
    """
    def __init__(
        self,
        agent_index: int,
        state_dim_per_agent: int,
        destination,
        stop_heading: float,
        stop_radius: float = 1.0,
        transition_width: float = 2.0,
        position_weight: float = 50.0,
        speed_weight: float = 50.0,
    ):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.destination = np.asarray(destination, dtype=float).reshape(2)
        self.stop_direction = _heading_to_unit(float(stop_heading))
        self.stop_radius = float(stop_radius)
        self.transition_width = float(transition_width)
        self.position_weight = float(position_weight)
        self.speed_weight = float(speed_weight)
        start = self.agent_index * self.state_dim_per_agent
        self._pos_idx = np.array([start, start + 1], dtype=int)
        self._speed_idx = start + 2
        self._position_slice = slice(start, start + 2)
        if self.transition_width <= 0.0:
            raise ValueError("transition_width must be positive")

    def _indices(self):
        return self._pos_idx, self._speed_idx

    def _gate_and_position_grad(self, pos):
        remaining = float((self.destination - pos) @ self.stop_direction)
        s = (remaining - self.stop_radius) / self.transition_width
        before_gate, dbefore_ds = _smootherstep01(s)
        gate = 1.0 - before_gate
        # d remaining / d pos = -stop_direction.
        gate_grad = (dbefore_ds / self.transition_width) * self.stop_direction
        return float(gate), gate_grad

    def evaluate(self, x, us, k=None):
        if self.position_weight == 0.0 and self.speed_weight == 0.0:
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)
        pos_idx, speed_idx = self._indices()
        pos = x[pos_idx]
        speed = float(x[speed_idx])
        gate, _ = self._gate_and_position_grad(pos)
        position_cost = 0.5 * self.position_weight * float((pos - self.destination) @ (pos - self.destination))
        speed_cost = 0.5 * self.speed_weight * speed ** 2
        return float(gate * (position_cost + speed_cost))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        if self.position_weight == 0.0 and self.speed_weight == 0.0:
            return QuadraticApprox(Q=Q, q=q, const=0.0)

        pos_idx, speed_idx = self._indices()
        pos = x[pos_idx]
        speed = float(x[speed_idx])
        pos_error = pos - self.destination
        gate, gate_grad = self._gate_and_position_grad(pos)

        position_cost = 0.5 * self.position_weight * float(pos_error @ pos_error)
        speed_cost = 0.5 * self.speed_weight * speed ** 2
        base_cost = position_cost + speed_cost

        grad = np.zeros(nx, dtype=float)
        grad[pos_idx] = gate * self.position_weight * pos_error + base_cost * gate_grad
        grad[speed_idx] = gate * self.speed_weight * speed

        Q[self._position_slice, self._position_slice] = gate * self.position_weight * np.eye(2)
        Q[speed_idx, speed_idx] = gate * self.speed_weight

        q = grad - Q @ x
        const = float(gate * base_cost - grad @ x + 0.5 * x @ Q @ x)
        return QuadraticApprox(Q=Q, q=q, const=const)

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        pos = x[self._pos_idx]
        speed = float(x[self._speed_idx])
        pos_error = pos - self.destination
        gate, gate_grad = self._gate_and_position_grad(pos)
        position_cost = 0.5 * self.position_weight * float(pos_error @ pos_error)
        speed_cost = 0.5 * self.speed_weight * speed ** 2
        base_cost = position_cost + speed_cost
        grad_pos = gate * self.position_weight * pos_error + base_cost * gate_grad
        grad_speed = gate * self.speed_weight * speed
        pos_curvature = gate * self.position_weight
        speed_curvature = gate * self.speed_weight

        approx.Q[self._pos_idx[0], self._pos_idx[0]] += pos_curvature
        approx.Q[self._pos_idx[1], self._pos_idx[1]] += pos_curvature
        approx.Q[self._speed_idx, self._speed_idx] += speed_curvature
        approx.q[self._pos_idx] += grad_pos - pos_curvature * pos
        approx.q[self._speed_idx] += grad_speed - speed_curvature * speed
        if not compute_const:
            return 0.0
        return (
            gate * base_cost
            - grad_pos @ pos
            - grad_speed * speed
            + 0.5 * pos_curvature * (pos @ pos)
            + 0.5 * speed_curvature * speed ** 2
        )


class AgentRunningSpeedCost(BaseCost):
    """Quadratic running cost on agent speed error.

    Away from the destination the target is desired_speed.  If stop_heading is
    supplied, the target tapers to zero over the final approach measured along
    that heading, and stays zero after the agent has passed the destination.
    State per agent: [px, py, speed, heading] — speed at local index 2.
    """
    def __init__(
        self,
        agent_index: int,
        state_dim_per_agent: int,
        q_speed: float = 1.0,
        desired_speed: float = 0.0,
        destination=None,
        stop_radius: float = 0.0,
        transition_width: float = 1.0,
        stop_heading: float | None = None,
    ):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.q_speed = float(q_speed)
        self.desired_speed = float(desired_speed)
        self.destination = None if destination is None else np.asarray(destination, dtype=float).reshape(2)
        self.stop_radius = float(stop_radius)
        self.transition_width = float(transition_width)
        self.stop_direction = None if stop_heading is None else _heading_to_unit(float(stop_heading))
        start = self.agent_index * self.state_dim_per_agent
        self._pos_idx = np.array([start, start + 1], dtype=int)
        self._speed_idx = start + 2
        if self.transition_width <= 0.0:
            raise ValueError("transition_width must be positive")

    def _speed_index(self):
        return self._speed_idx

    def _position_indices(self):
        return self._pos_idx

    def _speed_target_and_position_grad(self, x):
        if self.destination is None:
            return self.desired_speed, np.zeros(2, dtype=float)

        idx = self._position_indices()
        pos = np.asarray(x[idx], dtype=float).reshape(2)
        if self.stop_direction is not None:
            remaining = float((self.destination - pos) @ self.stop_direction)
            s = (remaining - self.stop_radius) / self.transition_width
            gate, dgate_ds = _smootherstep01(s)
            gate_grad = -(dgate_ds / self.transition_width) * self.stop_direction
        else:
            offset = pos - self.destination
            dist = float(np.linalg.norm(offset))
            s = (dist - self.stop_radius) / self.transition_width
            gate, dgate_ds = _smootherstep01(s)
            if dist > 1e-9:
                gate_grad = (dgate_ds / self.transition_width) * offset / dist
            else:
                gate_grad = np.zeros(2, dtype=float)

        target = self.desired_speed * gate
        target_grad = self.desired_speed * gate_grad
        return float(target), target_grad

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        speed = float(x[self._speed_index()])
        speed_target, _ = self._speed_target_and_position_grad(x)
        return float(0.5 * self.q_speed * (speed - speed_target) ** 2)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        idx = self._speed_index()
        pos_idx = self._position_indices()
        speed = float(x[idx])
        speed_target, target_grad = self._speed_target_and_position_grad(x)
        err = speed - speed_target
        
        # Keep the exact local gradient.  The position-dependent target creates
        # speed-position cross terms; the LQ packer does not consume those, so
        # we use only the PSD speed curvature here.
        Q[idx, idx] = self.q_speed
        full_grad = np.zeros(nx, dtype=float)
        full_grad[idx] = self.q_speed * err
        full_grad[pos_idx] = -self.q_speed * err * target_grad
        q = full_grad - Q @ x
        const = float(0.5 * self.q_speed * err ** 2 - full_grad @ x + 0.5 * x @ Q @ x)
        
        return QuadraticApprox(Q=Q, q=q, const=const)

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        speed = float(x[self._speed_idx])
        speed_target, target_grad = self._speed_target_and_position_grad(x)
        err = speed - speed_target
        grad_speed = self.q_speed * err
        grad_pos = -self.q_speed * err * target_grad

        approx.Q[self._speed_idx, self._speed_idx] += self.q_speed
        approx.q[self._speed_idx] += grad_speed - self.q_speed * speed
        approx.q[self._pos_idx] += grad_pos
        if not compute_const:
            return 0.0
        return (
            0.5 * self.q_speed * err ** 2
            - grad_speed * speed
            - grad_pos @ x[self._pos_idx]
            + 0.5 * self.q_speed * speed ** 2
        )


class AgentSpeedDependentControlCost(BaseCost):
    """Speed-dependent quadratic control cost for one agent.

    Running cost (exact notebook formula):
        (β₂/2) · κ² · v⁴  +  (β₃/2) · a²

    Controls per agent are [kappa, a].
    State per agent is    [px, py, speed, heading] — speed at local index 2.

    R(v) = diag(β₂·v⁴ + ε, β₃ + ε)   (state-dependent, evaluated at nominal v)
    """
    def __init__(
        self,
        agent_index: int,
        state_dim_per_agent: int,
        player_index: int,
        beta_2: float = 5.0,
        beta_3: float = 10.0,
        v_min: float = 0.5,
        floor_width: float = 0.05,
    ):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.player_index = int(player_index)
        self.beta_2 = float(beta_2)
        self.beta_3 = float(beta_3)
        self.v_min = float(v_min)
        self.floor_width = float(floor_width)
        self._speed_idx = self.agent_index * self.state_dim_per_agent + 2
        if self.floor_width <= 0.0:
            raise ValueError("floor_width must be positive")

    def _speed_index(self):
        return self._speed_idx

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(us[self.player_index], dtype=float).reshape(-1)
        v, _, _ = _smooth_floor(float(x[self._speed_index()]), self.v_min, self.floor_width)
        kappa, a = float(u[0]), float(u[1])
        return float(0.5 * self.beta_2 * kappa ** 2 * v ** 4 + 0.5 * self.beta_3 * a ** 2)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        u = np.asarray(us[self.player_index], dtype=float).reshape(-1)
        raw_speed = float(x[self._speed_index()])
        v, v_grad, v_hess = _smooth_floor(raw_speed, self.v_min, self.floor_width)
        kappa, a = float(u[0]), float(u[1])

        f0 = 0.5 * self.beta_2 * kappa ** 2 * v ** 4 + 0.5 * self.beta_3 * a ** 2
        speed_grad = 2.0 * self.beta_2 * kappa ** 2 * v ** 3 * v_grad
        speed_hess = self.beta_2 * kappa ** 2 * (
            6.0 * v ** 2 * v_grad ** 2 + 2.0 * v ** 3 * v_hess
        )
        speed_hess = max(0.0, speed_hess)

        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        speed_idx = self._speed_index()
        Q[speed_idx, speed_idx] = speed_hess
        q[speed_idx] = speed_grad - speed_hess * raw_speed

        reg = 1e-3
        R = np.diag([self.beta_2 * v ** 4 + reg, self.beta_3 + reg])
        control_grad = np.array([self.beta_2 * kappa * v ** 4, self.beta_3 * a], dtype=float)
        r = control_grad - R @ u
        const = (
            f0
            - speed_grad * raw_speed
            + 0.5 * speed_hess * raw_speed ** 2
            - control_grad @ u
            + 0.5 * u @ R @ u
        )
        return QuadraticApprox(
            Q=Q,
            q=q,
            R={self.player_index: R},
            r={self.player_index: r},
            const=float(const),
        )

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        u = us[self.player_index]
        raw_speed = float(x[self._speed_idx])
        v, v_grad, v_hess = _smooth_floor(raw_speed, self.v_min, self.floor_width)
        kappa, a = float(u[0]), float(u[1])
        v4 = v ** 4
        speed_grad = 2.0 * self.beta_2 * kappa ** 2 * v ** 3 * v_grad
        speed_hess = max(
            0.0,
            self.beta_2 * kappa ** 2 * (
                6.0 * v ** 2 * v_grad ** 2 + 2.0 * v ** 3 * v_hess
            ),
        )
        r0 = self.beta_2 * v4 + 1e-3
        r1 = self.beta_3 + 1e-3
        grad_u0 = self.beta_2 * kappa * v4
        grad_u1 = self.beta_3 * a

        approx.Q[self._speed_idx, self._speed_idx] += speed_hess
        approx.q[self._speed_idx] += speed_grad - speed_hess * raw_speed
        approx.R[self.player_index][0, 0] += r0
        approx.R[self.player_index][1, 1] += r1
        approx.r[self.player_index][0] += grad_u0 - r0 * kappa
        approx.r[self.player_index][1] += grad_u1 - r1 * a
        if not compute_const:
            return 0.0
        f0 = 0.5 * self.beta_2 * kappa ** 2 * v4 + 0.5 * self.beta_3 * a ** 2
        return (
            f0
            - speed_grad * raw_speed
            + 0.5 * speed_hess * raw_speed ** 2
            - grad_u0 * kappa
            - grad_u1 * a
            + 0.5 * r0 * kappa ** 2
            + 0.5 * r1 * a ** 2
        )


class AgentFullTerminalCost(BaseCost):
    """4-component terminal cost matching the notebook Qf exactly.

    Terminal cost (active only at terminal_step):
        ψ = (b_px/2)(px−px*)² + (b_py/2)(py−py*)²
          + (b_speed/2)(speed−v*)² + (b_heading/2)·(2 sin((heading−θ*)/2))²

    State per agent: [px, py, speed, heading].
    Heading uses a smooth periodic proxy instead of angle wrapping.
    """
    def __init__(
        self,
        agent_index: int,
        state_dim_per_agent: int,
        destination,
        desired_speed: float,
        desired_heading: float,
        terminal_step: int,
        b_px: float = 250.0,
        b_py: float = 250.0,
        b_speed: float = 5.0,
        b_heading: float = 50.0,
    ):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.destination = np.asarray(destination, dtype=float).reshape(2)
        self.desired_speed = float(desired_speed)
        self.desired_heading = float(desired_heading)
        self.terminal_step = int(terminal_step)
        self.b_px = float(b_px)
        self.b_py = float(b_py)
        self.b_speed = float(b_speed)
        self.b_heading = float(b_heading)
        self._idx = self._indices()

    def _indices(self):
        if hasattr(self, "_idx"):
            return self._idx
        base = self.agent_index * self.state_dim_per_agent
        return base, base + 1, base + 2, base + 3

    def _active(self, k):
        return k is None or int(k) == self.terminal_step

    def _errors(self, x):
        ix, iy, ispeed, iheading = self._indices()
        return (
            float(x[ix]) - self.destination[0],
            float(x[iy]) - self.destination[1],
            float(x[ispeed]) - self.desired_speed,
            float(x[iheading]) - self.desired_heading,
        )

    def evaluate(self, x, us, k=None):
        if not self._active(k):
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)
        dx, dy, dspeed, dheading = self._errors(x)
        heading_proxy = 2.0 * np.sin(0.5 * dheading)
        return float(
            0.5 * self.b_px * dx ** 2 +
            0.5 * self.b_py * dy ** 2 +
            0.5 * self.b_speed * dspeed ** 2 +
            0.5 * self.b_heading * heading_proxy ** 2
        )

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        const = 0.0
        if not self._active(k):
            return QuadraticApprox(Q=Q, q=q, const=const)

        ix, iy, ispeed, iheading = self._indices()
        dx, dy, dspeed, dheading = self._errors(x)

        Q[ix, ix] = self.b_px
        Q[iy, iy] = self.b_py
        Q[ispeed, ispeed] = self.b_speed
        heading_grad = self.b_heading * np.sin(dheading)
        heading_hess = self.b_heading * np.cos(dheading)
        Q[iheading, iheading] = max(0.0, heading_hess)

        # q chosen so that Q @ x + q = true gradient at x.
        q[ix] = -self.b_px * self.destination[0]
        q[iy] = -self.b_py * self.destination[1]
        q[ispeed] = -self.b_speed * self.desired_speed
        q[iheading] = heading_grad - Q[iheading, iheading] * float(x[iheading])

        heading_cost = self.b_heading * (1.0 - np.cos(dheading))
        const = float(
            0.5 * self.b_px * self.destination[0] ** 2 +
            0.5 * self.b_py * self.destination[1] ** 2 +
            0.5 * self.b_speed * self.desired_speed ** 2 +
            heading_cost
            - heading_grad * float(x[iheading])
            + 0.5 * Q[iheading, iheading] * float(x[iheading]) ** 2
        )
        return QuadraticApprox(Q=Q, q=q, const=const)

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        if not self._active(k):
            return 0.0
        ix, iy, ispeed, iheading = self._idx
        _, _, _, dheading = self._errors(x)
        heading_grad = self.b_heading * np.sin(dheading)
        heading_curvature = max(0.0, self.b_heading * np.cos(dheading))
        heading = float(x[iheading])

        approx.Q[ix, ix] += self.b_px
        approx.Q[iy, iy] += self.b_py
        approx.Q[ispeed, ispeed] += self.b_speed
        approx.Q[iheading, iheading] += heading_curvature
        approx.q[ix] -= self.b_px * self.destination[0]
        approx.q[iy] -= self.b_py * self.destination[1]
        approx.q[ispeed] -= self.b_speed * self.desired_speed
        approx.q[iheading] += heading_grad - heading_curvature * heading
        if not compute_const:
            return 0.0
        heading_cost = self.b_heading * (1.0 - np.cos(dheading))
        return (
            0.5 * self.b_px * self.destination[0] ** 2
            + 0.5 * self.b_py * self.destination[1] ** 2
            + 0.5 * self.b_speed * self.desired_speed ** 2
            + heading_cost
            - heading_grad * heading
            + 0.5 * heading_curvature * heading ** 2
        )


class PairwiseAgentRepulsionCost(BaseCost):
    """Repulsion between two dynamic agents at the same time index.

    Cost for player i from another player j:
        weight * gate_i * gate_j / (||position_i - position_j||^2 + epsilon)

    If active_only_when_both_moving is True, the term is smoothly deactivated
    as either agent reaches the final approach to its own destination.  When
    stop_headings are supplied, this uses signed remaining distance along the
    terminal lane direction, so the gate stays off after pass-through instead of
    reactivating when Euclidean distance grows again.
    """
    def __init__(
        self,
        agent_index: int,
        other_agent_index: int,
        state_dim_per_agent: int,
        destinations,
        stop_radius: float,
        weight: float = 10.0,
        epsilon: float = 0.25,
        active_only_when_both_moving: bool = True,
        activation_width: float = 0.3,
        stop_headings=None,
    ):
        self.agent_index = int(agent_index)
        self.other_agent_index = int(other_agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.destinations = [np.asarray(d, dtype=float).reshape(2) for d in destinations]
        if stop_headings is None:
            self.stop_directions = None
        else:
            self.stop_directions = [_heading_to_unit(h) for h in stop_headings]
            if len(self.stop_directions) != len(self.destinations):
                raise ValueError("stop_headings must have one entry per destination")
        self.stop_radius = float(stop_radius)
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        self.active_only_when_both_moving = bool(active_only_when_both_moving)
        self.activation_width = float(activation_width)
        start_i = self.agent_index * self.state_dim_per_agent
        start_j = self.other_agent_index * self.state_dim_per_agent
        self._idx_i = np.array([start_i, start_i + 1], dtype=int)
        self._idx_j = np.array([start_j, start_j + 1], dtype=int)
        self._slice_i = slice(start_i, start_i + 2)
        self._slice_j = slice(start_j, start_j + 2)
        if self.activation_width <= 0.0:
            raise ValueError("activation_width must be positive")

    def _position_indices(self, agent_index):
        start = int(agent_index) * self.state_dim_per_agent
        return np.array([start, start + 1], dtype=int)

    @staticmethod
    def _sigmoid(value):
        value = float(value)
        if value >= 0.0:
            z = np.exp(-value)
            return float(1.0 / (1.0 + z))
        z = np.exp(value)
        return float(z / (1.0 + z))

    def _gate_scale(self):
        return max(
            2.0 * max(self.stop_radius, 0.0) * self.activation_width,
            self.activation_width ** 2,
            1e-8,
        )

    def _gate_and_grad(self, x, agent_index):
        if not self.active_only_when_both_moving:
            return 1.0, np.zeros(2, dtype=float)

        if agent_index == self.agent_index:
            idx = self._idx_i
        elif agent_index == self.other_agent_index:
            idx = self._idx_j
        else:
            idx = self._position_indices(agent_index)
        pos = np.asarray(x[idx], dtype=float).reshape(2)
        if self.stop_directions is not None:
            direction = self.stop_directions[agent_index]
            remaining = float((self.destinations[agent_index] - pos) @ direction)
            arg = (remaining - self.stop_radius) / self.activation_width
            gate = self._sigmoid(arg)
            grad = -(gate * (1.0 - gate) / self.activation_width) * direction
            return gate, grad

        offset = pos - self.destinations[agent_index]
        scale = self._gate_scale()
        arg = (float(offset @ offset) - self.stop_radius ** 2) / scale
        gate = self._sigmoid(arg)
        grad = (2.0 * gate * (1.0 - gate) / scale) * offset
        return gate, grad

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.weight == 0.0:
            return 0.0
        idx_i = self._idx_i
        idx_j = self._idx_j
        gate_i, _ = self._gate_and_grad(x, self.agent_index)
        gate_j, _ = self._gate_and_grad(x, self.other_agent_index)
        delta = x[idx_i] - x[idx_j]
        dist_sq = float(delta @ delta)
        return float(self.weight * gate_i * gate_j / (dist_sq + self.epsilon))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        if self.weight == 0.0:
            return QuadraticApprox(Q=Q, q=q, const=0.0)

        idx_i = self._idx_i
        idx_j = self._idx_j
        p_i = x[idx_i]
        p_j = x[idx_j]
        gate_i, grad_gate_i = self._gate_and_grad(x, self.agent_index)
        gate_j, grad_gate_j = self._gate_and_grad(x, self.other_agent_index)
        gate_prod = gate_i * gate_j
        delta = p_i - p_j
        z = float(delta @ delta + self.epsilon)
        f0 = self.weight * gate_prod / z

        grad = np.zeros(nx, dtype=float)
        grad[idx_i] += self.weight * (
            gate_j * grad_gate_i / z
            - 2.0 * gate_prod * delta / (z ** 2)
        )
        grad[idx_j] += self.weight * (
            gate_i * grad_gate_j / z
            + 2.0 * gate_prod * delta / (z ** 2)
        )

        # PSD curvature in the relative-position variable d = p_i - p_j.
        # This produces block curvature [[H, -H], [-H, H]].
        curvature = max(1e-8, 2.0 * self.weight * gate_prod / (z ** 2))
        H = curvature * np.eye(2)
        Q[self._slice_i, self._slice_i] += H
        Q[self._slice_j, self._slice_j] += H
        Q[self._slice_i, self._slice_j] -= H
        Q[self._slice_j, self._slice_i] -= H

        q = grad - Q @ x
        const = f0 - grad @ x + 0.5 * x @ Q @ x
        return QuadraticApprox(Q=Q, q=q, const=float(const))

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        """Accumulate the analytical approximation without full-size temporaries."""
        idx_i = self._idx_i
        idx_j = self._idx_j
        p_i = x[idx_i]
        p_j = x[idx_j]
        gate_i, grad_gate_i = self._gate_and_grad(x, self.agent_index)
        gate_j, grad_gate_j = self._gate_and_grad(x, self.other_agent_index)
        gate_prod = gate_i * gate_j
        delta = p_i - p_j
        z = float(delta @ delta + self.epsilon)
        grad_i = self.weight * (
            gate_j * grad_gate_i / z
            - 2.0 * gate_prod * delta / (z ** 2)
        )
        grad_j = self.weight * (
            gate_i * grad_gate_j / z
            + 2.0 * gate_prod * delta / (z ** 2)
        )
        curvature = max(1e-8, 2.0 * self.weight * gate_prod / (z ** 2))
        H_delta = curvature * delta

        for axis in range(2):
            ii = idx_i[axis]
            jj = idx_j[axis]
            approx.Q[ii, ii] += curvature
            approx.Q[jj, jj] += curvature
            approx.Q[ii, jj] -= curvature
            approx.Q[jj, ii] -= curvature
        approx.q[idx_i] += grad_i - H_delta
        approx.q[idx_j] += grad_j + H_delta
        if not compute_const:
            return 0.0
        f0 = self.weight * gate_prod / z
        return f0 - grad_i @ p_i - grad_j @ p_j + 0.5 * curvature * (delta @ delta)


class AgentProximitySpeedCost(BaseCost):
    """Penalizes agent speed proportional to proximity to another agent.

    Cost: weight * gate * smooth_max(speed_i, 0) / (||p_i - p_j||^2 + epsilon)

    When agents are spatially close, the coefficient on speed_i rises, creating a
    direct gradient that says "slow down."  This couples the temporal (speed) and
    spatial (position) dimensions that PairwiseAgentRepulsionCost misses.  If
    activation_distance is supplied, a C2 distance gate turns the cost off at
    normal gaps and smoothly ramps it to full strength as the pair gets close.
    """
    def __init__(
        self,
        agent_index: int,
        other_agent_index: int,
        state_dim_per_agent: int,
        weight: float = 10.0,
        epsilon: float = 2.0,
        speed_transition_width: float = 0.1,
        activation_distance: float | None = None,
        activation_width: float = 2.0,
    ):
        self.agent_index = int(agent_index)
        self.other_agent_index = int(other_agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        self.speed_transition_width = float(speed_transition_width)
        self.activation_distance = None if activation_distance is None else float(activation_distance)
        self.activation_width = float(activation_width)
        start_i = self.agent_index * self.state_dim_per_agent
        start_j = self.other_agent_index * self.state_dim_per_agent
        self._idx_i = np.array([start_i, start_i + 1], dtype=int)
        self._idx_j = np.array([start_j, start_j + 1], dtype=int)
        self._slice_i = slice(start_i, start_i + 2)
        self._slice_j = slice(start_j, start_j + 2)
        self._speed_idx_i = start_i + 2
        if self.speed_transition_width <= 0.0:
            raise ValueError("speed_transition_width must be positive")
        if self.activation_distance is not None and self.activation_distance <= 0.0:
            raise ValueError("activation_distance must be positive")
        if self.activation_width <= 0.0:
            raise ValueError("activation_width must be positive")

    def _position_indices(self, agent_index):
        start = int(agent_index) * self.state_dim_per_agent
        return np.array([start, start + 1], dtype=int)

    def _speed_index(self, agent_index):
        return int(agent_index) * self.state_dim_per_agent + 2

    def _gate_and_grad_dist_sq(self, dist_sq: float):
        if self.activation_distance is None:
            return 1.0, 0.0
        outer_sq = self.activation_distance ** 2
        inner_distance = max(0.0, self.activation_distance - self.activation_width)
        inner_sq = inner_distance ** 2
        width_sq = outer_sq - inner_sq
        if width_sq <= 1e-12 or dist_sq <= inner_sq:
            return 1.0, 0.0
        if dist_sq >= outer_sq:
            return 0.0, 0.0
        s = (outer_sq - dist_sq) / width_sq
        gate, dgate_ds = _smootherstep01(s)
        dgate_ddist_sq = -dgate_ds / width_sq
        return gate, dgate_ddist_sq

    def evaluate(self, x, us, k=None):
        if self.weight == 0.0:
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)
        idx_i = self._idx_i
        idx_j = self._idx_j
        v, _, _ = _smooth_floor(
            float(x[self._speed_idx_i]),
            0.0,
            self.speed_transition_width,
        )
        delta = x[idx_i] - x[idx_j]
        dist_sq = float(delta @ delta)
        gate, _ = self._gate_and_grad_dist_sq(dist_sq)
        return float(self.weight * gate * v / (dist_sq + self.epsilon))

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q_vec = np.zeros(nx, dtype=float)
        if self.weight == 0.0:
            return QuadraticApprox(Q=Q, q=q_vec, const=0.0)

        idx_i = self._idx_i
        idx_j = self._idx_j
        isp = self._speed_idx_i

        p_i = x[idx_i]
        p_j = x[idx_j]
        raw_speed = float(x[isp])
        v, speed_grad, speed_hess = _smooth_floor(
            raw_speed,
            0.0,
            self.speed_transition_width,
        )
        delta = p_i - p_j
        dist_sq = float(delta @ delta)
        z = float(dist_sq + self.epsilon)
        gate, dgate_ddist_sq = self._gate_and_grad_dist_sq(dist_sq)
        f0 = self.weight * gate * v / z

        # Gradient: ∂/∂speed = weight*smooth_max'(speed)/z,
        # ∂/∂p_i = -2*weight*smooth_max(speed)*delta/z².
        grad_v = self.weight * gate * speed_grad / z
        grad_pi = -2.0 * self.weight * gate * v * delta / (z ** 2)
        gate_grad_pi = 2.0 * dgate_ddist_sq * delta
        grad_pi += self.weight * v * gate_grad_pi / z
        grad_pj = -grad_pi

        full_grad = np.zeros(nx, dtype=float)
        full_grad[isp] = grad_v
        full_grad[idx_i] = grad_pi
        full_grad[idx_j] = grad_pj

        # PSD Hessian: position block only (speed term is linear in v → zero Hessian)
        curvature = max(1e-8, 2.0 * self.weight * max(gate, 0.0) * v / (z ** 2))
        H_pos = curvature * np.eye(2)
        Q[self._slice_i, self._slice_i] += H_pos
        Q[self._slice_j, self._slice_j] += H_pos
        Q[self._slice_i, self._slice_j] -= H_pos
        Q[self._slice_j, self._slice_i] -= H_pos

        Q[isp, isp] = max(0.0, self.weight * gate * speed_hess / z)

        q_vec = full_grad - Q @ x
        const = f0 - full_grad @ x + 0.5 * x @ Q @ x
        return QuadraticApprox(Q=Q, q=q_vec, const=float(const))

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        """Accumulate the analytical approximation without full-size temporaries."""
        idx_i = self._idx_i
        idx_j = self._idx_j
        isp = self._speed_idx_i
        p_i = x[idx_i]
        p_j = x[idx_j]
        raw_speed = float(x[isp])
        v, speed_grad, speed_hess = _smooth_floor(
            raw_speed,
            0.0,
            self.speed_transition_width,
        )
        delta = p_i - p_j
        dist_sq = float(delta @ delta)
        z = float(dist_sq + self.epsilon)
        gate, dgate_ddist_sq = self._gate_and_grad_dist_sq(dist_sq)
        grad_v = self.weight * gate * speed_grad / z
        grad_i = -2.0 * self.weight * gate * v * delta / (z ** 2)
        grad_i += self.weight * v * (2.0 * dgate_ddist_sq * delta) / z
        grad_j = -grad_i

        curvature = max(1e-8, 2.0 * self.weight * max(gate, 0.0) * v / (z ** 2))
        H_delta = curvature * delta
        speed_curvature = max(0.0, self.weight * gate * speed_hess / z)

        for axis in range(2):
            ii = idx_i[axis]
            jj = idx_j[axis]
            approx.Q[ii, ii] += curvature
            approx.Q[jj, jj] += curvature
            approx.Q[ii, jj] -= curvature
            approx.Q[jj, ii] -= curvature
        approx.Q[isp, isp] += speed_curvature
        approx.q[idx_i] += grad_i - H_delta
        approx.q[idx_j] += grad_j + H_delta
        approx.q[isp] += grad_v - speed_curvature * raw_speed
        if not compute_const:
            return 0.0
        f0 = self.weight * gate * v / z
        return (
            f0
            - grad_i @ p_i
            - grad_j @ p_j
            - grad_v * raw_speed
            + 0.5 * curvature * (delta @ delta)
            + 0.5 * speed_curvature * raw_speed ** 2
        )


class AgentLeaderYieldLineCost(BaseCost):
    """Keep a follower behind a yield line until its direct leader clears.

    Cost:
        gate(leader_clearance) * C2_hinge(follower_progress - hold_value)

    This is useful for a through vehicle queued behind a left-turning leader:
    distance-only repulsion lets the follower enter the intersection with a
    gap, while this term keeps it behind the line until the leader has moved
    out of the shared lane.
    """
    def __init__(
        self,
        agent_index: int,
        leader_index: int,
        state_dim_per_agent: int,
        hold_axis: int,
        hold_value: float,
        hold_direction: float,
        clear_axis: int,
        clear_value: float,
        clear_direction: float,
        weight: float,
        transition_width: float = 0.5,
        clearance_width: float = 2.0,
    ):
        self.agent_index = int(agent_index)
        self.leader_index = int(leader_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.hold_axis = int(hold_axis)
        self.hold_value = float(hold_value)
        self.hold_direction = float(np.sign(hold_direction) or 1.0)
        self.clear_axis = int(clear_axis)
        self.clear_value = float(clear_value)
        self.clear_direction = float(np.sign(clear_direction) or 1.0)
        self.weight = float(weight)
        self.transition_width = float(transition_width)
        self.clearance_width = float(clearance_width)
        if self.hold_axis not in (0, 1) or self.clear_axis not in (0, 1):
            raise ValueError("hold_axis and clear_axis must be 0 (x) or 1 (y)")
        if self.transition_width <= 0.0:
            raise ValueError("transition_width must be positive")
        if self.clearance_width <= 0.0:
            raise ValueError("clearance_width must be positive")

    def _agent_coord_index(self):
        return self.agent_index * self.state_dim_per_agent + self.hold_axis

    def _leader_coord_index(self):
        return self.leader_index * self.state_dim_per_agent + self.clear_axis

    def _gate_and_grad(self, leader_coord: float):
        unclear = self.clear_direction * (float(leader_coord) - self.clear_value)
        s = unclear / self.clearance_width
        gate, dgate_ds = _smootherstep01(s)
        dgate_dleader = dgate_ds * self.clear_direction / self.clearance_width
        return gate, dgate_dleader

    def evaluate(self, x, us, k=None):
        if self.weight == 0.0:
            return 0.0
        x = np.asarray(x, dtype=float).reshape(-1)
        follower_coord = float(x[self._agent_coord_index()])
        leader_coord = float(x[self._leader_coord_index()])

        gate, _ = self._gate_and_grad(leader_coord)
        err = self.hold_direction * (follower_coord - self.hold_value)
        hold_cost, _, _ = _c2_quadratic_hinge(err, self.weight, self.transition_width)
        return float(gate * hold_cost)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        if self.weight == 0.0:
            return QuadraticApprox(Q=Q, q=q, const=0.0)

        follower_idx = self._agent_coord_index()
        leader_idx = self._leader_coord_index()
        follower_coord = float(x[follower_idx])
        leader_coord = float(x[leader_idx])

        gate, dgate_dleader = self._gate_and_grad(leader_coord)
        err = self.hold_direction * (follower_coord - self.hold_value)
        hold_cost, grad_err, hess_err = _c2_quadratic_hinge(
            err,
            self.weight,
            self.transition_width,
        )

        grad = np.zeros(nx, dtype=float)
        grad[follower_idx] = gate * grad_err * self.hold_direction
        grad[leader_idx] = hold_cost * dgate_dleader

        # Keep the local curvature PSD. The follower barrier has nonnegative
        # curvature; the leader gate supplies an exact first-order update only.
        Q[follower_idx, follower_idx] = gate * hess_err

        q = grad - Q @ x
        f0 = gate * hold_cost
        const = f0 - grad @ x + 0.5 * x @ Q @ x
        return QuadraticApprox(Q=Q, q=q, const=float(const))


def _c2_quadratic_hinge(err: float, weight: float, transition_width: float):
    """C2 splice from zero cost into 0.5 * weight * err^2."""
    err = float(err)
    weight = float(weight)
    delta = float(transition_width)
    if delta <= 0.0:
        raise ValueError("transition_width must be positive")
    if err <= 0.0 or weight == 0.0:
        return 0.0, 0.0, 0.0
    if err >= delta:
        return (
            0.5 * weight * err ** 2,
            weight * err,
            weight,
        )

    s = err / delta
    value = weight * delta ** 2 * (
        1.5 * s ** 3 - 1.5 * s ** 4 + 0.5 * s ** 5
    )
    grad = weight * delta * (
        4.5 * s ** 2 - 6.0 * s ** 3 + 2.5 * s ** 4
    )
    hess = weight * (
        9.0 * s - 18.0 * s ** 2 + 10.0 * s ** 3
    )
    return float(value), float(grad), float(hess)


class AgentLaneKeepingCost(BaseCost):
    """Quadratic running cost that keeps an agent in a specific lane (x or y).

    Cost: 0.5 * weight * (pos - lane_coord)^2
    """
    def __init__(
        self,
        agent_index: int,
        state_dim_per_agent: int,
        lane_x: float = None,
        lane_y: float = None,
        weight: float = 1.0,
    ):
        self.agent_index = int(agent_index)
        self.state_dim_per_agent = int(state_dim_per_agent)
        self.lane_x = lane_x
        self.lane_y = lane_y
        self.weight = float(weight)
        start = self.agent_index * self.state_dim_per_agent
        self._ix = start
        self._iy = start + 1

    def evaluate(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        start = self.agent_index * self.state_dim_per_agent
        px, py = x[start], x[start+1]
        cost = 0.0
        if self.lane_x is not None:
            cost += 0.5 * self.weight * (px - self.lane_x)**2
        if self.lane_y is not None:
            cost += 0.5 * self.weight * (py - self.lane_y)**2
        return float(cost)

    def quadraticize(self, x, us, k=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        start = self.agent_index * self.state_dim_per_agent
        ix, iy = start, start + 1
        px, py = float(x[ix]), float(x[iy])
        full_grad = np.zeros(nx, dtype=float)
        if self.lane_x is not None:
            err_x = px - self.lane_x
            full_grad[ix] += self.weight * err_x
            Q[ix, ix] += self.weight
        if self.lane_y is not None:
            err_y = py - self.lane_y
            full_grad[iy] += self.weight * err_y
            Q[iy, iy] += self.weight

        q = full_grad - Q @ x
        const = self.evaluate(x, us, k) - full_grad @ x + 0.5 * x @ Q @ x
        return QuadraticApprox(Q=Q, q=q, const=float(const))

    def accumulate_quadratic(self, x, us, k, approx, compute_const=True):
        const = 0.0
        if self.lane_x is not None:
            approx.Q[self._ix, self._ix] += self.weight
            approx.q[self._ix] -= self.weight * self.lane_x
            const += 0.5 * self.weight * self.lane_x ** 2
        if self.lane_y is not None:
            approx.Q[self._iy, self._iy] += self.weight
            approx.q[self._iy] -= self.weight * self.lane_y
            const += 0.5 * self.weight * self.lane_y ** 2
        return const if compute_const else 0.0
