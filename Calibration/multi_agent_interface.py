"""Programmatic multi-agent ILQR runner for ego calibration.

The ego is the only optimized player.  The co-present neighbors are replayed
from data and enter the ego's optimization as time-varying (moving) obstacles
via :mod:`Calibration.moving_obstacles`.  The behavioral base costs are reused
verbatim from the single-agent builder so the two pipelines stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ilqr_interface import (
    STATE_DIM_PER_AGENT,
    AgentParameters,
    ScenarioSpec,
    SolveResult,
    _suppress_output,
    build_single_agent_game,
)
from .moving_obstacles import MovingObstacleRepulsionCost, MovingProximitySpeedCost

from ilq.ilq_solver import ILQSolver  # noqa: E402  (path set up by ilqr_interface)


@dataclass(frozen=True)
class EgoParameters:
    """Calibratable ego cost weights: behavioral (single-agent) + interaction.

    The first block mirrors :class:`AgentParameters`; the ``neighbor_*`` block
    governs how strongly the ego reacts to the replayed neighbors.
    """

    # Behavioral (shared with the single-agent calibration).
    q_speed: float = 100.0
    desired_speed: float = 6.0
    beta_2: float = 1.5
    beta_3: float = 80.0
    running_destination: float = 1.0
    terminal_position_weight: float = 250.0
    terminal_speed_weight: float = 50.0
    terminal_heading_weight: float = 25.0
    v_min: float = 1.0
    # Interaction with replayed neighbors.
    neighbor_repulsion: float = 200.0
    neighbor_repulsion_epsilon: float = 1.0
    neighbor_proximity_speed: float = 10.0
    neighbor_proximity_epsilon: float = 2.0
    neighbor_activation_distance: float = 12.0

    def behavioral(self) -> AgentParameters:
        return AgentParameters(
            q_speed=self.q_speed,
            desired_speed=self.desired_speed,
            beta_2=self.beta_2,
            beta_3=self.beta_3,
            running_destination=self.running_destination,
            terminal_position_weight=self.terminal_position_weight,
            terminal_speed_weight=self.terminal_speed_weight,
            terminal_heading_weight=self.terminal_heading_weight,
            v_min=self.v_min,
        )


def build_ego_game(
    scenario: ScenarioSpec,
    positions_per_step,
    params: EgoParameters,
    name: str = "calibration_ego",
):
    """Single-player ego game with replayed neighbors as moving obstacles."""
    game = build_single_agent_game(scenario, params.behavioral(), name=name)
    cost = game.player_costs[0]

    if params.neighbor_repulsion != 0.0:
        cost.add_cost(
            MovingObstacleRepulsionCost(
                positions_per_step=positions_per_step,
                agent_index=0,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                weight=params.neighbor_repulsion,
                epsilon=params.neighbor_repulsion_epsilon,
            )
        )
    if params.neighbor_proximity_speed != 0.0:
        cost.add_cost(
            MovingProximitySpeedCost(
                positions_per_step=positions_per_step,
                agent_index=0,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                weight=params.neighbor_proximity_speed,
                epsilon=params.neighbor_proximity_epsilon,
                activation_distance=params.neighbor_activation_distance,
            )
        )
    return game


def solve_ego(
    scenario: ScenarioSpec,
    positions_per_step,
    params: EgoParameters,
    *,
    max_iterations: int = 60,
    alpha_scaling: float = 0.5,
    lq_solver_type: str = "open_loop",
    verbose: bool = False,
) -> SolveResult:
    """Build and solve the ego game among replayed neighbors."""
    game = build_ego_game(scenario, positions_per_step, params)
    solver = ILQSolver(
        game,
        use_euler=True,
        alpha_scaling=alpha_scaling,
        max_iterations=max_iterations,
        lq_solver_type=lq_solver_type,
        alpha_line_search=True,
        alpha_line_search_start_iteration=max(1, max_iterations // 4),
    )
    if verbose:
        result = solver.solve()
    else:
        with _suppress_output():
            result = solver.solve()

    xs = np.asarray(result["xs"], dtype=float)
    us_player = result["us"][0]
    controls = np.asarray(us_player, dtype=float) if len(us_player) else np.zeros((len(xs), 2))
    if controls.shape[0] < xs.shape[0]:
        pad = np.zeros((xs.shape[0] - controls.shape[0], controls.shape[1]))
        controls = np.vstack([controls, pad])

    times = np.arange(xs.shape[0], dtype=float) * scenario.dt
    return SolveResult(
        times=times,
        states=xs,
        controls=controls,
        converged=bool(result.get("converged", False)),
    )
