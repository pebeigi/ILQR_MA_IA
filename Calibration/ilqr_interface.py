"""Programmatic single-agent ILQR runner used by the calibrator.

`main.py` in the ILQR package wires a fixed, synthetic six-vehicle scene and
runs everything at import under ``__main__``.  For calibration we instead build
a minimal single-agent game directly from the ILQR building blocks, seeded from
an observed case, so we can solve it many times with different cost weights.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ILQR_ROOT = PROJECT_ROOT / "ILQR_Multi_Agent_IntersectionAnalytical"
if str(ILQR_ROOT) not in sys.path:
    sys.path.insert(0, str(ILQR_ROOT))

from costs.base_cost import (  # noqa: E402
    AgentRunningSpeedCost,
    AgentRunningDestinationCost,
    AgentSpeedDependentControlCost,
    AgentFullTerminalCost,
    BatchedStaticObstacleRepulsionCost,
)
from costs.player_cost import PlayerCost  # noqa: E402
from dynamics.base_dynamics import ConcatenatedDynamics  # noqa: E402
from dynamics.player_dynamics import vehicle_4d  # noqa: E402
from game.game_definition import GameDefinition  # noqa: E402
from game.player import Player  # noqa: E402
from ilq.ilq_solver import ILQSolver  # noqa: E402

STATE_DIM_PER_AGENT = 4
CONTROL_DIM_PER_AGENT = 2
DEFAULT_DT = 0.1


@dataclass(frozen=True)
class AgentParameters:
    """Calibratable behavioral cost weights for one agent.

    These mirror the per-agent weights in the ILQR ``CostWeights``/``AgentSpec``
    structures but exclude scene-specific terms (static obstacles, lane keeping,
    inter-agent repulsion) that do not apply to an isolated calibration case.
    """

    q_speed: float = 100.0
    desired_speed: float = 6.0
    beta_2: float = 1.5
    beta_3: float = 80.0
    running_destination: float = 1.0
    terminal_position_weight: float = 250.0
    terminal_speed_weight: float = 50.0
    terminal_heading_weight: float = 25.0
    v_min: float = 1.0


@dataclass(frozen=True)
class ScenarioSpec:
    """Initial/terminal conditions for a single observed case, in a local frame.

    ``boundary_obstacles`` are optional street-curb repulsion points (in the same
    local frame as the trajectory); when provided the agent is pushed to stay
    within the real road, matching the data's street boundaries.
    """

    initial_state: np.ndarray  # [px, py, speed, heading]
    destination: np.ndarray  # [px, py]
    terminal_heading: float
    horizon_steps: int
    dt: float = DEFAULT_DT
    boundary_obstacles: np.ndarray | None = None  # (M, 2) curb points, local frame
    boundary_weight: float = 1000.0
    boundary_epsilon: float = 0.1


@dataclass(frozen=True)
class SolveResult:
    times: np.ndarray  # (T,)
    states: np.ndarray  # (T, 4) -> [px, py, speed, heading]
    controls: np.ndarray  # (T, 2) -> [kappa, a]
    converged: bool


def build_single_agent_game(
    scenario: ScenarioSpec,
    params: AgentParameters,
    name: str = "calibration_single_agent",
) -> GameDefinition:
    initial_state = np.asarray(scenario.initial_state, dtype=float).reshape(STATE_DIM_PER_AGENT)
    destination = np.asarray(scenario.destination, dtype=float).reshape(2)
    terminal_step = scenario.horizon_steps - 1

    dynamics = ConcatenatedDynamics(
        subsystems=[vehicle_4d],
        state_dims=[STATE_DIM_PER_AGENT],
        control_dims=[CONTROL_DIM_PER_AGENT],
    )
    player = Player(index=0, name="agent", state_dim=STATE_DIM_PER_AGENT, control_dim=CONTROL_DIM_PER_AGENT)

    cost = PlayerCost("agent", state_regularization=1e-6, control_regularization=1e-6)
    cost.add_cost(
        AgentRunningSpeedCost(
            agent_index=0,
            state_dim_per_agent=STATE_DIM_PER_AGENT,
            q_speed=params.q_speed,
            desired_speed=params.desired_speed,
            destination=destination,
            stop_radius=1.0,
            transition_width=5.0,
            stop_heading=scenario.terminal_heading,
        )
    )
    cost.add_cost(
        AgentRunningDestinationCost(
            agent_index=0,
            state_dim_per_agent=STATE_DIM_PER_AGENT,
            destination=destination,
            weight=params.running_destination,
        )
    )
    cost.add_cost(
        AgentSpeedDependentControlCost(
            agent_index=0,
            state_dim_per_agent=STATE_DIM_PER_AGENT,
            player_index=0,
            beta_2=params.beta_2,
            beta_3=params.beta_3,
            v_min=params.v_min,
        )
    )
    cost.add_cost(
        AgentFullTerminalCost(
            agent_index=0,
            state_dim_per_agent=STATE_DIM_PER_AGENT,
            destination=destination,
            desired_speed=0.0,
            desired_heading=scenario.terminal_heading,
            terminal_step=terminal_step,
            b_px=params.terminal_position_weight,
            b_py=params.terminal_position_weight,
            b_speed=params.terminal_speed_weight,
            b_heading=params.terminal_heading_weight,
        )
    )

    # Optional real street-curb boundaries (kept out of the calibrated weights;
    # they describe the scene, not the agent's preferences).
    if scenario.boundary_obstacles is not None and len(scenario.boundary_obstacles) > 0:
        cost.add_cost(
            BatchedStaticObstacleRepulsionCost(
                agent_index=0,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                obstacles=np.asarray(scenario.boundary_obstacles, dtype=float),
                weight=scenario.boundary_weight,
                epsilon=scenario.boundary_epsilon,
            )
        )

    return GameDefinition(
        players=[player],
        dynamics=dynamics,
        player_costs=[cost],
        x0=initial_state,
        dt=scenario.dt,
        horizon_steps=scenario.horizon_steps,
        name=name,
    )


@contextlib.contextmanager
def _suppress_output():
    """Silence the solver's tqdm/print chatter during repeated calibration solves."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def solve_single_agent(
    scenario: ScenarioSpec,
    params: AgentParameters,
    *,
    max_iterations: int = 60,
    alpha_scaling: float = 0.5,
    lq_solver_type: str = "open_loop",
    verbose: bool = False,
) -> SolveResult:
    """Build and solve a single-agent ILQR scenario, returning its trajectory."""

    game = build_single_agent_game(scenario, params)
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

    xs = np.asarray(result["xs"], dtype=float)  # (T, 4)
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
