"""N-player Nash-game runner for joint multi-agent calibration.

Every agent in the :class:`~Calibration.nash_cases.GameScene` is a real ILQR
player with its own cost weights.  Agents are coupled through
``PairwiseAgentRepulsionCost`` and ``AgentProximitySpeedCost`` so the solved
trajectories are a genuine Nash equilibrium (each agent best-responds to the
others), not an optimal-control rollout against fixed obstacles.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ilqr_interface import (
    CONTROL_DIM_PER_AGENT,
    STATE_DIM_PER_AGENT,
    SolveResult,
    _suppress_output,
)
from .nash_cases import GameScene

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ILQR_ROOT = PROJECT_ROOT / "ILQR_Multi_Agent_IntersectionAnalytical"
if str(ILQR_ROOT) not in sys.path:
    sys.path.insert(0, str(ILQR_ROOT))

from costs.base_cost import (  # noqa: E402
    AgentFullTerminalCost,
    AgentProximitySpeedCost,
    AgentRunningDestinationCost,
    AgentRunningSpeedCost,
    AgentSpeedDependentControlCost,
    BatchedStaticObstacleRepulsionCost,
    PairwiseAgentRepulsionCost,
)
from costs.player_cost import PlayerCost  # noqa: E402
from dynamics.base_dynamics import ConcatenatedDynamics  # noqa: E402
from dynamics.player_dynamics import vehicle_4d  # noqa: E402
from game.game_definition import GameDefinition  # noqa: E402
from game.player import Player  # noqa: E402
from ilq.ilq_solver import ILQSolver  # noqa: E402
from ilq.rollout import integrate  # noqa: E402

STOP_RADIUS = 1.0


@dataclass(frozen=True)
class NashAgentParameters:
    """Per-agent cost weights: behavioral + inter-agent interaction.

    The behavioral block matches the single-agent calibration; the ``agent_*``
    block governs how strongly this agent avoids and yields to the *other game
    players* (this is what makes the problem a game).
    """

    # Behavioral.
    q_speed: float = 100.0
    desired_speed: float = 6.0
    beta_2: float = 1.5
    beta_3: float = 80.0
    running_destination: float = 1.0
    terminal_position_weight: float = 250.0
    terminal_speed_weight: float = 50.0
    terminal_heading_weight: float = 25.0
    v_min: float = 1.0
    # Interaction with the other game players.
    agent_repulsion: float = 50.0
    agent_repulsion_epsilon: float = 2.0
    agent_proximity_speed: float = 10.0
    agent_proximity_epsilon: float = 4.0
    agent_activation_distance: float = 12.0


def default_params_for_scene(scene: GameScene) -> list[NashAgentParameters]:
    """Seed each agent's defaults, using its observed mean speed as desired_speed."""
    params: list[NashAgentParameters] = []
    for track in scene.agents:
        ds = float(np.clip(track.mean_speed, 1.0, 15.0)) if track.mean_speed > 0 else 6.0
        params.append(NashAgentParameters(desired_speed=ds))
    return params


def build_nash_game(
    scene: GameScene,
    params_list: list[NashAgentParameters],
    *,
    boundary_obstacles: np.ndarray | None = None,
    boundary_weight: float = 1000.0,
    boundary_epsilon: float = 0.1,
    name: str = "nash_calibration",
) -> GameDefinition:
    n = scene.n_agents
    if len(params_list) != n:
        raise ValueError("params_list must have one entry per agent")
    terminal_step = scene.horizon_steps - 1

    players = [
        Player(
            index=i,
            name=track.name,
            state_dim=STATE_DIM_PER_AGENT,
            control_dim=CONTROL_DIM_PER_AGENT,
        )
        for i, track in enumerate(scene.agents)
    ]
    dynamics = ConcatenatedDynamics(
        subsystems=[vehicle_4d for _ in range(n)],
        state_dims=[STATE_DIM_PER_AGENT for _ in range(n)],
        control_dims=[CONTROL_DIM_PER_AGENT for _ in range(n)],
    )

    destinations = [t.destination.reshape(2) for t in scene.agents]
    terminal_headings = [t.terminal_heading for t in scene.agents]

    player_costs: list[PlayerCost] = []
    for i, track in enumerate(scene.agents):
        p = params_list[i]
        cost = PlayerCost(track.name, state_regularization=1e-6, control_regularization=1e-6)
        cost.add_cost(
            AgentRunningSpeedCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                q_speed=p.q_speed,
                desired_speed=p.desired_speed,
                destination=destinations[i],
                stop_radius=STOP_RADIUS,
                transition_width=5.0,
                stop_heading=terminal_headings[i],
            )
        )
        cost.add_cost(
            AgentRunningDestinationCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                destination=destinations[i],
                weight=p.running_destination,
            )
        )
        cost.add_cost(
            AgentSpeedDependentControlCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                player_index=i,
                beta_2=p.beta_2,
                beta_3=p.beta_3,
                v_min=p.v_min,
            )
        )
        cost.add_cost(
            AgentFullTerminalCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                destination=destinations[i],
                desired_speed=0.0,
                desired_heading=terminal_headings[i],
                terminal_step=terminal_step,
                b_px=p.terminal_position_weight,
                b_py=p.terminal_position_weight,
                b_speed=p.terminal_speed_weight,
                b_heading=p.terminal_heading_weight,
            )
        )
        if boundary_obstacles is not None and len(boundary_obstacles) > 0:
            cost.add_cost(
                BatchedStaticObstacleRepulsionCost(
                    agent_index=i,
                    state_dim_per_agent=STATE_DIM_PER_AGENT,
                    obstacles=np.asarray(boundary_obstacles, dtype=float),
                    weight=boundary_weight,
                    epsilon=boundary_epsilon,
                )
            )

        # Inter-agent coupling — this is what makes it a Nash game.
        for j in range(n):
            if j == i:
                continue
            if p.agent_repulsion != 0.0:
                cost.add_cost(
                    PairwiseAgentRepulsionCost(
                        agent_index=i,
                        other_agent_index=j,
                        state_dim_per_agent=STATE_DIM_PER_AGENT,
                        destinations=destinations,
                        stop_radius=STOP_RADIUS,
                        weight=p.agent_repulsion,
                        epsilon=p.agent_repulsion_epsilon,
                        active_only_when_both_moving=True,
                        stop_headings=terminal_headings,
                    )
                )
            if p.agent_proximity_speed != 0.0:
                cost.add_cost(
                    AgentProximitySpeedCost(
                        agent_index=i,
                        other_agent_index=j,
                        state_dim_per_agent=STATE_DIM_PER_AGENT,
                        weight=p.agent_proximity_speed,
                        epsilon=p.agent_proximity_epsilon,
                        activation_distance=p.agent_activation_distance,
                    )
                )
        player_costs.append(cost)

    x0 = np.concatenate([t.initial_state.reshape(STATE_DIM_PER_AGENT) for t in scene.agents])
    return GameDefinition(
        players=players,
        dynamics=dynamics,
        player_costs=player_costs,
        x0=x0,
        dt=scene.dt,
        horizon_steps=scene.horizon_steps,
        name=name,
    )


def _agent_slice(i: int) -> slice:
    start = i * STATE_DIM_PER_AGENT
    return slice(start, start + STATE_DIM_PER_AGENT)


def make_nash_warm_start(
    game: GameDefinition,
    scene: GameScene,
    params_list: list[NashAgentParameters],
):
    """Destination-seeking nominal trajectory (no hard-coded avoidance).

    Late-entering agents hold position until their entry step, then steer toward
    their destination at their desired speed.
    """
    xs = [game.x0.copy()]
    us = [[] for _ in range(game.num_players)]
    t = 0.0
    max_brake = 4.0

    for k in range(game.horizon_steps):
        xk = xs[-1]
        u_k = []
        for i, track in enumerate(scene.agents):
            xi = xk[_agent_slice(i)]
            p = params_list[i]
            if k < track.entry_step:
                ui = np.array([0.0, np.clip(-xi[2], -max_brake, 0.0)], dtype=float)
                us[i].append(ui)
                u_k.append(ui)
                continue

            delta = track.destination - xi[:2]
            distance = float(np.hypot(delta[0], delta[1]))
            heading = float(xi[3])
            if distance < 0.5:
                desired_speed = 0.0
            else:
                desired_speed = min(p.desired_speed, max(0.4, 0.45 * distance))
            fwd_left = np.array([-np.sin(heading), np.cos(heading)])
            lateral = float(delta @ fwd_left)
            kappa = float(np.clip(2.0 * lateral / max(distance ** 2, 1e-3), -0.5, 0.5))
            accel = float(np.clip(2.0 * (desired_speed - xi[2]), -max_brake, 1.5))
            ui = np.array([kappa, accel], dtype=float)
            us[i].append(ui)
            u_k.append(ui)

        if k < game.horizon_steps - 1:
            xs.append(integrate(game.dynamics.evaluate, t, game.dt, xk, u_k, use_euler=True))
            t += game.dt

    costs = [
        [
            game.player_costs[i].evaluate(xs[k], [us[j][k] for j in range(game.num_players)], k)
            for k in range(game.horizon_steps)
        ]
        for i in range(game.num_players)
    ]
    return xs, us, costs


def solve_nash(
    scene: GameScene,
    params_list: list[NashAgentParameters],
    *,
    boundary_obstacles: np.ndarray | None = None,
    boundary_weight: float = 1000.0,
    boundary_epsilon: float = 0.1,
    max_iterations: int = 60,
    alpha_scaling: float = 0.5,
    lq_solver_type: str = "open_loop",
    warm_start: bool = True,
    verbose: bool = False,
) -> list[SolveResult]:
    """Solve the joint game; return one SolveResult per agent (in scene order)."""
    game = build_nash_game(
        scene,
        params_list,
        boundary_obstacles=boundary_obstacles,
        boundary_weight=boundary_weight,
        boundary_epsilon=boundary_epsilon,
    )
    solver = ILQSolver(
        game,
        use_euler=True,
        alpha_scaling=alpha_scaling,
        max_iterations=max_iterations,
        lq_solver_type=lq_solver_type,
        alpha_line_search=True,
        alpha_line_search_start_iteration=max(1, max_iterations // 4),
    )
    if warm_start:
        solver.current_operating_point = make_nash_warm_start(game, scene, params_list)

    if verbose:
        result = solver.solve()
    else:
        with _suppress_output():
            result = solver.solve()

    xs = np.asarray(result["xs"], dtype=float)  # (T, n*4)
    times = np.arange(xs.shape[0], dtype=float) * scene.dt
    results: list[SolveResult] = []
    for i in range(scene.n_agents):
        sl = _agent_slice(i)
        states = xs[:, sl]
        us_player = result["us"][i]
        controls = np.asarray(us_player, dtype=float) if len(us_player) else np.zeros((len(xs), 2))
        if controls.shape[0] < states.shape[0]:
            pad = np.zeros((states.shape[0] - controls.shape[0], controls.shape[1]))
            controls = np.vstack([controls, pad])
        results.append(
            SolveResult(
                times=times,
                states=states,
                controls=controls,
                converged=bool(result.get("converged", False)),
            )
        )
    return results
