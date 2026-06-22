"""Extendable multi-agent ILQR vehicle experiment.

State for each agent:   [px, py, speed, heading]
Control for each agent: [kappa, a]   (curvature rad/m, acceleration m/s²)

Minimal cost function:
  Running:  β₁·||p − dest||²          (destination spring, AgentRunningDestinationCost)
            β₂/2·κ²·v⁴ + β₃/2·a²     (state-dependent control cost, R(v))
            weight / (||p − obs||² + ε) (static obstacle repulsion)
            weight / (||pᵢ − pⱼ||² + ε) (pairwise agent repulsion)
  Yielding: weight·v / (||pᵢ − pⱼ||² + ε) (proximity speed cost)
            smooth leader-gated yield line for a through follower behind an LT leader

To add/remove agents, edit only the AGENTS list below.  Use leader_name for
direct leaders so inserting/removing unrelated agents does not silently change
platoon relationships.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import csv
import time
import numpy as np

from costs.base_cost import (
    AgentRunningSpeedCost,
    AgentRunningDestinationCost,
    AgentArrivalHoldCost,
    AgentFullTerminalCost,
    AgentSpeedDependentControlCost,
    PairwiseAgentRepulsionCost,
    BatchedStaticObstacleRepulsionCost,
    AgentProximitySpeedCost,
    AgentLeaderYieldLineCost,
    AgentLaneKeepingCost,
)
from costs.player_cost import PlayerCost
from dynamics.player_dynamics import vehicle_4d
from dynamics.base_dynamics import ConcatenatedDynamics
from game.player import Player
from game.game_definition import GameDefinition
from ilq.ilq_solver import ILQSolver
from ilq.rollout import integrate


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
DT = 0.1
HORIZON_STEPS = 200     # 20s — through vehicle at y=38 needs ~10s, LT needs ~12s
STOP_RADIUS = 2.0
PASS_THROUGH_RADIUS = 1.0
STOP_AFTER_ARRIVAL_STEPS = 5
STATE_DIM_PER_AGENT = 4
CONTROL_DIM_PER_AGENT = 2

# Intersection geometry.
INTERSECTION_X_MIN = 0.0
INTERSECTION_X_MAX = 22.0
NS_DIVIDER_X = 11.0
SOUTHBOUND_LANE_X = 5.5
NORTHBOUND_LANE_X = 16.5
INTERSECTION_Y_MIN = 7.0
INTERSECTION_Y_MAX = 21.0
EW_LANE_Y = 15.0
WEST_ARM_X_MIN = -20.0
EAST_ARM_X_MAX = 40.0
WESTBOUND_DEST_X = -18.0
EASTBOUND_DEST_X = 37.0
RIGHT_TURN_WESTBOUND_DEST_X = WESTBOUND_DEST_X + 12.0
RIGHT_TURN_EASTBOUND_DEST_X = EASTBOUND_DEST_X - 6.0
ROAD_SOUTH_Y_MIN = -28.0
ROAD_NORTH_Y_MAX = 68.0


@dataclass
class CostWeights:
    """Cost weights: speed tracking + destination spring + control + repulsion + road geometry."""
    q_speed: float = 100.0           # Speed tracking — must dominate running_destination
    beta_2: float = 1.5
    beta_3: float = 80.0
    v_min: float = 6.0
    running_destination: float = 1.0  # Kept small so it doesn't incentivise speeding up
    static_obstacle_repulsion: float = 1000.0
    static_obstacle_epsilon: float = 0.1
    # Pairwise repulsion — two semantically distinct buckets:
    #   leader_repulsion       : j is this agent's direct leader (gap-keeping)
    #   cross_traffic_repulsion: j is in a different platoon stream (yielding / collision avoidance)
    leader_repulsion: float = 0.0
    leader_proximity_speed_weight: float = 0.0
    leader_proximity_speed_epsilon: float = 4.0
    leader_proximity_speed_activation_distance: float | None = None
    leader_proximity_speed_activation_width: float = 2.0
    leader_yield_line_weight: float = 0.0
    leader_yield_line_hold_y: float = 5.0
    leader_yield_line_clear_x: float = 8.0
    leader_yield_line_transition: float = 0.5
    leader_yield_line_clearance_width: float = 2.5
    cross_traffic_repulsion: float = 0.0
    repulsion_epsilon: float = 2.0
    # proximity_speed_weight fires only for cross-traffic pairs (same semantics as
    # cross_traffic_repulsion — slows the agent down when a cross-stream vehicle is close)
    proximity_speed_weight: float = 0.0
    proximity_speed_epsilon: float = 4.0
    lane_x: float | None = None      # Soft lane-centre x constraint (None = disabled)
    lane_x_weight: float = 0.0
    arrival_speed_transition: float = 5.0  # Signed-distance taper to zero target speed before destination
    arrival_hold_transition: float = 2.0
    arrival_hold_position_weight: float = 50.0
    arrival_hold_speed_weight: float = 50.0
    terminal_position_weight: float = 250.0
    terminal_speed_weight: float = 50.0
    terminal_heading_weight: float = 25.0


@dataclass
class AgentSpec:
    """Definition of one vehicle/agent."""
    name: str
    initial_state: np.ndarray
    destination: np.ndarray
    cost: CostWeights = field(default_factory=CostWeights)
    desired_speed: float = 6.0
    warm_start_delay_steps: int = 0   # Hold position for this many steps before seeking destination
    leader_index: int | None = None   # Resolved direct-leader index; None = platoon root
    leader_name: str | None = None    # Preferred direct-leader reference, stable under AGENTS reordering
    terminal_heading: float | None = None


# Six-vehicle scenario from the working right-turn checkpoint:
#   - Base southbound through + follower and northbound left-turn + follower.
#   - One right-turn vehicle is included on each approach.
# Road geometry: N-S road x∈[0,22] with divider at x=11.
#   SB lane center x=5.5, NB lane center x=16.5.
#   E-W road y∈[7,21], lane center y=15.
AGENTS = [
    # Southbound through vehicle: front of the SB stream.
    AgentSpec(
        name="through_southbound",
        initial_state=np.array([SOUTHBOUND_LANE_X, 34.0, 6.0, -np.pi / 2.0]),
        destination=np.array([SOUTHBOUND_LANE_X, -28.0]),
        desired_speed=6.0,
        leader_name=None,            # Root of SB platoon
        terminal_heading=-np.pi / 2.0,
        cost=CostWeights(
            q_speed=200.0,
            beta_2=1.5,
            beta_3=80.0,
            v_min=6.0,
            running_destination=1.0,
            static_obstacle_repulsion=1000.0,
            static_obstacle_epsilon=0.1,
            leader_repulsion=0.0,
            cross_traffic_repulsion=0.0,   # Through vehicle has right-of-way — does not yield
            proximity_speed_weight=0.0,
            lane_x=SOUTHBOUND_LANE_X,
            lane_x_weight=200.0,
        ),
    ),
    # Left Turn Vehicle: NB lane x=16.5, turns west.
    AgentSpec(
        name="left_turn_northbound",
        initial_state=np.array([NORTHBOUND_LANE_X, -7.0, 6.0, np.pi / 2.0]),
        destination=np.array([WESTBOUND_DEST_X, EW_LANE_Y]),
        desired_speed=6.0,
        leader_name=None,            # Root of LT platoon
        terminal_heading=np.pi,
        cost=CostWeights(
            q_speed=20.0,            # Weak enough to allow genuine yielding
            beta_2=10,               # LOW β₂ is critical — high β₂ makes rightward swing WORSE
            beta_3=8.0,
            v_min=1.0,
            running_destination=1.0,
            static_obstacle_repulsion=1000,
            static_obstacle_epsilon=0.1,
            leader_repulsion=0.0,
            cross_traffic_repulsion=20.0,  # Yields to SB stream (through + follower)
            proximity_speed_weight=3000,   # Slows down near SB cross-traffic only; 10000 caused OverflowError
            proximity_speed_epsilon=2.0,
        ),
    ),
    # Southbound through follower: follows the through vehicle in the SB lane.
    AgentSpec(
        name="follower_southbound",
        initial_state=np.array([SOUTHBOUND_LANE_X, 46.0, 6.0, -np.pi / 2.0]),
        destination=np.array([SOUTHBOUND_LANE_X, -20.0]),
        desired_speed=6.0,
        leader_name="through_southbound",
        terminal_heading=-np.pi / 2.0,
        cost=CostWeights(
            q_speed=200.0,
            beta_2=1.5,
            beta_3=80.0,
            v_min=6.0,
            running_destination=1.0,
            static_obstacle_repulsion=1000.0,
            static_obstacle_epsilon=0.1,
            leader_repulsion=150,
            leader_proximity_speed_weight=1500.0,
            leader_proximity_speed_epsilon=8.0,
            cross_traffic_repulsion=0.0,
            proximity_speed_weight=0.0,
            lane_x=SOUTHBOUND_LANE_X,
            lane_x_weight=200.0,
        ),
    ),
    # Follower of left-turn vehicle: same NB lane, starts 8m behind leader, also turns left.
    AgentSpec(
        name="follower_left_turn",
        initial_state=np.array([NORTHBOUND_LANE_X, -15.0, 6.0, np.pi / 2.0]),
        destination=np.array([WESTBOUND_DEST_X + 6.0, EW_LANE_Y]),
        desired_speed=6.0,
        leader_name="left_turn_northbound",
        terminal_heading=np.pi,
        cost=CostWeights(
            q_speed=10,
            beta_2=15,
            beta_3=4,
            v_min=1.0,
            running_destination=1.0,
            static_obstacle_repulsion=1000,
            static_obstacle_epsilon=0.1,
            leader_repulsion=150,
            leader_proximity_speed_weight=1500.0,
            leader_proximity_speed_epsilon=8.0,
            cross_traffic_repulsion=100,
            proximity_speed_weight=2000,   # Slows down near SB cross-traffic only
            proximity_speed_epsilon=2.0,
        ),
    ),
    # Southbound right-turn vehicle: follows the SB follower, then turns west.
    AgentSpec(
        name="right_turn_southbound",
        initial_state=np.array([SOUTHBOUND_LANE_X, 55, 6.0, -np.pi / 2.0]),
        destination=np.array([RIGHT_TURN_WESTBOUND_DEST_X, EW_LANE_Y]),
        desired_speed=6.0,
        leader_name="follower_southbound",
        terminal_heading=np.pi,
        cost=CostWeights(
            q_speed=20.0,
            beta_2=10.0,
            beta_3=8.0,
            v_min=1.0,
            running_destination=1.0,
            static_obstacle_repulsion=1000.0,
            static_obstacle_epsilon=0.1,
            leader_repulsion=200,
            leader_proximity_speed_weight=1500.0,
            leader_proximity_speed_epsilon=8.0,
            cross_traffic_repulsion=20.0,
            proximity_speed_weight=3000.0,
            proximity_speed_epsilon=2.0,
        ),
    ),
    # Right-turn vehicle in NB lane: queues behind the left-turn follower, then turns east.
    AgentSpec(
        name="right_turn_northbound",
        initial_state=np.array([NORTHBOUND_LANE_X, -23.0, 6.0, np.pi / 2.0]),
        destination=np.array([RIGHT_TURN_EASTBOUND_DEST_X, EW_LANE_Y]),
        desired_speed=6.0,
        leader_name="follower_left_turn",
        terminal_heading=0.0,
        cost=CostWeights(
            q_speed=20.0,
            beta_2=10.0,
            beta_3=8.0,
            v_min=1.0,
            running_destination=1.0,
            static_obstacle_repulsion=1000.0,
            static_obstacle_epsilon=0.1,
            leader_repulsion=200,
            leader_proximity_speed_weight=1500.0,
            leader_proximity_speed_epsilon=8.0,
            cross_traffic_repulsion=20.0,
            proximity_speed_weight=3000.0,
            proximity_speed_epsilon=2.0,
        ),
    ),
]


# Intersection boundary obstacles.
# Road layout (symmetric 4-way cross intersection):
#   N-S road x∈[0,22].  SB lane x∈[0,11], NB lane x∈[11,22].
#   Lane centres: SB x=5.5, NB x=16.5.  Lane divider x=11 stops before the
#   intersection and exists on both the south and north approaches.
#   E-W road y∈[7,21], west arm x∈[-20,0], east arm x∈[22,40].
#   Intersection box: x∈[0,22], y∈[7,21].
#
# Obstacle spacing 2 m.  Weight=1000, epsilon=0.1 (set in CostWeights).
# Divider leaves a 1m gap before the intersection on each side so vehicles can
# turn freely inside.
_SOUTH_APPROACH_OBS_YS = np.arange(-10.0, INTERSECTION_Y_MIN, 2.0)
_NORTH_APPROACH_OBS_YS = np.arange(INTERSECTION_Y_MAX + 1.0, 55.0, 2.0)
_WEST_ARM_OBS_XS = np.arange(-2.0, WEST_ARM_X_MIN - 0.1, -2.0)
_EAST_ARM_OBS_XS = np.arange(INTERSECTION_X_MAX + 2.0, EAST_ARM_X_MAX + 0.1, 2.0)

STATIC_OBSTACLES = [
    [INTERSECTION_X_MIN, INTERSECTION_Y_MIN],
    [INTERSECTION_X_MIN, INTERSECTION_Y_MAX],
    *[[NS_DIVIDER_X, y] for y in _SOUTH_APPROACH_OBS_YS],
    *[[NS_DIVIDER_X, y] for y in _NORTH_APPROACH_OBS_YS],
    *[[INTERSECTION_X_MIN, y] for y in _SOUTH_APPROACH_OBS_YS],
    *[[INTERSECTION_X_MAX, y] for y in _SOUTH_APPROACH_OBS_YS],
    *[[INTERSECTION_X_MIN, y] for y in _NORTH_APPROACH_OBS_YS],
    *[[INTERSECTION_X_MAX, y] for y in _NORTH_APPROACH_OBS_YS],
    *[[x, INTERSECTION_Y_MIN] for x in _WEST_ARM_OBS_XS],
    *[[x, INTERSECTION_Y_MAX] for x in _WEST_ARM_OBS_XS],
    *[[x, INTERSECTION_Y_MIN] for x in _EAST_ARM_OBS_XS],
    *[[x, INTERSECTION_Y_MAX] for x in _EAST_ARM_OBS_XS],
]

# ILQR settings.
MAX_ITERATIONS = 400
ALPHA_SCALING = 0.5
USE_ALPHA_LINE_SEARCH = True
ALPHA_LINE_SEARCH_MIN = 0.03125
ALPHA_LINE_SEARCH_SHRINK = 0.5
ALPHA_LINE_SEARCH_MAX_GROWTH = 5.0
ALPHA_LINE_SEARCH_START_ITERATION = 40
CONVERGENCE_TOL = 1e-2
RELATIVE_COST_CONVERGENCE_TOL = 1e-6
CONVERGENCE_PATIENCE = 3
LQ_SOLVER_TYPE = "open_loop"  # Set to "feedback" or "open_loop"
_OUTPUTS_DIR = Path(__file__).parent / "outputs"
_OUTPUTS_DIR.mkdir(exist_ok=True)
OUTPUT_PREFIX = str(_OUTPUTS_DIR / f"multi_agent_trajectory_{LQ_SOLVER_TYPE}")
PLOT_DPI = 400
PLOT_FONT_SERIF = ["Times New Roman", "Times", "DejaVu Serif"]
PLOT_LINE_WIDTH = 1.8
PLOT_MARKER_SIZE = 2.4
CONVERGENCE_MARKER_SIZE = 13
CONVERGENCE_MARKER_ALPHA = 0.50


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def agent_slice(agent_index: int) -> slice:
    start = agent_index * STATE_DIM_PER_AGENT
    return slice(start, start + STATE_DIM_PER_AGENT)


def agent_position(x: np.ndarray, agent_index: int) -> np.ndarray:
    return np.asarray(x[agent_slice(agent_index)][:2], dtype=float)


def agent_destination(agent_index: int) -> np.ndarray:
    return np.asarray(AGENTS[agent_index].destination, dtype=float).reshape(2)


def distance_to_destination(x: np.ndarray, agent_index: int) -> float:
    return float(np.linalg.norm(agent_position(x, agent_index) - agent_destination(agent_index)))


def desired_terminal_heading(agent: AgentSpec) -> float:
    if agent.terminal_heading is not None:
        return float(agent.terminal_heading)
    delta = np.asarray(agent.destination, dtype=float).reshape(2) - np.asarray(agent.initial_state[:2], dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def vehicle_corners(pos: np.ndarray, heading: float,
                    length: float = 4.0, width: float = 2.0) -> np.ndarray:
    """Return (4, 2) array of rectangle corners for a vehicle body."""
    fwd  = np.array([ np.cos(heading),  np.sin(heading)]) * (length / 2)
    side = np.array([-np.sin(heading),  np.cos(heading)]) * (width  / 2)
    return np.array([pos + fwd + side,
                     pos + fwd - side,
                     pos - fwd - side,
                     pos - fwd + side])


# -----------------------------------------------------------------------------
# Game construction
# -----------------------------------------------------------------------------
def resolve_leader_references(agent_specs: list[AgentSpec]) -> list[AgentSpec]:
    """Return specs with leader_index resolved from leader_name and validated."""
    name_to_index: dict[str, int] = {}
    for i, agent in enumerate(agent_specs):
        if agent.name in name_to_index:
            raise ValueError(f"duplicate agent name {agent.name!r}")
        name_to_index[agent.name] = i

    resolved: list[AgentSpec] = []
    for i, agent in enumerate(agent_specs):
        leader_index = agent.leader_index
        if agent.leader_name is not None:
            if agent.leader_name not in name_to_index:
                raise ValueError(
                    f"agent {agent.name!r} references missing leader_name {agent.leader_name!r}"
                )
            named_index = name_to_index[agent.leader_name]
            if leader_index is not None and leader_index != named_index:
                raise ValueError(
                    f"agent {agent.name!r} has conflicting leader_index={leader_index} "
                    f"and leader_name={agent.leader_name!r}"
                )
            leader_index = named_index

        if leader_index is not None:
            if not 0 <= leader_index < len(agent_specs):
                raise ValueError(
                    f"agent {agent.name!r} has leader_index={leader_index}, "
                    f"but there are only {len(agent_specs)} agents"
                )
            if leader_index == i:
                raise ValueError(f"agent {agent.name!r} cannot be its own leader")

        resolved.append(replace(agent, leader_index=leader_index))

    for i, agent in enumerate(resolved):
        seen = {i}
        path = [i]
        current = i
        while resolved[current].leader_index is not None:
            current = resolved[current].leader_index
            if current in seen:
                path.append(current)
                chain = " -> ".join(resolved[idx].name for idx in path)
                raise ValueError(f"leader cycle detected: {chain}")
            seen.add(current)
            path.append(current)

    for agent in resolved:
        w = agent.cost
        has_leader_specific_cost = (
            w.leader_repulsion > 0.0
            or w.leader_proximity_speed_weight > 0.0
            or w.leader_yield_line_weight > 0.0
        )
        if has_leader_specific_cost and agent.leader_index is None:
            raise ValueError(
                f"agent {agent.name!r} has leader-specific costs but no leader_name/leader_index"
            )

    return resolved


def _platoon_root(agent_specs: list[AgentSpec], i: int) -> int:
    """Walk the leader_index chain upward to find the root of agent i's platoon."""
    seen = set()
    while agent_specs[i].leader_index is not None:
        i = agent_specs[i].leader_index
        if i in seen:
            break
        seen.add(i)
    return i


def build_multi_agent_game(agent_specs: list[AgentSpec]) -> GameDefinition:
    """Build an ILQ game from an arbitrary list of AgentSpec objects."""
    if len(agent_specs) == 0:
        raise ValueError("agent_specs must contain at least one agent")
    agent_specs = resolve_leader_references(agent_specs)

    players = [
        Player(index=i, name=agent.name, state_dim=STATE_DIM_PER_AGENT, control_dim=CONTROL_DIM_PER_AGENT)
        for i, agent in enumerate(agent_specs)
    ]

    dynamics = ConcatenatedDynamics(
        subsystems=[vehicle_4d for _ in agent_specs],
        state_dims=[STATE_DIM_PER_AGENT for _ in agent_specs],
        control_dims=[CONTROL_DIM_PER_AGENT for _ in agent_specs],
    )

    destinations = [np.asarray(agent.destination, dtype=float).reshape(2) for agent in agent_specs]
    terminal_headings = [desired_terminal_heading(agent) for agent in agent_specs]

    player_costs = []
    for i, agent in enumerate(agent_specs):
        w = agent.cost
        cost = PlayerCost(
            agent.name,
            state_regularization=1e-6,
            control_regularization=1e-6,
        )

        # Speed tracking: (q_speed/2)·(v − v_desired)²
        # Necessary to prevent the destination spring from incentivizing
        # unlimited acceleration (minimizing travel time at any cost).
        cost.add_cost(
            AgentRunningSpeedCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                q_speed=w.q_speed,
                desired_speed=agent.desired_speed,
                destination=agent.destination,
                stop_radius=PASS_THROUGH_RADIUS,
                transition_width=w.arrival_speed_transition,
                stop_heading=terminal_headings[i],
            )
        )

        # β₁·||p − dest||²  (destination spring)
        cost.add_cost(
            AgentRunningDestinationCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                destination=agent.destination,
                weight=w.running_destination,
            )
        )

        # Post-arrival hold: once the agent reaches the terminal plane, keep it
        # parked there inside the optimization instead of only freezing outputs.
        cost.add_cost(
            AgentArrivalHoldCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                destination=agent.destination,
                stop_heading=terminal_headings[i],
                stop_radius=PASS_THROUGH_RADIUS,
                transition_width=w.arrival_hold_transition,
                position_weight=w.arrival_hold_position_weight,
                speed_weight=w.arrival_hold_speed_weight,
            )
        )

        # Static obstacle repulsion: sum_i weight/(||p-obs_i||²+ε) — batched for speed
        cost.add_cost(
            BatchedStaticObstacleRepulsionCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                obstacles=STATIC_OBSTACLES,
                weight=w.static_obstacle_repulsion,
                epsilon=w.static_obstacle_epsilon,
            )
        )

        # Pairwise agent repulsion + optional proximity speed cost (yielding).
        # Weight selection per pair:
        #   leader_repulsion        — j is this agent's direct leader (gap-keeping)
        #   cross_traffic_repulsion — j is in a different platoon (yielding)
        # Same-platoon non-leader pairs are not coupled for yielding.
        # proximity_speed_weight fires only for cross-traffic pairs; the separate
        # leader_proximity_speed_weight slows a follower near its direct leader.
        root_i = _platoon_root(agent_specs, i)
        for j in range(len(agent_specs)):
            if j == i:
                continue
            is_my_leader = (j == agent_specs[i].leader_index)
            same_platoon = (root_i == _platoon_root(agent_specs, j))
            rep_w = (
                w.leader_repulsion
                if is_my_leader
                else 0.0 if same_platoon else w.cross_traffic_repulsion
            )
            if rep_w != 0.0:
                cost.add_cost(
                    PairwiseAgentRepulsionCost(
                        agent_index=i,
                        other_agent_index=j,
                        state_dim_per_agent=STATE_DIM_PER_AGENT,
                        destinations=destinations,
                        stop_radius=STOP_RADIUS,
                        weight=rep_w,
                        epsilon=w.repulsion_epsilon,
                        active_only_when_both_moving=True,
                        stop_headings=terminal_headings,
                    )
                )
            if not same_platoon and w.proximity_speed_weight > 0.0:
                cost.add_cost(
                    AgentProximitySpeedCost(
                        agent_index=i,
                        other_agent_index=j,
                        state_dim_per_agent=STATE_DIM_PER_AGENT,
                        weight=w.proximity_speed_weight,
                        epsilon=w.proximity_speed_epsilon,
                    )
                )
            if is_my_leader and w.leader_proximity_speed_weight > 0.0:
                cost.add_cost(
                    AgentProximitySpeedCost(
                        agent_index=i,
                        other_agent_index=j,
                        state_dim_per_agent=STATE_DIM_PER_AGENT,
                        weight=w.leader_proximity_speed_weight,
                        epsilon=w.leader_proximity_speed_epsilon,
                        activation_distance=w.leader_proximity_speed_activation_distance,
                        activation_width=w.leader_proximity_speed_activation_width,
                    )
                )
            if is_my_leader and w.leader_yield_line_weight > 0.0:
                cost.add_cost(
                    AgentLeaderYieldLineCost(
                        agent_index=i,
                        leader_index=j,
                        state_dim_per_agent=STATE_DIM_PER_AGENT,
                        hold_axis=1,             # follower y coordinate
                        hold_value=w.leader_yield_line_hold_y,
                        hold_direction=1.0,      # active when follower moves north past hold_y
                        clear_axis=0,            # leader x coordinate
                        clear_value=w.leader_yield_line_clear_x,
                        clear_direction=1.0,     # active while leader is still east/right of clear_x
                        weight=w.leader_yield_line_weight,
                        transition_width=w.leader_yield_line_transition,
                        clearance_width=w.leader_yield_line_clearance_width,
                    )
                )

        # Soft lane-centre constraint: keeps agent in its designated lane.
        if w.lane_x is not None and w.lane_x_weight > 0.0:
            cost.add_cost(
                AgentLaneKeepingCost(
                    agent_index=i,
                    state_dim_per_agent=STATE_DIM_PER_AGENT,
                    lane_x=w.lane_x,
                    weight=w.lane_x_weight,
                )
            )

        # β₂/2·κ²·v⁴ + β₃/2·a²  (state-dependent control cost)
        cost.add_cost(
            AgentSpeedDependentControlCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                player_index=i,
                beta_2=w.beta_2,
                beta_3=w.beta_3,
                v_min=w.v_min,
            )
        )
        cost.add_cost(
            AgentFullTerminalCost(
                agent_index=i,
                state_dim_per_agent=STATE_DIM_PER_AGENT,
                destination=agent.destination,
                desired_speed=0.0,
                desired_heading=desired_terminal_heading(agent),
                terminal_step=HORIZON_STEPS - 1,
                b_px=w.terminal_position_weight,
                b_py=w.terminal_position_weight,
                b_speed=w.terminal_speed_weight,
                b_heading=w.terminal_heading_weight,
            )
        )
        player_costs.append(cost)

    x0 = np.concatenate([np.asarray(agent.initial_state, dtype=float).reshape(STATE_DIM_PER_AGENT) for agent in agent_specs])

    return GameDefinition(
        players=players,
        dynamics=dynamics,
        player_costs=player_costs,
        x0=x0,
        dt=DT,
        horizon_steps=HORIZON_STEPS,
        name="extendable_multi_agent_vehicle_game",
        metadata={
            "state_order_per_agent": "[px, py, speed, heading]",
            "control_order_per_agent": "[kappa, a]",
            "stop_radius": STOP_RADIUS,
            "static_obstacles": [np.asarray(o, dtype=float).tolist() for o in STATIC_OBSTACLES],
            "agents": [
                {
                    "name": agent.name,
                    "initial_state": np.asarray(agent.initial_state, dtype=float).tolist(),
                    "destination": np.asarray(agent.destination, dtype=float).tolist(),
                    "cost_weights": agent.cost.__dict__.copy(),
                }
                for agent in agent_specs
            ],
        },
    )


# -----------------------------------------------------------------------------
# Initial nominal trajectory
# -----------------------------------------------------------------------------
def make_initial_nominal_trajectory(game: GameDefinition, agent_specs: list[AgentSpec]):
    """Create a feasible destination-seeking nominal trajectory.

    This warm start does not contain hard-coded inter-agent avoidance.  Pairwise
    avoidance should come from PairwiseAgentRepulsionCost, so changing that
    coefficient changes the optimized trajectory.
    """
    xs = [game.x0.copy()]
    us = [[] for _ in range(game.num_players)]
    t = 0.0
    terminal_headings = [desired_terminal_heading(agent) for agent in agent_specs]
    terminal_directions = [
        np.array([np.cos(heading), np.sin(heading)], dtype=float)
        for heading in terminal_headings
    ]

    for k in range(game.horizon_steps):
        xk = xs[-1]
        u_k = []
        for i, agent in enumerate(agent_specs):
            sl = agent_slice(i)
            xi = xk[sl]
            destination = np.asarray(agent.destination, dtype=float)
            delta = destination - xi[:2]
            distance = float(np.linalg.norm(delta))
            terminal_direction = terminal_directions[i]
            remaining = float(delta @ terminal_direction)
            lateral_error = float(np.linalg.norm(delta - remaining * terminal_direction))
            max_warm_start_brake = 4.0
            warm_start_stop_radius = 0.25
            stop_buffer = warm_start_stop_radius
            if lateral_error < PASS_THROUGH_RADIUS:
                stopping_speed = np.sqrt(max(0.0, 2.0 * max_warm_start_brake * max(remaining - stop_buffer, 0.0)))
            else:
                stopping_speed = agent.desired_speed

            if k < agent.warm_start_delay_steps:
                # Hold position: brake to zero and keep current heading.
                kappa = 0.0
                acceleration = np.clip(2.0 * (0.0 - xi[2]), -max_warm_start_brake, 0.0)
                ui = np.array([kappa, acceleration], dtype=float)
                us[i].append(ui)
                u_k.append(ui)
                continue
            if distance < warm_start_stop_radius or (remaining <= 0.0 and lateral_error < PASS_THROUGH_RADIUS):
                kappa = 0.0
                desired_speed = 0.0
                acceleration = np.clip(2.0 * (desired_speed - xi[2]), -max_warm_start_brake, 1.5)
            else:
                distance_speed = max(0.4, 0.45 * distance)
                desired_speed = min(agent.desired_speed, distance_speed, stopping_speed)
                effective_dest = np.asarray(agent.destination, dtype=float)
                heading = float(xi[3])

                target_heading = terminal_headings[i]
                turns_to_east_west = abs(np.sin(target_heading)) < 0.25
                starts_southbound = np.sin(float(agent.initial_state[3])) < -0.5
                starts_northbound = np.sin(float(agent.initial_state[3])) > 0.5

                if turns_to_east_west and starts_southbound and xi[1] > INTERSECTION_Y_MAX:
                    # Stay in the southbound lane until the north edge of the
                    # intersection, so the warm start does not cut through the
                    # approach divider.
                    effective_dest = np.array([xi[0], INTERSECTION_Y_MAX - 1.0])
                elif turns_to_east_west and starts_northbound and xi[1] < INTERSECTION_Y_MIN:
                    # Stay in the northbound lane until the south edge of the
                    # intersection before turning.
                    effective_dest = np.array([xi[0], INTERSECTION_Y_MIN + 1.0])
                elif turns_to_east_west and remaining > 8.0:
                    # Smooth left/right turn driven by signed heading error.
                    # Positive kappa turns counter-clockwise; negative kappa
                    # handles the new right-turn vehicles.
                    heading_error = wrap_angle(target_heading - heading)
                    if abs(heading_error) < 0.15:
                        kappa = 0.0
                    else:
                        kappa = float(np.clip(0.6 * heading_error, -0.40, 0.40))
                    acceleration = np.clip(2.0 * (desired_speed - xi[2]), -max_warm_start_brake, 1.5)
                    ui = np.array([kappa, acceleration], dtype=float)
                    us[i].append(ui)
                    u_k.append(ui)
                    continue
                elif xi[0] > NS_DIVIDER_X and xi[1] < INTERSECTION_Y_MIN + 0.5:
                    # Through northbound vehicles still get a straight waypoint
                    # to keep their warm start in-lane before the intersection.
                    effective_dest = np.array([xi[0], INTERSECTION_Y_MIN + 1.0])

                eff_delta = effective_dest - xi[:2]
                eff_dist = float(np.linalg.norm(eff_delta))
                if eff_dist > 0.1:
                    delta = eff_delta
                    distance = eff_dist
                fwd_left = np.array([-np.sin(heading), np.cos(heading)])
                lateral = float(np.dot(delta, fwd_left))
                if abs(lateral) > 0.01:
                    kappa = float(np.clip(2.0 * lateral / distance ** 2, -0.5, 0.5))
                else:
                    kappa = 0.0
                acceleration = np.clip(2.0 * (desired_speed - xi[2]), -max_warm_start_brake, 1.5)
            ui = np.array([kappa, acceleration], dtype=float)
            us[i].append(ui)
            u_k.append(ui)

        if k < game.horizon_steps - 1:
            xs.append(integrate(game.dynamics.evaluate, t, game.dt, xk, u_k, use_euler=True))
            t += game.dt

    costs = [
        [game.player_costs[i].evaluate(xs[k], [us[j][k] for j in range(game.num_players)], k)
         for k in range(game.horizon_steps)]
        for i in range(game.num_players)
    ]
    return xs, us, costs


# -----------------------------------------------------------------------------
# Reporting and plotting
# -----------------------------------------------------------------------------
def first_arrival_steps(xs: list[np.ndarray], agent_specs: list[AgentSpec]) -> list[int]:
    """First index at which each agent passes through PASS_THROUGH_RADIUS of its destination."""
    arrivals = []
    for i, agent in enumerate(agent_specs):
        destination = np.asarray(agent.destination, dtype=float)
        distances = [float(np.linalg.norm(np.asarray(x[agent_slice(i)][:2]) - destination)) for x in xs]
        arrivals.append(next((k for k, d in enumerate(distances) if d < PASS_THROUGH_RADIUS), len(xs) - 1))
    return arrivals


def game_end_step(xs: list[np.ndarray], agent_specs: list[AgentSpec]) -> int:
    """Flexible-T termination index.

    Returns max(per-agent pass-through steps) + STOP_AFTER_ARRIVAL_STEPS,
    clamped to the last available step.
    """
    arrivals = first_arrival_steps(xs, agent_specs)
    end = max(arrivals) + STOP_AFTER_ARRIVAL_STEPS
    return min(end, len(xs) - 1)


def agent_xy_until_arrival(
    xs: list[np.ndarray],
    agent_specs: list[AgentSpec],
    agent_index: int,
) -> np.ndarray:
    """Return one agent's x-y path clipped at its first pass-through step."""
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)

    arrival = first_arrival_steps(xs, agent_specs)[agent_index]
    stop = min(arrival + 1, len(xs))
    return np.asarray([x[agent_slice(agent_index)][:2] for x in xs[:stop]], dtype=float)


def freeze_arrived_agents(
    xs: list[np.ndarray],
    us: list[list[np.ndarray]],
    agent_specs: list[AgentSpec],
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Independently freeze each agent the moment it passes through PASS_THROUGH_RADIUS.

    For every step k > arrivals[i], agent i's position is held at the arrival
    position, its speed is zeroed, and its controls are zeroed.  Other agents
    continue on their optimized trajectories unmodified.
    """
    arrivals = first_arrival_steps(xs, agent_specs)
    xs = [x.copy() for x in xs]
    us_new = [[u.copy() for u in player_us] for player_us in us]

    for i in range(len(agent_specs)):
        arrival = arrivals[i]
        if arrival >= len(xs):
            continue
        freeze = xs[arrival][agent_slice(i)].copy()
        freeze[2] = 0.0  # zero speed at frozen state
        for k in range(arrival + 1, len(xs)):
            xs[k] = xs[k].copy()
            xs[k][agent_slice(i)] = freeze
        for k in range(arrival, len(us_new[i])):
            us_new[i][k] = np.zeros(CONTROL_DIM_PER_AGENT, dtype=float)

    return xs, us_new


def save_multi_agent_csv(xs, us, agent_specs: list[AgentSpec], path=f"{OUTPUT_PREFIX}.csv") -> Path:
    path = Path(path)
    arrivals = first_arrival_steps(xs, agent_specs)

    header = ["step", "time_s"]
    for i, agent in enumerate(agent_specs):
        header += [
            f"{agent.name}_active",
            f"{agent.name}_x_m",
            f"{agent.name}_y_m",
            f"{agent.name}_speed_mps",
            f"{agent.name}_heading_rad",
            f"{agent.name}_kappa_radpm",
            f"{agent.name}_a_mps2",
            f"{agent.name}_distance_to_destination_m",
        ]
    for i in range(len(agent_specs)):
        for j in range(i + 1, len(agent_specs)):
            header.append(f"distance_{agent_specs[i].name}_to_{agent_specs[j].name}_m")

    end_step = game_end_step(xs, agent_specs)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(end_step + 1):
            row = [k, k * DT]
            xk = xs[k]
            for i, agent in enumerate(agent_specs):
                xi = xk[agent_slice(i)]
                ui = us[i][k] if k < len(us[i]) else np.array([np.nan, np.nan])
                dist = float(np.linalg.norm(xi[:2] - np.asarray(agent.destination, dtype=float)))
                row += [
                    bool(k <= arrivals[i]),
                    xi[0], xi[1], xi[2], xi[3],
                    ui[0], ui[1],
                    dist,
                ]
            for i in range(len(agent_specs)):
                for j in range(i + 1, len(agent_specs)):
                    row.append(float(np.linalg.norm(agent_position(xk, i) - agent_position(xk, j))))
            writer.writerow(row)
    return path


def draw_road_scene(ax) -> None:
    """Draw road surface and lane markings as narrow rectangles (no scatter dots)."""
    import matplotlib.patches as mpatches

    road_color = '#c8c8c8'
    curb_color = '#383838'
    div_color  = '#f0c020'
    cw = 0.5  # curb strip width [m]

    # Road surfaces.
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MIN, ROAD_SOUTH_Y_MIN),
        INTERSECTION_X_MAX - INTERSECTION_X_MIN,
        ROAD_NORTH_Y_MAX - ROAD_SOUTH_Y_MIN,
        fc=road_color, ec='none', zorder=0,
    ))
    ax.add_patch(mpatches.Rectangle(
        (WEST_ARM_X_MIN, INTERSECTION_Y_MIN),
        INTERSECTION_X_MIN - WEST_ARM_X_MIN,
        INTERSECTION_Y_MAX - INTERSECTION_Y_MIN,
        fc=road_color, ec='none', zorder=0,
    ))
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MAX, INTERSECTION_Y_MIN),
        EAST_ARM_X_MAX - INTERSECTION_X_MAX,
        INTERSECTION_Y_MAX - INTERSECTION_Y_MIN,
        fc=road_color, ec='none', zorder=0,
    ))

    # Curb walls — narrow dark strips along each road boundary
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MIN - cw / 2, ROAD_SOUTH_Y_MIN),
        cw,
        INTERSECTION_Y_MIN - ROAD_SOUTH_Y_MIN + cw / 2,
        fc=curb_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MIN - cw / 2, INTERSECTION_Y_MAX),
        cw,
        ROAD_NORTH_Y_MAX - INTERSECTION_Y_MAX,
        fc=curb_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MAX - cw / 2, ROAD_SOUTH_Y_MIN),
        cw,
        INTERSECTION_Y_MIN - ROAD_SOUTH_Y_MIN + cw / 2,
        fc=curb_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MAX - cw / 2, INTERSECTION_Y_MAX),
        cw,
        ROAD_NORTH_Y_MAX - INTERSECTION_Y_MAX,
        fc=curb_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (WEST_ARM_X_MIN, INTERSECTION_Y_MIN - cw / 2),
        INTERSECTION_X_MIN - WEST_ARM_X_MIN,
        cw,
        fc=curb_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (WEST_ARM_X_MIN, INTERSECTION_Y_MAX - cw / 2),
        INTERSECTION_X_MIN - WEST_ARM_X_MIN,
        cw,
        fc=curb_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MAX, INTERSECTION_Y_MIN - cw / 2),
        EAST_ARM_X_MAX - INTERSECTION_X_MAX,
        cw,
        fc=curb_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (INTERSECTION_X_MAX, INTERSECTION_Y_MAX - cw / 2),
        EAST_ARM_X_MAX - INTERSECTION_X_MAX,
        cw,
        fc=curb_color, ec='none', zorder=1,
    ))

    # Lane divider on both N-S approaches, leaving the intersection box open.
    ax.add_patch(mpatches.Rectangle(
        (NS_DIVIDER_X - cw / 4, ROAD_SOUTH_Y_MIN),
        cw / 2,
        INTERSECTION_Y_MIN - ROAD_SOUTH_Y_MIN,
        fc=div_color, ec='none', zorder=1,
    ))
    ax.add_patch(mpatches.Rectangle(
        (NS_DIVIDER_X - cw / 4, INTERSECTION_Y_MAX),
        cw / 2,
        ROAD_NORTH_Y_MAX - INTERSECTION_Y_MAX,
        fc=div_color, ec='none', zorder=1,
    ))


def apply_publication_style(plt) -> None:
    """Apply consistent publishable Matplotlib styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": PLOT_FONT_SERIF,
        "mathtext.fontset": "stix",
        "figure.dpi": 140,
        "savefig.dpi": PLOT_DPI,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "lines.linewidth": PLOT_LINE_WIDTH,
        "patch.linewidth": 0.9,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.28,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def format_publication_axes(ax, *, equal: bool = False) -> None:
    """Readable axes, light grid, and clean spines."""
    ax.grid(True, color="#b6b6b6", linewidth=0.55, alpha=0.35, zorder=2)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#222222")
    ax.tick_params(direction="out", length=4.0, width=0.8, colors="#222222")


def agent_label(agent: AgentSpec) -> str:
    return agent.name.replace("_", " ").title()


def maybe_save_multi_agent_plot(xs, agent_specs: list[AgentSpec], path=f"{OUTPUT_PREFIX}.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    apply_publication_style(plt)

    end_step = game_end_step(xs, agent_specs)
    arrivals = first_arrival_steps(xs, agent_specs)
    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    draw_road_scene(ax)
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_specs)))
    trajectory_handles = []
    for i, agent in enumerate(agent_specs):
        stop = min(arrivals[i] + 1, end_step + 1)
        xy = np.asarray([x[agent_slice(i)][:2] for x in xs[:stop]])
        color = colors[i]
        line, = ax.plot(
            xy[:, 0], xy[:, 1],
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=0.0,
            alpha=0.92,
            label=agent_label(agent),
            zorder=4,
        )
        trajectory_handles.append(line)
        ax.scatter(
            [agent.initial_state[0]], [agent.initial_state[1]],
            marker="o", s=42, color=color, edgecolors="black",
            linewidths=0.55, zorder=6,
        )
        ax.scatter(
            [agent.destination[0]], [agent.destination[1]],
            marker="X", s=54, color=color, edgecolors="black",
            linewidths=0.55, zorder=6,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Multi-agent ILQR trajectories")
    format_publication_axes(ax, equal=True)
    ax.legend(
        handles=trajectory_handles,
        title="Vehicle",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.94,
        borderpad=0.8,
        labelspacing=0.55,
    )
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(path)



def maybe_save_position_time_plots(
    xs,
    agent_specs: list[AgentSpec],
    prefix=OUTPUT_PREFIX,
):
    """Save x(t) and y(t) plots for every agent.

    Output files:
    - multi_agent_trajectory_x_vs_t.png
    - multi_agent_trajectory_y_vs_t.png
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None, None
    apply_publication_style(plt)

    end_step = game_end_step(xs, agent_specs)
    arrivals = first_arrival_steps(xs, agent_specs)
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_specs)))

    # x(t) plot
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for i, agent in enumerate(agent_specs):
        stop = min(arrivals[i] + 1, end_step + 1)
        time_values = np.arange(stop) * DT
        x_values = np.asarray([xk[agent_slice(i)][0] for xk in xs[:stop]], dtype=float)
        ax.plot(
            time_values, x_values,
            color=colors[i],
            linewidth=1.9,
            marker="o",
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=0.0,
            alpha=0.92,
            label=agent_label(agent),
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("x position [m]")
    ax.set_title("x Position vs. Time")
    format_publication_axes(ax)
    ax.legend(
        title="Vehicle",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.94,
    )
    x_path = Path(f"{prefix}_x_vs_t.png")
    fig.savefig(x_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # y(t) plot
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for i, agent in enumerate(agent_specs):
        stop = min(arrivals[i] + 1, end_step + 1)
        time_values = np.arange(stop) * DT
        y_values = np.asarray([xk[agent_slice(i)][1] for xk in xs[:stop]], dtype=float)
        ax.plot(
            time_values, y_values,
            color=colors[i],
            linewidth=1.9,
            marker="o",
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=0.0,
            alpha=0.92,
            label=agent_label(agent),
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("y position [m]")
    ax.set_title("y Position vs. Time")
    format_publication_axes(ax)
    ax.legend(
        title="Vehicle",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.94,
    )
    y_path = Path(f"{prefix}_y_vs_t.png")
    fig.savefig(y_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return x_path, y_path


def maybe_save_xy_animation(
    xs,
    us,
    agent_specs: list[AgentSpec],
    path=f"{OUTPUT_PREFIX}_animation.gif",
    fps: int = 10,
):
    """Save an animated x-y trajectory GIF.

    The animation shows each agent moving in the x-y plane. Once an agent passes
    within PASS_THROUGH_RADIUS of its destination, its marker remains at that
    arrival point.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Polygon as MplPolygon
    except Exception as exc:
        print(f"Animation skipped because matplotlib/pillow is unavailable: {exc}")
        return None
    apply_publication_style(plt)

    arrivals = first_arrival_steps(xs, agent_specs)
    final_frame = game_end_step(xs, agent_specs)
    if final_frame < 1:
        final_frame = min(len(xs) - 1, HORIZON_STEPS - 1)

    # Pre-compute positions for clean indexing during animation.
    positions = []
    for i in range(len(agent_specs)):
        positions.append(
            np.asarray([xk[agent_slice(i)][:2] for xk in xs[: final_frame + 1]], dtype=float)
        )

    # Axis limits: include trajectories, starts, destinations, and road extent.
    points_for_limits = []
    for i, agent in enumerate(agent_specs):
        points_for_limits.append(positions[i])
        points_for_limits.append(np.asarray(agent.initial_state[:2], dtype=float).reshape(1, 2))
        points_for_limits.append(np.asarray(agent.destination, dtype=float).reshape(1, 2))
    # Include road corners so the lane rectangle is always in frame.
    points_for_limits.append(np.array([
        [WEST_ARM_X_MIN, INTERSECTION_Y_MIN],
        [EAST_ARM_X_MAX, INTERSECTION_Y_MAX],
        [INTERSECTION_X_MIN, ROAD_SOUTH_Y_MIN],
        [INTERSECTION_X_MAX, ROAD_NORTH_Y_MAX],
    ], dtype=float))

    all_points = np.vstack(points_for_limits)
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    margin = 0.15 * span

    fig, ax = plt.subplots(figsize=(7.4, 7.2), constrained_layout=True)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Multi-agent ILQR trajectory animation")
    format_publication_axes(ax, equal=True)

    draw_road_scene(ax)

    for agent in agent_specs:
        ax.scatter([agent.initial_state[0]], [agent.initial_state[1]], marker="o", zorder=5, label=f"{agent.name} start")
        ax.scatter([agent.destination[0]], [agent.destination[1]], marker="x", zorder=5, label=f"{agent.name} destination")

    trajectory_lines = []
    vehicle_patches = []
    speed_texts = []
    for agent in agent_specs:
        line, = ax.plot([], [], linewidth=1.7, label=f"{agent.name} path", zorder=4)
        trajectory_lines.append(line)
        # Vehicle rectangle — colour matched to trajectory line
        color = line.get_color()
        patch = MplPolygon(np.zeros((4, 2)), closed=True,
                           fc=color, ec='black', alpha=0.85, linewidth=1.2, zorder=5)
        ax.add_patch(patch)
        vehicle_patches.append(patch)
        # Speed label shown above the vehicle
        st = ax.text(0, 0, "", fontsize=8, color="black",
                     ha="center", va="bottom", fontweight="bold", zorder=6)
        speed_texts.append(st)

    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    # ax.legend(fontsize=7, loc="best")

    def init():
        for line, patch, st in zip(trajectory_lines, vehicle_patches, speed_texts):
            line.set_data([], [])
            patch.set_xy(np.zeros((4, 2)))
            st.set_text("")
        time_text.set_text("")
        return [*trajectory_lines, *vehicle_patches, *speed_texts, time_text]

    def update(frame: int):
        xk = xs[frame]
        for i in range(len(agent_specs)):
            draw_until = min(frame, arrivals[i], len(positions[i]) - 1)
            xy = positions[i][: draw_until + 1]
            trajectory_lines[i].set_data(xy[:, 0], xy[:, 1])

            # Vehicle rectangle at current (or frozen) position
            xi = xk[agent_slice(i)]
            pos     = np.array(xi[:2], dtype=float)
            heading = float(xi[3])
            speed   = float(xi[2])
            vehicle_patches[i].set_xy(vehicle_corners(pos, heading))

            # Speed label centred 3 m above vehicle
            speed_texts[i].set_position((pos[0], pos[1] + 3.0))
            speed_texts[i].set_text(f"{agent_specs[i].name.split('_')[0]}: {speed:.1f} m/s")

        time_text.set_text(f"t = {frame * DT:.1f} s")
        return [*trajectory_lines, *vehicle_patches, *speed_texts, time_text]

    animation = FuncAnimation(
        fig,
        update,
        frames=final_frame + 1,
        init_func=init,
        interval=1000.0 / fps,
        blit=True,
    )

    output_path = Path(path)
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps))
    except Exception as exc:
        plt.close(fig)
        print(f"Animation skipped because GIF export failed: {exc}")
        return None
    plt.close(fig)
    return output_path


def summarize_solution(result, agent_specs: list[AgentSpec]) -> None:
    xs, us = result["xs"], result["us"]
    arrivals = first_arrival_steps(xs, agent_specs)
    end_step = game_end_step(xs, agent_specs)

    print(f"Solved: {result.get('name', 'extendable_multi_agent_vehicle_game')}")
    print(f"Number of agents: {len(agent_specs)}")
    print(f"Time step: {DT:.2f} s")
    print(f"Optimization horizon: {HORIZON_STEPS} steps = {HORIZON_STEPS * DT:.1f} s")
    print(f"Game end step (flexible-T): {end_step}  ({end_step * DT:.2f} s)  "
          f"[last arrival + {STOP_AFTER_ARRIVAL_STEPS} steps]")
    print(f"Static obstacles: {[np.asarray(o).tolist() for o in STATIC_OBSTACLES]}")
    print("State order per agent: [px, py, speed, heading]")
    print("Control order per agent: [kappa, a]")
    print("")

    for i, agent in enumerate(agent_specs):
        arrival = arrivals[i]
        xf_arr = xs[arrival][agent_slice(i)]
        dest = np.asarray(agent.destination, dtype=float)
        all_dists = [float(np.linalg.norm(xs[k][agent_slice(i)][:2] - dest)) for k in range(end_step + 1)]
        min_dist = min(all_dists)
        min_step = int(np.argmin(all_dists))
        xf_end = xs[end_step][agent_slice(i)]
        dist_end = float(np.linalg.norm(xf_end[:2] - dest))
        dist_arr = float(np.linalg.norm(xf_arr[:2] - dest))
        print(f"{agent.name}:")
        print(f"  destination: {dest}")
        print(f"  passed through ({PASS_THROUGH_RADIUS} m): step {arrival}  ({arrival * DT:.2f} s)  dist={dist_arr:.3f} m")
        print(f"  closest approach (up to game end): step {min_step}  ({min_step * DT:.2f} s)  dist={min_dist:.3f} m")
        print(f"  state at game end (step {end_step}): {xf_end}  dist={dist_end:.3f} m")

    if len(agent_specs) >= 2:
        print("")
        print("Pairwise minimum distances over game horizon:")
        for i in range(len(agent_specs)):
            for j in range(i + 1, len(agent_specs)):
                dists = [float(np.linalg.norm(agent_position(xs[k], i) - agent_position(xs[k], j)))
                         for k in range(end_step + 1)]
                print(f"  {agent_specs[i].name} <-> {agent_specs[j].name}: {min(dists):.3f} m")

    print("")
    print(f"ILQR cost history: {[[round(v, 3) for v in h] for h in result['history']]}")


def maybe_save_trajectory_evolution_per_agent(
    iteration_trajectories,
    initial_xs,
    agent_specs: list[AgentSpec],
    prefix=OUTPUT_PREFIX,
):
    """Subplot grid (one panel per agent) showing trajectory convergence.

    Up to 20 evenly-spaced ILQR iterations are drawn using the 'turbo' colourmap
    (deep-blue = iteration 1 → dark-red = final).  Line weight and opacity grow
    with iteration number so later trajectories stand out more.  The initial
    warm-start guess is drawn last as a thick bright-magenta dashed line so it
    is always clearly visible on top of everything else.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
    except Exception:
        return None
    apply_publication_style(plt)

    num_iters = len(iteration_trajectories)
    if num_iters == 0 or initial_xs is None:
        return None

    # Pick up to 20 evenly-spaced indices; always include first and last.
    max_shown = 20
    if num_iters <= max_shown:
        iter_indices = list(range(num_iters))
    else:
        iter_indices = sorted({
            int(round(k * (num_iters - 1) / (max_shown - 1)))
            for k in range(max_shown)
        })

    n_shown = len(iter_indices)

    # turbo: vivid blue → cyan → green → yellow → orange → red
    # Normalise over the *full* iteration range so colours are well spread.
    cmap = plt.colormaps["turbo"]
    norm = Normalize(vmin=0, vmax=max(num_iters - 1, 1))

    n_agents = len(agent_specs)
    ncols = 2 if n_agents <= 6 else 3
    nrows = int(np.ceil(n_agents / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9 * ncols, 5.1 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (agent, ax) in enumerate(zip(agent_specs, axes_flat)):
        draw_road_scene(ax)

        init_xy = agent_xy_until_arrival(initial_xs, agent_specs, i)
        iter_xys = [
            agent_xy_until_arrival(iteration_trajectories[idx], agent_specs, i)
            for idx in iter_indices
        ]

        # Axis limits: encompass initial guess + all shown iterations.
        all_pts = np.vstack([init_xy] + iter_xys)
        pad = 3.0
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.set_aspect("equal", adjustable="box")
        x_span = all_pts[:, 0].max() - all_pts[:, 0].min() + 2.0 * pad
        y_span = all_pts[:, 1].max() - all_pts[:, 1].min() + 2.0 * pad
        if x_span < 8.0 and y_span > 25.0:
            x_center = 0.5 * sum(ax.get_xlim())
            ax.set_xticks([round(x_center, 1)])

        # --- ILQR iterations (drawn first so initial guess sits on top) ---
        for rank, idx in enumerate(iter_indices):
            progress = rank / max(n_shown - 1, 1)          # 0 (first) → 1 (last)
            color = cmap(norm(idx))
            lw    = 0.75 + 1.45 * progress                 # 0.75 -> 2.2
            alpha = 0.50 + 0.42 * progress                 # 0.50 -> 0.92
            ax.plot(
                iter_xys[rank][:, 0], iter_xys[rank][:, 1],
                color=color, linewidth=lw, alpha=alpha, zorder=3,
            )

        # --- Initial guess — bright magenta, drawn last so it is always visible ---
        ax.plot(
            init_xy[:, 0], init_xy[:, 1],
            linestyle="--", color="#D000B8", linewidth=2.7, alpha=0.95,
            label="Initial guess", zorder=5,
        )

        # Start and destination markers (above everything).
        ax.scatter(
            [agent.initial_state[0]], [agent.initial_state[1]],
            marker="o", s=64, color="limegreen", edgecolors="black",
            linewidths=0.8, zorder=6, label="Start",
        )
        ax.scatter(
            [agent.destination[0]], [agent.destination[1]],
            marker="*", s=112, color="gold", edgecolors="black",
            linewidths=0.8, zorder=6, label="Destination",
        )

        ax.set_title(agent_label(agent), fontsize=12, fontweight="bold")
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        format_publication_axes(ax, equal=True)

    for ax in axes_flat[n_agents:]:
        ax.set_visible(False)

    legend_handles = [
        Line2D(
            [0], [0], linestyle="--", color="#D000B8", linewidth=2.7,
            label="Initial guess",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None", markersize=7,
            markerfacecolor="limegreen", markeredgecolor="black",
            label="Start",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="None", markersize=10,
            markerfacecolor="gold", markeredgecolor="black",
            label="Destination",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.025),
        frameon=True,
        framealpha=0.94,
        borderpad=0.55,
        handlelength=2.4,
    )

    # Shared colourbar with ticks at every shown iteration.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("ILQR Iteration", fontsize=11)
    # Place ticks only at shown iterations; limit label count to ≤ 10 for readability.
    tick_step = max(1, n_shown // 10)
    tick_iters = iter_indices[::tick_step]
    if iter_indices[-1] not in tick_iters:
        tick_iters = tick_iters + [iter_indices[-1]]
    cbar.set_ticks([idx for idx in tick_iters])
    cbar.set_ticklabels([str(idx + 1) for idx in tick_iters])

    fig.suptitle(
        f"Trajectory Evolution per Agent — {LQ_SOLVER_TYPE}\n"
        f"{num_iters} total iterations; {n_shown} shown; "
        f"blue=early, red=late; magenta dashed=initial guess",
        fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(top=0.91, bottom=0.09, right=0.88, wspace=0.32, hspace=0.38)
    out_path = f"{prefix}_trajectory_per_agent_per_iteration.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def maybe_save_iteration_plots(
    history,
    iteration_trajectories,
    iteration_delta_xs,
    agent_specs: list[AgentSpec],
    prefix=OUTPUT_PREFIX,
):
    """Save per-iteration plots for costs, trajectories, and delta_x."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Iteration plots skipped because matplotlib is unavailable: {exc}")
        return None
    apply_publication_style(plt)

    saved_paths = [
        Path(f"{prefix}_costs_per_iteration.png"),
        Path(f"{prefix}_trajectory_evolution.png"),
        Path(f"{prefix}_delta_x_per_iteration.png"),
    ]

    # 1. Cost per agent per iteration
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    history_arr = np.array(history)  # (iterations, agents)
    x_iters = np.arange(1, len(history) + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_specs)))
    for i, agent in enumerate(agent_specs):
        ax.plot(
            x_iters,
            history_arr[:, i],
            color=colors[i],
            linewidth=1.75,
            alpha=0.88,
            label=agent_label(agent),
        )
        ax.scatter(
            x_iters,
            history_arr[:, i],
            s=CONVERGENCE_MARKER_SIZE,
            color=colors[i],
            alpha=CONVERGENCE_MARKER_ALPHA,
            edgecolors="none",
            zorder=4,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total cost")
    ax.set_title(f"Cost Convergence per Agent ({LQ_SOLVER_TYPE})")
    format_publication_axes(ax)
    ax.legend(
        title="Vehicle",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.94,
    )
    fig.savefig(saved_paths[0], dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2. Trajectories per iteration (subset to avoid clutter)
    fig, ax = plt.subplots(figsize=(8.4, 6.8), constrained_layout=True)
    draw_road_scene(ax)
    
    num_iters = len(iteration_trajectories)
    # Plot first, middle, and last iterations
    indices_to_plot = sorted(list(set([0, num_iters // 2, num_iters - 1])))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))
    
    for idx, color in zip(indices_to_plot, colors):
        xs_iter = iteration_trajectories[idx]
        for i, agent in enumerate(agent_specs):
            xy = agent_xy_until_arrival(xs_iter, agent_specs, i)
            label = f"Iteration {idx + 1}" if i == 0 else None
            ax.plot(
                xy[:, 0], xy[:, 1],
                color=color,
                linewidth=1.9,
                alpha=0.50,
                label=label,
                zorder=3,
            )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Trajectory Evolution ({LQ_SOLVER_TYPE})")
    format_publication_axes(ax, equal=True)
    ax.legend(
        title="Shown iteration",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.94,
    )
    fig.savefig(saved_paths[1], dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # 3. Max delta x magnitude per iteration
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    max_dxs = [max(dxs) for dxs in iteration_delta_xs]
    x_delta = np.arange(1, len(max_dxs) + 1)
    ax.plot(
        x_delta,
        max_dxs,
        color="#B00020",
        linewidth=1.9,
        alpha=0.88,
        label=r"max $\|\Delta x\|$",
    )
    ax.scatter(
        x_delta,
        max_dxs,
        s=CONVERGENCE_MARKER_SIZE,
        color="#B00020",
        alpha=CONVERGENCE_MARKER_ALPHA,
        edgecolors="none",
        zorder=4,
    )
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Max $\|\Delta x\|$")
    ax.set_title(f"State-Update Convergence ({LQ_SOLVER_TYPE})")
    ax.set_yscale("log")
    format_publication_axes(ax)
    ax.grid(True, which="both", color="#b6b6b6", linewidth=0.55, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    fig.savefig(saved_paths[2], dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return saved_paths


if __name__ == "__main__":
    t_total_start = time.perf_counter()
    game = build_multi_agent_game(AGENTS)
    solver = ILQSolver(
        game,
        max_iterations=MAX_ITERATIONS,
        convergence_tol=CONVERGENCE_TOL,
        relative_cost_convergence_tol=RELATIVE_COST_CONVERGENCE_TOL,
        convergence_patience=CONVERGENCE_PATIENCE,
        alpha_scaling=ALPHA_SCALING,
        use_euler=True,
        lq_solver_type=LQ_SOLVER_TYPE,
        alpha_line_search=USE_ALPHA_LINE_SEARCH,
        alpha_line_search_min=ALPHA_LINE_SEARCH_MIN,
        alpha_line_search_shrink=ALPHA_LINE_SEARCH_SHRINK,
        alpha_line_search_max_growth=ALPHA_LINE_SEARCH_MAX_GROWTH,
        alpha_line_search_start_iteration=ALPHA_LINE_SEARCH_START_ITERATION,
    )
    _init_nominal = make_initial_nominal_trajectory(game, AGENTS)
    solver.current_operating_point = _init_nominal
    initial_xs = _init_nominal[0]
    result = solver.solve()
    result["name"] = game.name

    # Save per-iteration convergence data and plots
    iteration_plot_paths = maybe_save_iteration_plots(
        result["history"],
        result["iteration_trajectories"],
        result["iteration_delta_xs"],
        AGENTS,
        prefix=OUTPUT_PREFIX,
    )

    # Per-agent trajectory evolution plot (one subplot per agent, initial guess shown)
    traj_evol_path = maybe_save_trajectory_evolution_per_agent(
        result["iteration_trajectories"],
        initial_xs,
        AGENTS,
        prefix=OUTPUT_PREFIX,
    )
    if traj_evol_path is not None:
        print(f"Saved per-agent trajectory evolution: {traj_evol_path}")

    print(f"Total solver wall-clock time: {time.perf_counter() - t_total_start:.1f}s")

    # Freeze each agent independently at its pass-through step so it does not
    # drift, reverse, or oscillate while waiting for slower agents to finish.
    result["xs"], result["us"] = freeze_arrived_agents(result["xs"], result["us"], AGENTS)

    csv_path = save_multi_agent_csv(result["xs"], result["us"], AGENTS, path=f"{OUTPUT_PREFIX}.csv")
    plot_path = maybe_save_multi_agent_plot(result["xs"], AGENTS, path=f"{OUTPUT_PREFIX}.png")
    x_time_path, y_time_path = maybe_save_position_time_plots(result["xs"], AGENTS, prefix=OUTPUT_PREFIX)
    animation_path = maybe_save_xy_animation(
        result["xs"],
        result["us"],
        AGENTS,
        path=f"{OUTPUT_PREFIX}_animation.gif",
        fps=10,
    )

    summarize_solution(result, AGENTS)
    print(f"Saved CSV: {csv_path.resolve()}")
    if plot_path is not None:
        print(f"Saved trajectory plot: {plot_path.resolve()}")
    if animation_path is not None:
        print(f"Saved trajectory animation: {animation_path.resolve()}")
    if iteration_plot_paths is not None:
        for iteration_plot_path in iteration_plot_paths:
            print(f"Saved iteration plot: {iteration_plot_path.resolve()}")
