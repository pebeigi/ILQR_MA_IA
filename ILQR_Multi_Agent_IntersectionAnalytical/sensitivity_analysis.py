"""Sensitivity-analysis driver for the multi-agent ILQ intersection game.

This file intentionally does not replace ``main.py``.  It imports the baseline
vehicle model and runs small batches of robustness experiments.

Set SWITCH below or pass ``--switch``:
    1 -> different game setup analysis
    2 -> different initial-guess analysis
    3 -> different parameter-value analysis

Each case saves:
    - final trajectory CSV and PNG
    - cost convergence PNG
    - trajectory convergence PNG with 20 shown iterations by default
    - delta-x convergence PNG
    - per-agent initial trajectory to optimized trajectory evolution PNG
    - trajectory animation GIF
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import main as experiment
from ilq.ilq_solver import ILQSolver


# -----------------------------------------------------------------------------
# User-facing sensitivity settings
# -----------------------------------------------------------------------------
SWITCH = 3

# Keep sweeps useful for convergence plots.  The default runner forces at least
# this many iterations unless --allow-early-convergence is supplied.
MIN_TRAJECTORY_ITERATIONS = 20
TRAJECTORY_ITERATIONS_TO_PLOT = 20
SENSITIVITY_MAX_ITERATIONS = 300
SENSITIVITY_CONVERGENCE_TOL = experiment.CONVERGENCE_TOL
SENSITIVITY_ALPHA_SCALING = experiment.ALPHA_SCALING
SENSITIVITY_USE_ALPHA_LINE_SEARCH = experiment.USE_ALPHA_LINE_SEARCH
SENSITIVITY_ALPHA_LINE_SEARCH_MIN = experiment.ALPHA_LINE_SEARCH_MIN
SENSITIVITY_ALPHA_LINE_SEARCH_SHRINK = experiment.ALPHA_LINE_SEARCH_SHRINK
SENSITIVITY_ALPHA_LINE_SEARCH_MAX_GROWTH = experiment.ALPHA_LINE_SEARCH_MAX_GROWTH
SENSITIVITY_ALPHA_LINE_SEARCH_START_ITERATION = experiment.ALPHA_LINE_SEARCH_START_ITERATION
SENSITIVITY_LQ_SOLVER_TYPE = experiment.LQ_SOLVER_TYPE
SAVE_PLOTS = True
SAVE_GIFS = True
CASE_LIMIT: int | None = None

OUTPUT_DIR = Path(__file__).parent / "outputs" / "sensitivity_analysis"


@dataclass
class SensitivityCase:
    name: str
    agents: list[experiment.AgentSpec]
    initial_guess: str = "nominal"
    description: str = ""


def copy_agent(agent: experiment.AgentSpec, **updates) -> experiment.AgentSpec:
    """Copy an AgentSpec without sharing arrays or stale leader indices."""
    cost = updates.pop("cost", replace(agent.cost))
    initial_state = np.asarray(
        updates.pop("initial_state", agent.initial_state),
        dtype=float,
    ).copy()
    destination = np.asarray(
        updates.pop("destination", agent.destination),
        dtype=float,
    ).copy()
    leader_index = updates.pop("leader_index", None)
    return replace(
        agent,
        initial_state=initial_state,
        destination=destination,
        cost=cost,
        leader_index=leader_index,
        **updates,
    )


def copy_agents(agents: list[experiment.AgentSpec]) -> list[experiment.AgentSpec]:
    return [copy_agent(agent) for agent in agents]


def clear_leader_cost(cost: experiment.CostWeights) -> experiment.CostWeights:
    return replace(
        cost,
        leader_repulsion=0.0,
        leader_proximity_speed_weight=0.0,
        leader_yield_line_weight=0.0,
    )


def select_agents(names: list[str]) -> list[experiment.AgentSpec]:
    """Select a subset and clear leader costs if the leader is not included."""
    by_name = {agent.name: agent for agent in experiment.AGENTS}
    selected_names = set(names)
    selected = []
    for name in names:
        agent = by_name[name]
        leader_name = agent.leader_name
        cost = replace(agent.cost)
        if leader_name not in selected_names:
            leader_name = None
            cost = clear_leader_cost(cost)
        selected.append(copy_agent(agent, leader_name=leader_name, cost=cost))
    return selected


def update_agent(
    agents: list[experiment.AgentSpec],
    name: str,
    *,
    initial_state: np.ndarray | None = None,
    destination: np.ndarray | None = None,
    cost: experiment.CostWeights | None = None,
    desired_speed: float | None = None,
    warm_start_delay_steps: int | None = None,
) -> list[experiment.AgentSpec]:
    updated = []
    for agent in agents:
        if agent.name != name:
            updated.append(copy_agent(agent))
            continue
        changes = {}
        if initial_state is not None:
            changes["initial_state"] = initial_state
        if destination is not None:
            changes["destination"] = destination
        if cost is not None:
            changes["cost"] = cost
        if desired_speed is not None:
            changes["desired_speed"] = desired_speed
        if warm_start_delay_steps is not None:
            changes["warm_start_delay_steps"] = warm_start_delay_steps
        updated.append(copy_agent(agent, **changes))
    return updated


def shift_initial_y(
    agents: list[experiment.AgentSpec],
    shifts_by_name: dict[str, float],
) -> list[experiment.AgentSpec]:
    shifted = []
    for agent in agents:
        state = np.asarray(agent.initial_state, dtype=float).copy()
        state[1] += shifts_by_name.get(agent.name, 0.0)
        shifted.append(copy_agent(agent, initial_state=state))
    return shifted


def is_turning_agent(agent: experiment.AgentSpec) -> bool:
    heading_change = experiment.wrap_angle(
        experiment.desired_terminal_heading(agent) - float(agent.initial_state[3])
    )
    return abs(heading_change) > 0.35


def with_turning_warm_start_delay(
    agents: list[experiment.AgentSpec],
    delay_steps: int,
) -> list[experiment.AgentSpec]:
    delayed = []
    for agent in agents:
        delay = delay_steps if is_turning_agent(agent) else 0
        delayed.append(copy_agent(agent, warm_start_delay_steps=delay))
    return delayed


def with_guess_desired_speed(
    agents: list[experiment.AgentSpec],
    desired_speed: float,
) -> list[experiment.AgentSpec]:
    return [copy_agent(agent, desired_speed=desired_speed) for agent in agents]


def scale_agent_costs(
    agents: list[experiment.AgentSpec],
    *,
    interaction_scale: float = 1.0,
    control_scale: float = 1.0,
    obstacle_scale: float = 1.0,
    speed_scale: float = 1.0,
) -> list[experiment.AgentSpec]:
    scaled = []
    for agent in agents:
        w = agent.cost
        cost = replace(
            w,
            q_speed=w.q_speed * speed_scale,
            beta_2=w.beta_2 * control_scale,
            beta_3=w.beta_3 * control_scale,
            static_obstacle_repulsion=w.static_obstacle_repulsion * obstacle_scale,
            leader_repulsion=w.leader_repulsion * interaction_scale,
            cross_traffic_repulsion=w.cross_traffic_repulsion * interaction_scale,
            proximity_speed_weight=w.proximity_speed_weight * interaction_scale,
            leader_proximity_speed_weight=w.leader_proximity_speed_weight * interaction_scale,
        )
        scaled.append(copy_agent(agent, cost=cost))
    return scaled


def build_game_setup_cases() -> list[SensitivityCase]:
    baseline = copy_agents(experiment.AGENTS)
    tight_timing = shift_initial_y(
        baseline,
        {
            "left_turn_northbound": 2.0,
            "follower_left_turn": 2.0,
            "right_turn_southbound": -4.0,
            "right_turn_northbound": 2.0,
        },
    )

    return [
        SensitivityCase(
            name="one_through_one_left",
            agents=select_agents(["through_southbound", "left_turn_northbound"]),
            description="Minimal gap-acceptance case: one through vehicle and one left-turn vehicle.",
        ),
        SensitivityCase(
            name="through_left_with_followers",
            agents=select_agents(
                [
                    "through_southbound",
                    "left_turn_northbound",
                    "follower_southbound",
                    "follower_left_turn",
                ]
            ),
            description="Four-vehicle setup with car-following in both streams.",
        ),
        SensitivityCase(
            name="left_and_right_merge_pair",
            agents=select_agents(["left_turn_northbound", "right_turn_southbound"]),
            description="Two turning vehicles that both enter the westbound lane.",
        ),
        SensitivityCase(
            name="baseline_six_vehicle",
            agents=baseline,
            description="Current six-vehicle working setup.",
        ),
        SensitivityCase(
            name="baseline_six_vehicle_tight_timing",
            agents=tight_timing,
            description="Six-vehicle setup with arrivals moved closer together.",
        ),
    ]


def build_initial_guess_cases() -> list[SensitivityCase]:
    agents = copy_agents(experiment.AGENTS)
    return [
        SensitivityCase(
            name="guess_nominal",
            agents=agents,
            initial_guess="nominal",
            description="Baseline destination-seeking warm start.",
        ),
        SensitivityCase(
            name="guess_zero_control",
            agents=copy_agents(experiment.AGENTS),
            initial_guess="zero_control",
            description="Initial guess from straight zero-control rollout.",
        ),
        SensitivityCase(
            name="guess_turns_delayed_10",
            agents=copy_agents(experiment.AGENTS),
            initial_guess="turns_delayed_10",
            description="Only the warm-start reference delays all turning vehicles by 10 steps.",
        ),
        SensitivityCase(
            name="guess_turns_delayed_20",
            agents=copy_agents(experiment.AGENTS),
            initial_guess="turns_delayed_20",
            description="Only the warm-start reference delays all turning vehicles by 20 steps.",
        ),
        SensitivityCase(
            name="guess_slow_speed",
            agents=copy_agents(experiment.AGENTS),
            initial_guess="slow_speed",
            description="Warm start generated with lower desired speed.",
        ),
        SensitivityCase(
            name="guess_perturbed_reference",
            agents=copy_agents(experiment.AGENTS),
            initial_guess="perturbed_reference",
            description="Nominal warm start with small deterministic position perturbations.",
        ),
    ]


def build_parameter_cases() -> list[SensitivityCase]:
    baseline = copy_agents(experiment.AGENTS)
    return [
        SensitivityCase(
            name="params_baseline",
            agents=baseline,
            description="Baseline cost parameters.",
        ),
        SensitivityCase(
            name="params_interaction_low_0p75",
            agents=scale_agent_costs(copy_agents(experiment.AGENTS), interaction_scale=0.75),
            description="Pairwise and proximity interaction weights scaled down by 25 percent.",
        ),
        SensitivityCase(
            name="params_interaction_high_1p25",
            agents=scale_agent_costs(copy_agents(experiment.AGENTS), interaction_scale=1.25),
            description="Pairwise and proximity interaction weights scaled up by 25 percent.",
        ),
        SensitivityCase(
            name="params_control_low_0p75",
            agents=scale_agent_costs(copy_agents(experiment.AGENTS), control_scale=0.75),
            description="Curvature and acceleration costs scaled down by 25 percent.",
        ),
        SensitivityCase(
            name="params_control_high_1p25",
            agents=scale_agent_costs(copy_agents(experiment.AGENTS), control_scale=1.25),
            description="Curvature and acceleration costs scaled up by 25 percent.",
        ),
        SensitivityCase(
            name="params_obstacles_high_1p25",
            agents=scale_agent_costs(copy_agents(experiment.AGENTS), obstacle_scale=1.25),
            description="Static-obstacle repulsion scaled up by 25 percent.",
        ),
        SensitivityCase(
            name="params_speed_tracking_high_1p25",
            agents=scale_agent_costs(copy_agents(experiment.AGENTS), speed_scale=1.25),
            description="Speed-tracking cost scaled up by 25 percent.",
        ),
    ]


def build_cases(switch: int) -> list[SensitivityCase]:
    if switch == 1:
        return build_game_setup_cases()
    if switch == 2:
        return build_initial_guess_cases()
    if switch == 3:
        return build_parameter_cases()
    raise ValueError("switch must be 1, 2, or 3")


def costs_for_trajectory(game, xs, us):
    return [
        [
            game.player_costs[i].evaluate(
                xs[k],
                [us[j][k] for j in range(game.num_players)],
                k,
            )
            for k in range(game.horizon_steps)
        ]
        for i in range(game.num_players)
    ]


def make_zero_control_initial_guess(game):
    xs = [game.x0.copy()]
    us = [[] for _ in range(game.num_players)]
    t = 0.0
    for k in range(game.horizon_steps):
        u_k = [np.zeros(game.control_dims[i], dtype=float) for i in range(game.num_players)]
        for i, ui in enumerate(u_k):
            us[i].append(ui)
        if k < game.horizon_steps - 1:
            xs.append(
                experiment.integrate(
                    game.dynamics.evaluate,
                    t,
                    game.dt,
                    xs[-1],
                    u_k,
                    use_euler=True,
                )
            )
            t += game.dt
    return xs, us, costs_for_trajectory(game, xs, us)


def make_perturbed_reference_initial_guess(game, agents):
    xs, us, _ = experiment.make_initial_nominal_trajectory(game, agents)
    perturbed_xs = [np.asarray(x, dtype=float).copy() for x in xs]
    for k, xk in enumerate(perturbed_xs):
        if k == 0:
            continue
        phase = 0.12 * k
        for i in range(game.num_players):
            sl = experiment.agent_slice(i)
            xk[sl.start] += 0.35 * np.sin(phase + 0.7 * i)
            xk[sl.start + 1] += 0.25 * np.cos(phase + 0.5 * i)
    perturbed_xs[0] = game.x0.copy()
    return perturbed_xs, us, costs_for_trajectory(game, perturbed_xs, us)


def make_initial_guess(game, agents, variant: str):
    if variant == "nominal":
        return experiment.make_initial_nominal_trajectory(game, agents)
    if variant == "zero_control":
        return make_zero_control_initial_guess(game)
    if variant == "turns_delayed_10":
        guess_agents = with_turning_warm_start_delay(agents, 10)
        return experiment.make_initial_nominal_trajectory(game, guess_agents)
    if variant == "turns_delayed_20":
        guess_agents = with_turning_warm_start_delay(agents, 20)
        return experiment.make_initial_nominal_trajectory(game, guess_agents)
    if variant == "slow_speed":
        guess_agents = with_guess_desired_speed(agents, 4.0)
        return experiment.make_initial_nominal_trajectory(game, guess_agents)
    if variant == "perturbed_reference":
        return make_perturbed_reference_initial_guess(game, agents)
    raise ValueError(f"unknown initial guess variant {variant!r}")


def min_pairwise_distance(xs, agents) -> float:
    if len(agents) < 2:
        return float("nan")
    min_dist = float("inf")
    for xk in xs:
        for i in range(len(agents)):
            pi = experiment.agent_position(xk, i)
            for j in range(i + 1, len(agents)):
                dist = float(np.linalg.norm(pi - experiment.agent_position(xk, j)))
                min_dist = min(min_dist, dist)
    return min_dist


def min_static_obstacle_distance(xs, agents) -> float:
    obstacles = np.asarray(experiment.STATIC_OBSTACLES, dtype=float)
    if obstacles.size == 0:
        return float("nan")
    min_dist = float("inf")
    for xk in xs:
        for i in range(len(agents)):
            pos = experiment.agent_position(xk, i)
            min_dist = min(min_dist, float(np.min(np.linalg.norm(obstacles - pos, axis=1))))
    return min_dist


def trajectory_is_finite(xs, us) -> bool:
    return all(np.all(np.isfinite(x)) for x in xs) and all(
        np.all(np.isfinite(ui)) for player_us in us for ui in player_us
    )


def max_control_abs(us, control_index: int) -> float:
    values = [
        abs(float(ui[control_index]))
        for player_us in us
        for ui in player_us
        if len(ui) > control_index
    ]
    return max(values) if values else float("nan")


def max_speed_abs(xs, agents) -> float:
    values = []
    for xk in xs:
        for i in range(len(agents)):
            values.append(abs(float(xk[experiment.agent_slice(i).start + 2])))
    return max(values) if values else float("nan")


def summarize_result(case, result, agents, elapsed_sec, max_iterations) -> dict[str, object]:
    xs = result["xs"]
    us = result["us"]
    history = result["history"]
    total_costs = [float(sum(row)) for row in history]
    arrival_steps = experiment.first_arrival_steps(xs, agents)
    arrived_flags = []
    for i, agent in enumerate(agents):
        destination = np.asarray(agent.destination, dtype=float)
        dists = [
            float(np.linalg.norm(np.asarray(x[experiment.agent_slice(i)][:2]) - destination))
            for x in xs
        ]
        arrived_flags.append(min(dists) < experiment.PASS_THROUGH_RADIUS)

    final_delta_x = float("nan")
    if result["iteration_delta_xs"]:
        final_delta_x = float(max(result["iteration_delta_xs"][-1]))

    return {
        "case": case.name,
        "description": case.description,
        "status": "ok",
        "agents": len(agents),
        "agent_names": ";".join(agent.name for agent in agents),
        "initial_guess": case.initial_guess,
        "iterations": len(history),
        "hit_iteration_cap": len(history) >= max_iterations,
        "elapsed_sec": round(elapsed_sec, 3),
        "initial_total_cost": round(total_costs[0], 6) if total_costs else float("nan"),
        "final_total_cost": round(total_costs[-1], 6) if total_costs else float("nan"),
        "cost_change": round(total_costs[-1] - total_costs[0], 6) if len(total_costs) >= 2 else 0.0,
        "final_max_delta_x": round(final_delta_x, 6),
        "min_pairwise_distance": round(min_pairwise_distance(xs, agents), 6),
        "min_static_obstacle_distance": round(min_static_obstacle_distance(xs, agents), 6),
        "arrived_count": int(sum(arrived_flags)),
        "arrival_steps": ";".join(str(step) for step in arrival_steps),
        "finite": trajectory_is_finite(xs, us),
        "max_abs_speed": round(max_speed_abs(xs, agents), 6),
        "max_abs_kappa": round(max_control_abs(us, 0), 6),
        "max_abs_accel": round(max_control_abs(us, 1), 6),
    }


def trajectory_iteration_indices(num_iters: int) -> list[int]:
    """Return iteration indices for trajectory-convergence plots."""
    if num_iters <= 0:
        return []
    if num_iters <= TRAJECTORY_ITERATIONS_TO_PLOT:
        return list(range(num_iters))
    return sorted({
        int(round(k * (num_iters - 1) / (TRAJECTORY_ITERATIONS_TO_PLOT - 1)))
        for k in range(TRAJECTORY_ITERATIONS_TO_PLOT)
    })


PIL_AGENT_COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (148, 103, 189),
    (255, 127, 14),
    (23, 190, 207),
    (140, 86, 75),
    (127, 127, 127),
]


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"Pillow fallback skipped because Pillow is unavailable: {exc}")
        return None, None, None
    return Image, ImageDraw, ImageFont


def _font(ImageFont, size: int, bold: bool = False):
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _iteration_color(rank: int, count: int) -> tuple[int, int, int]:
    progress = rank / max(count - 1, 1)
    hue = (2.0 / 3.0) * (1.0 - progress)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.88)
    return int(255 * r), int(255 * g), int(255 * b)


def _draw_circle(draw, xy, radius, fill, outline=(0, 0, 0), width=1):
    x, y = xy
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=fill,
        outline=outline,
        width=width,
    )


def _draw_star(draw, xy, radius, fill, outline=(0, 0, 0)):
    x, y = xy
    points = []
    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        r = radius if k % 2 == 0 else 0.45 * radius
        points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=outline)


def _draw_polyline(draw, points, fill, width=2, dash: int | None = None):
    if len(points) < 2:
        return
    if dash is None:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 1e-9:
            continue
        steps = max(1, int(length // dash))
        for step in range(steps + 1):
            if step % 2:
                continue
            a0 = step / max(steps, 1)
            a1 = min((step + 1) / max(steps, 1), 1.0)
            q0 = (x0 + a0 * (x1 - x0), y0 + a0 * (y1 - y0))
            q1 = (x0 + a1 * (x1 - x0), y0 + a1 * (y1 - y0))
            draw.line([q0, q1], fill=fill, width=width)


def _point_sets_bounds(point_sets, *, equal: bool = True):
    road_points = np.array(
        [
            [experiment.WEST_ARM_X_MIN, experiment.INTERSECTION_Y_MIN],
            [experiment.EAST_ARM_X_MAX, experiment.INTERSECTION_Y_MAX],
            [experiment.INTERSECTION_X_MIN, experiment.ROAD_SOUTH_Y_MIN],
            [experiment.INTERSECTION_X_MAX, experiment.ROAD_NORTH_Y_MAX],
        ],
        dtype=float,
    )
    valid_sets = [np.asarray(points, dtype=float).reshape(-1, 2) for points in point_sets if len(points)]
    valid_sets.append(road_points)
    all_points = np.vstack(valid_sets)
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    xspan = max(float(xmax - xmin), 1.0)
    yspan = max(float(ymax - ymin), 1.0)
    if equal:
        span = max(xspan, yspan)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        xmin, xmax = cx - 0.5 * span, cx + 0.5 * span
        ymin, ymax = cy - 0.5 * span, cy + 0.5 * span
        xspan = yspan = span
    pad_x = 0.08 * xspan
    pad_y = 0.08 * yspan
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def _make_mapper(bounds, rect):
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = rect
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    def map_point(point):
        x, y = float(point[0]), float(point[1])
        px = left + (x - xmin) / xspan * (right - left)
        py = bottom - (y - ymin) / yspan * (bottom - top)
        return px, py

    return map_point


def _draw_road_pillow(draw, map_point):
    road_fill = (239, 239, 239)
    road_edge = (120, 120, 120)
    divider = (80, 80, 80)
    obstacle = (170, 40, 40)

    ns = [
        [experiment.INTERSECTION_X_MIN, experiment.ROAD_SOUTH_Y_MIN],
        [experiment.INTERSECTION_X_MAX, experiment.ROAD_SOUTH_Y_MIN],
        [experiment.INTERSECTION_X_MAX, experiment.ROAD_NORTH_Y_MAX],
        [experiment.INTERSECTION_X_MIN, experiment.ROAD_NORTH_Y_MAX],
    ]
    ew = [
        [experiment.WEST_ARM_X_MIN, experiment.INTERSECTION_Y_MIN],
        [experiment.EAST_ARM_X_MAX, experiment.INTERSECTION_Y_MIN],
        [experiment.EAST_ARM_X_MAX, experiment.INTERSECTION_Y_MAX],
        [experiment.WEST_ARM_X_MIN, experiment.INTERSECTION_Y_MAX],
    ]
    draw.polygon([map_point(p) for p in ns], fill=road_fill, outline=road_edge)
    draw.polygon([map_point(p) for p in ew], fill=road_fill, outline=road_edge)
    draw.line(
        [
            map_point([experiment.NS_DIVIDER_X, experiment.ROAD_SOUTH_Y_MIN]),
            map_point([experiment.NS_DIVIDER_X, experiment.INTERSECTION_Y_MIN]),
        ],
        fill=divider,
        width=2,
    )
    draw.line(
        [
            map_point([experiment.NS_DIVIDER_X, experiment.INTERSECTION_Y_MAX]),
            map_point([experiment.NS_DIVIDER_X, experiment.ROAD_NORTH_Y_MAX]),
        ],
        fill=divider,
        width=2,
    )
    for obs in experiment.STATIC_OBSTACLES:
        _draw_circle(draw, map_point(obs), 2.4, obstacle, outline=obstacle)


def _save_pillow_line_chart(series, path: Path, title: str, ylabel: str, *, log_y: bool = False):
    Image, ImageDraw, ImageFont = _load_pillow()
    if Image is None:
        return None
    width, height = 1400, 900
    left, top, right, bottom = 105, 95, 1020, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(ImageFont, 28, bold=True)
    label_font = _font(ImageFont, 18)
    small_font = _font(ImageFont, 15)

    draw.text((left, 30), title, fill=(20, 20, 20), font=title_font)
    draw.rectangle([left, top, right, bottom], outline=(60, 60, 60), width=2)

    n = max((len(values) for _, values, _ in series), default=1)
    transformed = []
    for name, values, color in series:
        arr = np.asarray(values, dtype=float)
        if log_y:
            arr = np.log10(np.maximum(arr, 1e-9))
        transformed.append((name, arr, color))
    all_y = np.concatenate([arr for _, arr, _ in transformed if len(arr)]) if transformed else np.array([0.0, 1.0])
    ymin = float(np.min(all_y))
    ymax = float(np.max(all_y))
    if abs(ymax - ymin) < 1e-9:
        ymin -= 1.0
        ymax += 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    def xy(iter_idx, value):
        x = left + iter_idx / max(n - 1, 1) * (right - left)
        y = bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
        return x, y

    for g in range(6):
        y = top + g / 5 * (bottom - top)
        draw.line([(left, y), (right, y)], fill=(225, 225, 225), width=1)
    for tick in [1, max(1, n // 2), n]:
        x = left + (tick - 1) / max(n - 1, 1) * (right - left)
        draw.line([(x, bottom), (x, bottom + 6)], fill=(60, 60, 60), width=1)
        draw.text((x - 8, bottom + 12), str(tick), fill=(40, 40, 40), font=small_font)

    for rank, (name, arr, color) in enumerate(transformed):
        points = [xy(k, value) for k, value in enumerate(arr)]
        _draw_polyline(draw, points, color, width=3)
        for point in points:
            _draw_circle(draw, point, 3, color, outline=color)
        legend_y = top + 28 * rank
        draw.line([(1060, legend_y + 8), (1100, legend_y + 8)], fill=color, width=4)
        draw.text((1110, legend_y), name, fill=(30, 30, 30), font=small_font)

    y_label = f"log10({ylabel})" if log_y else ylabel
    draw.text((left, height - 85), "Iteration", fill=(30, 30, 30), font=label_font)
    draw.text((25, top), y_label, fill=(30, 30, 30), font=label_font)
    image.save(path)
    return path


def _save_pillow_trajectory_convergence(iteration_trajectories, agent_specs, path: Path):
    Image, ImageDraw, ImageFont = _load_pillow()
    if Image is None:
        return None
    width, height = 1400, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(ImageFont, 28, bold=True)
    small_font = _font(ImageFont, 15)
    rect = (80, 100, 1040, 980)
    indices = trajectory_iteration_indices(len(iteration_trajectories))
    point_sets = []
    for idx in indices:
        xs_iter = iteration_trajectories[idx]
        for i in range(len(agent_specs)):
            point_sets.append(experiment.agent_xy_until_arrival(xs_iter, agent_specs, i))
    for agent in agent_specs:
        point_sets.append(np.asarray(agent.initial_state[:2], dtype=float).reshape(1, 2))
        point_sets.append(np.asarray(agent.destination, dtype=float).reshape(1, 2))
    mapper = _make_mapper(_point_sets_bounds(point_sets), rect)

    draw.text((80, 35), f"Trajectory Convergence ({len(indices)} shown)", fill=(20, 20, 20), font=title_font)
    _draw_road_pillow(draw, mapper)
    draw.rectangle(rect, outline=(60, 60, 60), width=2)

    for rank, idx in enumerate(indices):
        color = _iteration_color(rank, len(indices))
        line_width = 1 + int(3 * rank / max(len(indices) - 1, 1))
        xs_iter = iteration_trajectories[idx]
        for i in range(len(agent_specs)):
            xy = experiment.agent_xy_until_arrival(xs_iter, agent_specs, i)
            _draw_polyline(draw, [mapper(p) for p in xy], color, width=line_width)
        legend_y = 110 + 24 * rank
        draw.line([(1080, legend_y + 8), (1120, legend_y + 8)], fill=color, width=4)
        draw.text((1130, legend_y), f"iter {idx + 1}", fill=(30, 30, 30), font=small_font)

    for agent in agent_specs:
        _draw_circle(draw, mapper(agent.initial_state[:2]), 6, (50, 205, 50), width=1)
        _draw_star(draw, mapper(agent.destination), 8, (245, 205, 0))
    draw.text((80, 1000), "green circle=start, gold star=destination", fill=(40, 40, 40), font=small_font)
    image.save(path)
    return path


def _save_pillow_per_agent_evolution(iteration_trajectories, initial_xs, agent_specs, path: Path):
    Image, ImageDraw, ImageFont = _load_pillow()
    if Image is None:
        return None
    n_agents = len(agent_specs)
    ncols = 2 if n_agents <= 6 else 3
    nrows = int(math.ceil(n_agents / ncols))
    panel_w, panel_h = 660, 540
    width, height = panel_w * ncols, panel_h * nrows + 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(ImageFont, 26, bold=True)
    label_font = _font(ImageFont, 16, bold=True)
    small_font = _font(ImageFont, 13)
    indices = trajectory_iteration_indices(len(iteration_trajectories))
    draw.text((35, 25), "Initial Trajectory Evolution to Optimized Solution", fill=(20, 20, 20), font=title_font)

    for i, agent in enumerate(agent_specs):
        col = i % ncols
        row = i // ncols
        x0 = col * panel_w + 35
        y0 = row * panel_h + 85
        rect = (x0, y0 + 35, x0 + panel_w - 45, y0 + panel_h - 45)
        initial_xy = experiment.agent_xy_until_arrival(initial_xs, agent_specs, i)
        point_sets = [initial_xy]
        for idx in indices:
            point_sets.append(experiment.agent_xy_until_arrival(iteration_trajectories[idx], agent_specs, i))
        mapper = _make_mapper(_point_sets_bounds(point_sets), rect)
        draw.text((x0, y0), experiment.agent_label(agent), fill=(20, 20, 20), font=label_font)
        _draw_road_pillow(draw, mapper)
        draw.rectangle(rect, outline=(60, 60, 60), width=1)
        for rank, idx in enumerate(indices):
            xy = experiment.agent_xy_until_arrival(iteration_trajectories[idx], agent_specs, i)
            _draw_polyline(draw, [mapper(p) for p in xy], _iteration_color(rank, len(indices)), width=2)
        _draw_polyline(draw, [mapper(p) for p in initial_xy], (210, 0, 184), width=4, dash=12)
        _draw_circle(draw, mapper(agent.initial_state[:2]), 5, (50, 205, 50))
        _draw_star(draw, mapper(agent.destination), 7, (245, 205, 0))
        draw.text((x0, y0 + panel_h - 34), "magenta dashed=initial guess", fill=(40, 40, 40), font=small_font)

    image.save(path)
    return path


def _save_pillow_final_trajectory(xs, agent_specs, path: Path):
    Image, ImageDraw, ImageFont = _load_pillow()
    if Image is None:
        return None
    width, height = 1200, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(ImageFont, 28, bold=True)
    small_font = _font(ImageFont, 15)
    rect = (80, 100, 920, 930)
    point_sets = [experiment.agent_xy_until_arrival(xs, agent_specs, i) for i in range(len(agent_specs))]
    mapper = _make_mapper(_point_sets_bounds(point_sets), rect)
    draw.text((80, 35), "Optimized Trajectories", fill=(20, 20, 20), font=title_font)
    _draw_road_pillow(draw, mapper)
    draw.rectangle(rect, outline=(60, 60, 60), width=2)
    for i, agent in enumerate(agent_specs):
        color = PIL_AGENT_COLORS[i % len(PIL_AGENT_COLORS)]
        xy = experiment.agent_xy_until_arrival(xs, agent_specs, i)
        _draw_polyline(draw, [mapper(p) for p in xy], color, width=4)
        _draw_circle(draw, mapper(agent.initial_state[:2]), 6, (50, 205, 50))
        _draw_star(draw, mapper(agent.destination), 8, (245, 205, 0))
        legend_y = 110 + 28 * i
        draw.line([(950, legend_y + 8), (990, legend_y + 8)], fill=color, width=4)
        draw.text((1000, legend_y), experiment.agent_label(agent), fill=(30, 30, 30), font=small_font)
    image.save(path)
    return path


def _save_pillow_animation(xs, us, agent_specs, path: Path, fps: int = 10):
    Image, ImageDraw, ImageFont = _load_pillow()
    if Image is None:
        return None
    final_frame = experiment.game_end_step(xs, agent_specs)
    final_frame = max(1, min(final_frame, len(xs) - 1))
    frame_stride = 2
    frame_indices = list(range(0, final_frame + 1, frame_stride))
    if frame_indices[-1] != final_frame:
        frame_indices.append(final_frame)
    width, height = 900, 900
    rect = (55, 70, 845, 840)
    point_sets = [np.asarray([x[experiment.agent_slice(i)][:2] for x in xs[: final_frame + 1]]) for i in range(len(agent_specs))]
    mapper = _make_mapper(_point_sets_bounds(point_sets), rect)
    arrivals = experiment.first_arrival_steps(xs, agent_specs)
    frames = []
    title_font = _font(ImageFont, 20, bold=True)
    small_font = _font(ImageFont, 13)
    for frame in frame_indices:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((55, 25), f"Trajectory animation, t={frame * experiment.DT:.1f}s", fill=(20, 20, 20), font=title_font)
        _draw_road_pillow(draw, mapper)
        draw.rectangle(rect, outline=(60, 60, 60), width=2)
        xk = xs[frame]
        for i, agent in enumerate(agent_specs):
            color = PIL_AGENT_COLORS[i % len(PIL_AGENT_COLORS)]
            draw_until = min(frame, arrivals[i], final_frame)
            xy = point_sets[i][: draw_until + 1]
            _draw_polyline(draw, [mapper(p) for p in xy], color, width=3)
            sl = experiment.agent_slice(i)
            pos = xk[sl][:2]
            speed = float(xk[sl.start + 2])
            _draw_circle(draw, mapper(pos), 7, color, width=1)
            label_xy = mapper(pos)
            draw.text((label_xy[0] + 8, label_xy[1] - 12), f"{i + 1}: {speed:.1f}", fill=(0, 0, 0), font=small_font)
        frames.append(image)
    duration_ms = max(20, int(1000 * frame_stride / fps))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return path


def _save_pillow_optimization_evolution_animation(
    iteration_trajectories,
    initial_xs,
    agent_specs,
    path: Path,
    fps: int = 3,
):
    """Animate the trajectory update from initial guess to final ILQR iteration."""
    Image, ImageDraw, ImageFont = _load_pillow()
    if Image is None:
        return None
    indices = trajectory_iteration_indices(len(iteration_trajectories))
    width, height = 1000, 900
    rect = (60, 85, 820, 830)
    point_sets = []
    for i in range(len(agent_specs)):
        point_sets.append(experiment.agent_xy_until_arrival(initial_xs, agent_specs, i))
    for idx in indices:
        for i in range(len(agent_specs)):
            point_sets.append(experiment.agent_xy_until_arrival(iteration_trajectories[idx], agent_specs, i))
    mapper = _make_mapper(_point_sets_bounds(point_sets), rect)
    title_font = _font(ImageFont, 20, bold=True)
    small_font = _font(ImageFont, 13)
    frames = []

    def draw_frame(label: str, xs):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((60, 25), label, fill=(20, 20, 20), font=title_font)
        _draw_road_pillow(draw, mapper)
        draw.rectangle(rect, outline=(60, 60, 60), width=2)
        for i, agent in enumerate(agent_specs):
            color = PIL_AGENT_COLORS[i % len(PIL_AGENT_COLORS)]
            xy = experiment.agent_xy_until_arrival(xs, agent_specs, i)
            _draw_polyline(draw, [mapper(p) for p in xy], color, width=4)
            _draw_circle(draw, mapper(agent.initial_state[:2]), 6, (50, 205, 50))
            _draw_star(draw, mapper(agent.destination), 8, (245, 205, 0))
            legend_y = 100 + 26 * i
            draw.line([(840, legend_y + 8), (875, legend_y + 8)], fill=color, width=4)
            draw.text((885, legend_y), experiment.agent_label(agent), fill=(30, 30, 30), font=small_font)
        return image

    frames.append(draw_frame("Initial warm-start trajectory", initial_xs))
    for idx in indices:
        frames.append(draw_frame(f"ILQR trajectory iteration {idx + 1}", iteration_trajectories[idx]))
    duration_ms = max(40, int(1000 / fps))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return path


def copy_output_alias(source, target: Path):
    if not source:
        return ""
    source_path = Path(source)
    if not source_path.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != target.resolve():
        shutil.copy2(source_path, target)
    return str(target.resolve())


def add_numbered_output_aliases(row: dict[str, object], output_dir: Path, case_number: int) -> None:
    """Create simple numbered names in addition to descriptive case names."""
    alias_specs = {
        "output_csv": ("trajectory_csv", f"output_trajectory_{case_number}.csv"),
        "output_trajectory_png": ("trajectory_plot", f"output_trajectory_{case_number}.png"),
        "output_trajectory_gif": ("animation_gif", f"output_trajectory_{case_number}.gif"),
        "output_cost_convergence": ("cost_convergence_plot", f"output_cost_convergence_{case_number}.png"),
        "output_trajectory_convergence": (
            "trajectory_convergence_plot",
            f"output_trajectory_convergence_{case_number}.png",
        ),
        "output_delta_x_convergence": ("delta_x_convergence_plot", f"output_delta_x_convergence_{case_number}.png"),
        "output_initial_to_optimum": (
            "trajectory_per_agent_plot",
            f"output_initial_to_optimum_{case_number}.png",
        ),
        "output_trajectory_evolution_gif": (
            "trajectory_evolution_gif",
            f"output_trajectory_evolution_{case_number}.gif",
        ),
    }
    for alias_key, (source_key, filename) in alias_specs.items():
        row[alias_key] = copy_output_alias(row.get(source_key, ""), output_dir / filename)


def save_pillow_iteration_plots(
    history,
    iteration_trajectories,
    iteration_delta_xs,
    agent_specs: list[experiment.AgentSpec],
    prefix: Path,
) -> dict[str, str]:
    colors = [PIL_AGENT_COLORS[i % len(PIL_AGENT_COLORS)] for i in range(len(agent_specs))]
    cost_series = [
        (experiment.agent_label(agent), np.asarray(history, dtype=float)[:, i], colors[i])
        for i, agent in enumerate(agent_specs)
    ]
    cost_path = Path(f"{prefix}_costs_per_iteration.png")
    traj_path = Path(f"{prefix}_trajectory_convergence.png")
    delta_path = Path(f"{prefix}_delta_x_per_iteration.png")
    _save_pillow_line_chart(cost_series, cost_path, "Cost Convergence per Agent", "Total cost")
    _save_pillow_trajectory_convergence(iteration_trajectories, agent_specs, traj_path)
    max_dxs = [max(dxs) for dxs in iteration_delta_xs]
    _save_pillow_line_chart(
        [("max ||Delta x||", max_dxs, (176, 0, 32))],
        delta_path,
        "State-Update Convergence",
        "max ||Delta x||",
        log_y=True,
    )
    return {
        "cost_convergence_plot": str(cost_path.resolve()),
        "trajectory_convergence_plot": str(traj_path.resolve()),
        "delta_x_convergence_plot": str(delta_path.resolve()),
    }


def save_sensitivity_iteration_plots(
    history,
    iteration_trajectories,
    iteration_delta_xs,
    agent_specs: list[experiment.AgentSpec],
    prefix: Path,
    lq_solver_type: str,
) -> dict[str, str]:
    """Save cost, trajectory, and state-update convergence plots for one case."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except Exception as exc:
        print(f"Matplotlib unavailable; using Pillow fallback for iteration plots: {exc}")
        return save_pillow_iteration_plots(
            history,
            iteration_trajectories,
            iteration_delta_xs,
            agent_specs,
            prefix,
        )
    experiment.apply_publication_style(plt)

    saved_paths = {
        "cost_convergence_plot": Path(f"{prefix}_costs_per_iteration.png"),
        "trajectory_convergence_plot": Path(f"{prefix}_trajectory_convergence.png"),
        "delta_x_convergence_plot": Path(f"{prefix}_delta_x_per_iteration.png"),
    }

    # 1. Cost convergence per agent.
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    history_arr = np.asarray(history, dtype=float)
    x_iters = np.arange(1, len(history) + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_specs)))
    for i, agent in enumerate(agent_specs):
        ax.plot(
            x_iters,
            history_arr[:, i],
            color=colors[i],
            linewidth=1.75,
            alpha=0.88,
            label=experiment.agent_label(agent),
        )
        ax.scatter(
            x_iters,
            history_arr[:, i],
            s=experiment.CONVERGENCE_MARKER_SIZE,
            color=colors[i],
            alpha=experiment.CONVERGENCE_MARKER_ALPHA,
            edgecolors="none",
            zorder=4,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total cost")
    ax.set_title(f"Cost Convergence per Agent ({lq_solver_type})")
    experiment.format_publication_axes(ax)
    ax.legend(
        title="Vehicle",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.94,
    )
    fig.savefig(saved_paths["cost_convergence_plot"], dpi=experiment.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2. Trajectory convergence.  Default sweeps run at least 20 iterations, so
    # this plot shows 20 evolution snapshots unless the user explicitly asks for
    # an early-convergence quick run.
    fig, ax = plt.subplots(figsize=(8.6, 7.0), constrained_layout=True)
    experiment.draw_road_scene(ax)
    iter_indices = trajectory_iteration_indices(len(iteration_trajectories))
    cmap = plt.colormaps["turbo"]
    norm = Normalize(vmin=0, vmax=max(len(iteration_trajectories) - 1, 1))

    for rank, idx in enumerate(iter_indices):
        progress = rank / max(len(iter_indices) - 1, 1)
        color = cmap(norm(idx))
        linewidth = 0.60 + 1.35 * progress
        alpha = 0.32 + 0.55 * progress
        xs_iter = iteration_trajectories[idx]
        for i, agent in enumerate(agent_specs):
            xy = experiment.agent_xy_until_arrival(xs_iter, agent_specs, i)
            label = f"Iteration {idx + 1}" if i == 0 else None
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                zorder=3,
            )

    for agent in agent_specs:
        ax.scatter(
            [agent.initial_state[0]],
            [agent.initial_state[1]],
            marker="o",
            s=54,
            color="limegreen",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )
        ax.scatter(
            [agent.destination[0]],
            [agent.destination[1]],
            marker="*",
            s=96,
            color="gold",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Trajectory Convergence ({lq_solver_type})\n"
        f"{len(iter_indices)} shown of {len(iteration_trajectories)} iterations"
    )
    experiment.format_publication_axes(ax, equal=True)
    ax.legend(
        title="Shown iteration",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.94,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("ILQR Iteration")
    if iter_indices:
        tick_step = max(1, len(iter_indices) // 10)
        tick_indices = iter_indices[::tick_step]
        if iter_indices[-1] not in tick_indices:
            tick_indices = tick_indices + [iter_indices[-1]]
        cbar.set_ticks(tick_indices)
        cbar.set_ticklabels([str(i + 1) for i in tick_indices])
    fig.savefig(saved_paths["trajectory_convergence_plot"], dpi=experiment.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # 3. Max delta-x convergence.
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
        s=experiment.CONVERGENCE_MARKER_SIZE,
        color="#B00020",
        alpha=experiment.CONVERGENCE_MARKER_ALPHA,
        edgecolors="none",
        zorder=4,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Max $\|\Delta x\|$")
    ax.set_title(f"State-Update Convergence ({lq_solver_type})")
    ax.set_yscale("log")
    experiment.format_publication_axes(ax)
    ax.grid(True, which="both", color="#b6b6b6", linewidth=0.55, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    fig.savefig(saved_paths["delta_x_convergence_plot"], dpi=experiment.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return {key: str(path.resolve()) for key, path in saved_paths.items()}


def run_case(
    case: SensitivityCase,
    *,
    case_number: int,
    output_dir: Path,
    max_iterations: int,
    min_iterations: int,
    convergence_tol: float,
    alpha_scaling: float,
    alpha_line_search: bool,
    alpha_line_search_min: float,
    alpha_line_search_shrink: float,
    alpha_line_search_max_growth: float,
    alpha_line_search_start_iteration: int,
    lq_solver_type: str,
    save_plots: bool,
    save_gifs: bool,
) -> dict[str, object]:
    print("")
    print(f"=== Running {case.name} ===")
    print(case.description)

    game = experiment.build_multi_agent_game(case.agents)
    solver = ILQSolver(
        game,
        max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        alpha_scaling=alpha_scaling,
        use_euler=True,
        lq_solver_type=lq_solver_type,
        min_iterations=min_iterations,
        alpha_line_search=alpha_line_search,
        alpha_line_search_min=alpha_line_search_min,
        alpha_line_search_shrink=alpha_line_search_shrink,
        alpha_line_search_max_growth=alpha_line_search_max_growth,
        alpha_line_search_start_iteration=alpha_line_search_start_iteration,
    )
    initial_operating_point = make_initial_guess(game, case.agents, case.initial_guess)
    solver.current_operating_point = initial_operating_point
    initial_xs = initial_operating_point[0]

    t0 = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - t0
    result["name"] = case.name

    result["xs"], result["us"] = experiment.freeze_arrived_agents(
        result["xs"],
        result["us"],
        case.agents,
    )

    case_prefix = output_dir / case.name
    csv_path = experiment.save_multi_agent_csv(
        result["xs"],
        result["us"],
        case.agents,
        path=f"{case_prefix}.csv",
    )
    plot_path = None
    iteration_plot_paths = {}
    trajectory_per_agent_path = None
    animation_path = None
    trajectory_evolution_gif_path = None
    if save_plots:
        iteration_plot_paths = save_sensitivity_iteration_plots(
            result["history"],
            result["iteration_trajectories"],
            result["iteration_delta_xs"],
            case.agents,
            prefix=case_prefix,
            lq_solver_type=lq_solver_type,
        )
        trajectory_per_agent_path = experiment.maybe_save_trajectory_evolution_per_agent(
            result["iteration_trajectories"],
            initial_xs,
            case.agents,
            prefix=case_prefix,
        )
        if trajectory_per_agent_path is None:
            trajectory_per_agent_path = _save_pillow_per_agent_evolution(
                result["iteration_trajectories"],
                initial_xs,
                case.agents,
                Path(f"{case_prefix}_trajectory_per_agent_per_iteration.png"),
            )
        plot_path = experiment.maybe_save_multi_agent_plot(
            result["xs"],
            case.agents,
            path=f"{case_prefix}.png",
        )
        if plot_path is None:
            plot_path = _save_pillow_final_trajectory(
                result["xs"],
                case.agents,
                Path(f"{case_prefix}.png"),
            )
        trajectory_evolution_gif_path = _save_pillow_optimization_evolution_animation(
            result["iteration_trajectories"],
            initial_xs,
            case.agents,
            Path(f"{case_prefix}_trajectory_evolution.gif"),
            fps=3,
        )
    if save_gifs:
        animation_path = experiment.maybe_save_xy_animation(
            result["xs"],
            result["us"],
            case.agents,
            path=f"{case_prefix}_animation.gif",
            fps=10,
        )
        if animation_path is None:
            animation_path = _save_pillow_animation(
                result["xs"],
                result["us"],
                case.agents,
                Path(f"{case_prefix}_animation.gif"),
                fps=10,
            )

    row = summarize_result(case, result, case.agents, elapsed, max_iterations)
    row["trajectory_csv"] = str(Path(csv_path).resolve())
    row["trajectory_plot"] = str(Path(plot_path).resolve()) if plot_path else ""
    row.update(iteration_plot_paths)
    row["trajectory_per_agent_plot"] = (
        str(Path(trajectory_per_agent_path).resolve()) if trajectory_per_agent_path else ""
    )
    row["animation_gif"] = str(Path(animation_path).resolve()) if animation_path else ""
    row["trajectory_evolution_gif"] = (
        str(Path(trajectory_evolution_gif_path).resolve()) if trajectory_evolution_gif_path else ""
    )
    row["trajectory_iterations_plotted"] = len(
        trajectory_iteration_indices(len(result["iteration_trajectories"]))
    )
    add_numbered_output_aliases(row, output_dir, case_number)
    print(
        f"{case.name}: final cost={row['final_total_cost']}, "
        f"min pairwise distance={row['min_pairwise_distance']}, "
        f"iterations={row['iterations']}"
    )
    return row


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ILQ sensitivity-analysis batches.")
    parser.add_argument("--switch", type=int, default=SWITCH, choices=[1, 2, 3])
    parser.add_argument("--max-iterations", type=int, default=SENSITIVITY_MAX_ITERATIONS)
    parser.add_argument("--convergence-tol", type=float, default=SENSITIVITY_CONVERGENCE_TOL)
    parser.add_argument("--alpha-scaling", type=float, default=SENSITIVITY_ALPHA_SCALING)
    parser.add_argument(
        "--no-alpha-line-search",
        action="store_true",
        help="Use the fixed --alpha-scaling value instead of backtracking line search.",
    )
    parser.add_argument("--alpha-line-search-min", type=float, default=SENSITIVITY_ALPHA_LINE_SEARCH_MIN)
    parser.add_argument("--alpha-line-search-shrink", type=float, default=SENSITIVITY_ALPHA_LINE_SEARCH_SHRINK)
    parser.add_argument(
        "--alpha-line-search-max-growth",
        type=float,
        default=SENSITIVITY_ALPHA_LINE_SEARCH_MAX_GROWTH,
    )
    parser.add_argument(
        "--alpha-line-search-start-iteration",
        type=int,
        default=SENSITIVITY_ALPHA_LINE_SEARCH_START_ITERATION,
    )
    parser.add_argument(
        "--lq-solver-type",
        choices=["feedback", "open_loop"],
        default=SENSITIVITY_LQ_SOLVER_TYPE,
    )
    parser.add_argument("--case-limit", type=int, default=CASE_LIMIT)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument(
        "--allow-early-convergence",
        action="store_true",
        help="Allow the solver to stop before 20 iterations; useful only for quick smoke runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = OUTPUT_DIR / f"switch_{args.switch}_{args.lq_solver_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_iterations = int(args.max_iterations)
    convergence_tol = float(args.convergence_tol)
    min_iterations = 0 if args.allow_early_convergence else MIN_TRAJECTORY_ITERATIONS
    if not args.allow_early_convergence:
        if max_iterations < MIN_TRAJECTORY_ITERATIONS:
            print(
                f"Raising max iterations from {max_iterations} to "
                f"{MIN_TRAJECTORY_ITERATIONS} so trajectory plots show at least "
                f"{MIN_TRAJECTORY_ITERATIONS} iterations."
            )
            max_iterations = MIN_TRAJECTORY_ITERATIONS

    cases = build_cases(args.switch)
    if args.case_limit is not None:
        cases = cases[: args.case_limit]

    rows: list[dict[str, object]] = []
    for case_number, case in enumerate(cases, start=1):
        try:
            row = run_case(
                case,
                case_number=case_number,
                output_dir=output_dir,
                max_iterations=max_iterations,
                min_iterations=min_iterations,
                convergence_tol=convergence_tol,
                alpha_scaling=args.alpha_scaling,
                alpha_line_search=(
                    SENSITIVITY_USE_ALPHA_LINE_SEARCH and not args.no_alpha_line_search
                ),
                alpha_line_search_min=args.alpha_line_search_min,
                alpha_line_search_shrink=args.alpha_line_search_shrink,
                alpha_line_search_max_growth=args.alpha_line_search_max_growth,
                alpha_line_search_start_iteration=args.alpha_line_search_start_iteration,
                lq_solver_type=args.lq_solver_type,
                save_plots=SAVE_PLOTS and not args.no_plots,
                save_gifs=SAVE_GIFS and not args.no_gifs,
            )
        except Exception as exc:
            row = {
                "case": case.name,
                "description": case.description,
                "status": "failed",
                "error": repr(exc),
                "agents": len(case.agents),
                "agent_names": ";".join(agent.name for agent in case.agents),
                "initial_guess": case.initial_guess,
            }
            print(f"{case.name}: FAILED: {exc!r}")
        rows.append(row)

    summary_path = output_dir / "summary.csv"
    write_summary(rows, summary_path)
    print("")
    print(f"Saved sensitivity summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
