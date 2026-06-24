"""Joint Nash-game calibration of per-agent ILQR cost weights.

All co-present agents are optimized together as a Nash game.  Each agent has its
own cost-weight vector, and the objective is the mean trajectory error across
*all* agents (every simulated path vs its observed path).

Because per-agent weights make a single joint Bayesian optimization very high
dimensional (N x D), we calibrate with **block coordinate descent**: we cycle
over agents, and for each agent run a low-dimensional BO over just that agent's
weights while the full game is re-solved (with the other agents' current
weights) at every evaluation.  This keeps each BO subproblem the same size as
the single-agent calibration while still fitting per-agent weights against the
joint game and the all-agents objective.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bayes_opt import minimize
from .nash_cases import GameScene, build_game_scene
from .nash_interface import (
    NashAgentParameters,
    default_params_for_scene,
    solve_nash,
)
from .parameters import DEFAULT_PARAMETER_DEFS, ParameterDef, ParameterSpace
from .paths import safe_case_filename
from .trajectory_error import trajectory_error

FAILURE_PENALTY = 1e6


def _json_default(obj):
    """Coerce numpy scalars/arrays so calibration results serialize to JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Per-agent interaction weights (with the other game players).
NASH_INTERACTION_DEFS: tuple[ParameterDef, ...] = (
    ParameterDef("agent_repulsion", 1.0, 5000.0, "log10"),
    ParameterDef("agent_proximity_speed", 0.0, 300.0, "linear"),
    ParameterDef("agent_activation_distance", 4.0, 25.0, "linear"),
)
NASH_PARAMETER_DEFS: tuple[ParameterDef, ...] = DEFAULT_PARAMETER_DEFS + NASH_INTERACTION_DEFS


@dataclass
class AgentCalibration:
    name: str
    vehicle_id: int | str
    is_ego: bool
    error: float
    params: dict


@dataclass
class IgnoredAgent:
    name: str
    vehicle_id: int | str
    path_length_m: float
    reason: str = "stopped (< min travel distance)"


@dataclass
class NashCalibrationResult:
    case_id: str
    movement_name: str
    n_agents: int
    n_ignored: int
    min_travel_m: float
    y_flipped: bool
    mean_error: float
    agents: list[AgentCalibration]
    ignored_agents: list[IgnoredAgent] = field(default_factory=list)
    n_evaluations: int = 0
    history: list[float] = field(default_factory=list)


def _per_agent_errors(
    scene: GameScene,
    params_list: list[NashAgentParameters],
    *,
    boundary_options: dict | None,
    max_iterations: int,
    n_samples: int,
) -> tuple[list[float], list]:
    boundary = _boundary_obstacles(scene, boundary_options)
    weight = (boundary_options or {}).get("weight", 1000.0)
    try:
        results = solve_nash(
            scene,
            params_list,
            boundary_obstacles=boundary,
            boundary_weight=weight,
            max_iterations=max_iterations,
        )
    except Exception:
        return [FAILURE_PENALTY] * scene.n_agents, []

    errors: list[float] = []
    for track, res in zip(scene.agents, results):
        sim = res.states[track.entry_step : track.exit_step + 1, :2]
        obs = track.observed_path
        if sim.shape[0] < 2 or not np.all(np.isfinite(sim)):
            errors.append(FAILURE_PENALTY)
            continue
        errors.append(trajectory_error(sim, obs, n_samples=n_samples).score)
    return errors, results


def _boundary_obstacles(scene: GameScene, boundary_options: dict | None) -> np.ndarray | None:
    if not boundary_options or not boundary_options.get("use_boundaries"):
        return None
    from .real_boundaries import local_boundary_points

    return local_boundary_points(
        scene.ego_origin,
        spacing_m=boundary_options.get("spacing_m", 2.0),
        radius_m=boundary_options.get("radius_m", 40.0),
        flip_y=scene.y_flipped,
    )


def calibrate_nash_case(
    case_id: str,
    *,
    trajectory_points_path: Path | None = None,
    radius_m: float = 35.0,
    dt: float = 0.1,
    max_agents: int | None = 6,
    min_presence_frac: float = 0.5,
    min_travel_m: float = 5.0,
    rounds: int = 2,
    n_initial: int = 6,
    n_iterations: int = 15,
    solver_iterations: int = 50,
    error_samples: int = 80,
    seed: int = 0,
    verbose: bool = True,
    boundary_options: dict | None = None,
    flip_y: bool = False,
) -> tuple[NashCalibrationResult, GameScene, list[NashAgentParameters]]:
    scene = build_game_scene(
        case_id,
        trajectory_points_path=trajectory_points_path,
        dt=dt,
        radius_m=radius_m,
        min_presence_frac=min_presence_frac,
        min_travel_m=min_travel_m,
        max_agents=max_agents,
        flip_y=flip_y,
    )
    space = ParameterSpace(defs=NASH_PARAMETER_DEFS)
    params_list = default_params_for_scene(scene)
    if verbose:
        print(
            f"[nash] case {case_id}: {scene.n_agents} active agents "
            f"(ego + {scene.n_neighbors} neighbors), "
            f"{scene.n_ignored} ignored (travel < {min_travel_m:.1f} m), "
            f"horizon {scene.horizon_steps} steps"
        )
        for track in scene.ignored_agents:
            print(
                f"  ignored {track.name} (veh {track.vehicle_id}): "
                f"path length {track.path_length_m:.2f} m"
            )

    def mean_error(plist: list[NashAgentParameters]) -> float:
        errs, _ = _per_agent_errors(
            scene, plist,
            boundary_options=boundary_options,
            max_iterations=solver_iterations,
            n_samples=error_samples,
        )
        return float(np.mean(errs))

    current_error = mean_error(params_list)
    best_error = current_error
    best_params = list(params_list)
    history = [current_error]
    n_eval = 1
    if verbose:
        print(f"[nash] initial mean error = {current_error:.3f} m")

    for rnd in range(rounds):
        for i, track in enumerate(scene.agents):
            def objective(vec: np.ndarray, _i: int = i) -> float:
                trial = list(params_list)
                trial[_i] = space.to_params(vec, base=params_list[_i])
                errs, _ = _per_agent_errors(
                    scene, trial,
                    boundary_options=boundary_options,
                    max_iterations=solver_iterations,
                    n_samples=error_samples,
                )
                return float(np.mean(errs))

            bo = minimize(
                objective,
                space.search_bounds,
                n_initial=n_initial,
                n_iterations=n_iterations,
                seed=seed + rnd * scene.n_agents + i,
                verbose=False,
            )
            n_eval += len(bo.ys)
            # Coordinate descent: accept this agent's update only if it improves
            # the joint objective (keeps the sweep monotone non-increasing).
            if bo.best_y < current_error:
                params_list[i] = space.to_params(bo.best_x, base=params_list[i])
                current_error = float(bo.best_y)
                if current_error < best_error:
                    best_error = current_error
                    best_params = list(params_list)
            history.append(current_error)
            if verbose:
                print(f"[nash] round {rnd + 1}/{rounds}  agent {i + 1}/{scene.n_agents} "
                      f"({track.name})  mean error -> {current_error:.3f} m")

    # Final evaluation with the best parameter set for per-agent breakdown.
    errors, _ = _per_agent_errors(
        scene, best_params,
        boundary_options=boundary_options,
        max_iterations=solver_iterations,
        n_samples=error_samples,
    )
    agent_calibs = [
        AgentCalibration(
            name=track.name,
            vehicle_id=track.vehicle_id,
            is_ego=track.is_ego,
            error=float(err),
            params=asdict(params),
        )
        for track, params, err in zip(scene.agents, best_params, errors)
    ]

    result = NashCalibrationResult(
        case_id=case_id,
        movement_name=scene.agents[0].name if scene.agents else "",
        n_agents=scene.n_agents,
        n_ignored=scene.n_ignored,
        min_travel_m=min_travel_m,
        y_flipped=scene.y_flipped,
        mean_error=float(np.mean(errors)),
        agents=agent_calibs,
        ignored_agents=[
            IgnoredAgent(
                name=t.name,
                vehicle_id=t.vehicle_id,
                path_length_m=float(t.path_length_m),
            )
            for t in scene.ignored_agents
        ],
        n_evaluations=int(n_eval),
        history=[float(v) for v in history],
    )
    return result, scene, best_params


def save_result(result: NashCalibrationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case = safe_case_filename(result.case_id)
    suffix = "_flip_y" if result.y_flipped else ""
    stem = f"nash_calibration_{safe_case}{suffix}"
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(asdict(result), indent=2, default=_json_default), encoding="utf-8"
    )
    _save_agent_params_csv(result, output_dir / f"{stem}_agent_params.csv")
    return json_path


def _save_agent_params_csv(result: NashCalibrationResult, path: Path) -> Path:
    import pandas as pd

    rows = []
    for a in result.agents:
        rows.append({
            "name": a.name,
            "vehicle_id": a.vehicle_id,
            "is_ego": a.is_ego,
            "calibrated": True,
            "error": a.error,
            **a.params,
        })
    for a in result.ignored_agents:
        rows.append({
            "name": a.name,
            "vehicle_id": a.vehicle_id,
            "is_ego": False,
            "calibrated": False,
            "error": np.nan,
            "path_length_m": a.path_length_m,
            "ignore_reason": a.reason,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
