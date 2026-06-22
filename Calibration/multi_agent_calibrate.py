"""Trajectory-level calibration of the EGO agent among replayed neighbors.

Same objective as the single-agent calibrator (pointwise arclength-resampled
distance between simulated and observed ego paths) but the ego is simulated in
the presence of the real co-present vehicles, which act as moving obstacles.
The search space adds the ego's interaction weights on top of its behavioral
weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bayes_opt import BOResult, minimize
from .calibration_outputs import save_calibration_artifacts
from .ilqr_interface import ScenarioSpec
from .multi_agent_cases import NeighborScene, build_neighbor_scene
from .multi_agent_interface import EgoParameters, solve_ego
from .parameters import DEFAULT_PARAMETER_DEFS, ParameterDef, ParameterSpace
from .paths import safe_case_filename
from .trajectory_error import TrajectoryError, trajectory_error

FAILURE_PENALTY = 1e6

# Behavioral defs (shared with single-agent) plus ego interaction weights.
EGO_INTERACTION_DEFS: tuple[ParameterDef, ...] = (
    ParameterDef("neighbor_repulsion", 1.0, 5000.0, "log10"),
    ParameterDef("neighbor_proximity_speed", 0.0, 300.0, "linear"),
    ParameterDef("neighbor_activation_distance", 4.0, 25.0, "linear"),
)
EGO_PARAMETER_DEFS: tuple[ParameterDef, ...] = DEFAULT_PARAMETER_DEFS + EGO_INTERACTION_DEFS


@dataclass
class MultiAgentCalibrationResult:
    case_id: str
    movement_name: str
    n_neighbors: int
    y_flipped: bool
    best_params: dict
    best_score: float
    error_breakdown: dict
    n_evaluations: int
    history: list[float]


def _build_scenario(scene: NeighborScene, boundary_options: dict | None) -> ScenarioSpec:
    case = scene.ego
    if not boundary_options or not boundary_options.get("use_boundaries"):
        return case.to_scenario(min_horizon=scene.horizon_steps)
    curbs = case.local_curb_points(
        spacing_m=boundary_options.get("spacing_m", 2.0),
        radius_m=boundary_options.get("radius_m", 40.0),
    )
    return case.to_scenario(
        min_horizon=scene.horizon_steps,
        boundary_obstacles=curbs,
        boundary_weight=boundary_options.get("weight", 1000.0),
        boundary_epsilon=boundary_options.get("epsilon", 0.1),
    )


def make_objective(
    scene: NeighborScene,
    space: ParameterSpace,
    *,
    max_iterations: int,
    n_samples: int,
    boundary_options: dict | None = None,
):
    scenario = _build_scenario(scene, boundary_options)
    observed = scene.ego.path_local

    def objective(search_vector: np.ndarray) -> float:
        params = space.to_params(search_vector, base=EgoParameters())
        try:
            result = solve_ego(
                scenario,
                scene.positions_per_step,
                params,
                max_iterations=max_iterations,
            )
        except Exception:
            return FAILURE_PENALTY
        sim_path = result.states[:, :2]
        if not np.all(np.isfinite(sim_path)):
            return FAILURE_PENALTY
        err = trajectory_error(sim_path, observed, n_samples=n_samples)
        return err.score

    return objective, scenario


def calibrate_ego_case(
    case_id: str,
    *,
    trajectory_points_path: Path | None = None,
    space: ParameterSpace | None = None,
    radius_m: float = 35.0,
    dt: float = 0.1,
    n_initial: int = 10,
    n_iterations: int = 40,
    solver_iterations: int = 60,
    error_samples: int = 100,
    seed: int = 0,
    verbose: bool = True,
    boundary_options: dict | None = None,
    flip_y: bool = False,
) -> tuple[MultiAgentCalibrationResult, NeighborScene, EgoParameters, BOResult, ParameterSpace]:
    space = space or ParameterSpace(defs=EGO_PARAMETER_DEFS)
    scene = build_neighbor_scene(
        case_id,
        trajectory_points_path=trajectory_points_path,
        dt=dt,
        radius_m=radius_m,
        flip_y=flip_y,
    )
    if verbose:
        print(f"[multi-agent] case {case_id}: {scene.n_neighbors} neighbor(s), "
              f"horizon {scene.horizon_steps} steps")

    objective, _ = make_objective(
        scene,
        space,
        max_iterations=solver_iterations,
        n_samples=error_samples,
        boundary_options=boundary_options,
    )

    bo: BOResult = minimize(
        objective,
        space.search_bounds,
        n_initial=n_initial,
        n_iterations=n_iterations,
        seed=seed,
        verbose=verbose,
    )

    best_params = space.to_params(bo.best_x, base=EgoParameters())
    scenario = _build_scenario(scene, boundary_options)
    best_result = solve_ego(
        scenario, scene.positions_per_step, best_params, max_iterations=solver_iterations
    )
    err: TrajectoryError = trajectory_error(
        best_result.states[:, :2], scene.ego.path_local, n_samples=error_samples
    )

    result = MultiAgentCalibrationResult(
        case_id=scene.ego.case_id,
        movement_name=scene.ego.movement_name,
        n_neighbors=scene.n_neighbors,
        y_flipped=scene.ego.y_flipped,
        best_params=asdict(best_params),
        best_score=float(bo.best_y),
        error_breakdown=asdict(err),
        n_evaluations=int(len(bo.ys)),
        history=[float(v) for v in bo.history],
    )
    return result, scene, best_params, bo, space


def save_result(
    result: MultiAgentCalibrationResult,
    output_dir: Path,
    *,
    bo: BOResult | None = None,
    space: ParameterSpace | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case = safe_case_filename(result.case_id)
    suffix = "_flip_y" if result.y_flipped else ""
    path = output_dir / f"multi_agent_calibration_{safe_case}{suffix}.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    if bo is not None and space is not None:
        save_calibration_artifacts(
            result,
            bo,
            space,
            output_dir,
            prefix=f"multi_agent_calibration_{safe_case}",
            suffix=suffix,
            base=EgoParameters(),
            plot_title=f"Multi-agent calibration: {result.case_id}",
        )
    return path
