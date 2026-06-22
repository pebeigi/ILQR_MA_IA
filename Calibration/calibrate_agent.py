"""Trajectory-level calibration of one agent's ILQR cost weights via BO.

For a given observed case we minimize the pointwise (arclength-resampled)
distance between the ILQR-simulated path and the observed path, searching over
the agent's behavioral cost weights with Bayesian optimization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bayes_opt import BOResult, minimize
from .ilqr_interface import AgentParameters, solve_single_agent
from .observed_cases import ObservedCase, load_case
from .parameters import ParameterSpace
from .trajectory_error import TrajectoryError, trajectory_error

FAILURE_PENALTY = 1e6


@dataclass
class CalibrationResult:
    case_id: str
    movement_name: str
    best_params: dict
    best_score: float
    error_breakdown: dict
    n_evaluations: int
    history: list[float]


def _build_scenario(case: ObservedCase, boundary_options: dict | None):
    if not boundary_options or not boundary_options.get("use_boundaries"):
        return case.to_scenario()
    curbs = case.local_curb_points(
        spacing_m=boundary_options.get("spacing_m", 2.0),
        radius_m=boundary_options.get("radius_m", 40.0),
    )
    return case.to_scenario(
        boundary_obstacles=curbs,
        boundary_weight=boundary_options.get("weight", 1000.0),
        boundary_epsilon=boundary_options.get("epsilon", 0.1),
    )


def make_objective(
    case: ObservedCase,
    space: ParameterSpace,
    *,
    max_iterations: int,
    n_samples: int,
    boundary_options: dict | None = None,
):
    scenario = _build_scenario(case, boundary_options)

    def objective(search_vector: np.ndarray) -> float:
        params = space.to_params(search_vector)
        try:
            result = solve_single_agent(scenario, params, max_iterations=max_iterations)
        except Exception:
            return FAILURE_PENALTY
        sim_path = result.states[:, :2]
        if not np.all(np.isfinite(sim_path)):
            return FAILURE_PENALTY
        err = trajectory_error(sim_path, case.path_local, n_samples=n_samples)
        return err.score

    return objective, scenario


def calibrate_case(
    case_id: str,
    *,
    trajectory_points_path: Path | None = None,
    space: ParameterSpace | None = None,
    n_initial: int = 8,
    n_iterations: int = 30,
    solver_iterations: int = 60,
    error_samples: int = 100,
    seed: int = 0,
    verbose: bool = True,
    boundary_options: dict | None = None,
    flip_y: bool = False,
) -> CalibrationResult:
    space = space or ParameterSpace()
    if trajectory_points_path is None:
        case = load_case(case_id, flip_y=flip_y)
    else:
        case = load_case(case_id, trajectory_points_path, flip_y=flip_y)

    objective, _ = make_objective(
        case,
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

    best_params = space.to_params(bo.best_x)
    scenario = _build_scenario(case, boundary_options)
    best_result = solve_single_agent(scenario, best_params, max_iterations=solver_iterations)
    err: TrajectoryError = trajectory_error(
        best_result.states[:, :2], case.path_local, n_samples=error_samples
    )

    return CalibrationResult(
        case_id=case.case_id,
        movement_name=case.movement_name,
        best_params=asdict(best_params),
        best_score=float(bo.best_y),
        error_breakdown=asdict(err),
        n_evaluations=int(len(bo.ys)),
        history=[float(v) for v in bo.history],
    )


def save_result(result: CalibrationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case = result.case_id.replace("/", "_")
    path = output_dir / f"calibration_{safe_case}.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path
