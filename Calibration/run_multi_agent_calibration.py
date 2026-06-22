"""CLI: calibrate the EGO agent's ILQR weights among replayed neighbors.

The non-ego vehicles present during the maneuver are replayed from the raw
TGSIM data and act as moving obstacles; only the ego is optimized.

Example:
    python -m Calibration.run_multi_agent_calibration \
        --case-id 719_1_2_5_I_WB_->_23_SB_middle --plot
    python -m Calibration.run_multi_agent_calibration --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .multi_agent_cases import NeighborScene, build_neighbor_scene
from .multi_agent_calibrate import (
    EGO_PARAMETER_DEFS,
    calibrate_ego_case,
    save_result,
)
from .calibration_outputs import save_batch_summary_csv
from .multi_agent_interface import EgoParameters, solve_ego
from .observed_cases import DEFAULT_TRAJECTORY_POINTS, list_cases
from .parameters import ParameterSpace
from .paths import safe_case_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Calibration" / "outputs" / "multi_agent_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent (ego) ILQR calibration via Bayesian optimization.")
    parser.add_argument("--case-id", type=str, default=None, help="Ego case_id to calibrate.")
    parser.add_argument("--max-cases", type=int, default=None, help="Calibrate the first N case_ids from --trajectory-points.")
    parser.add_argument("--list", action="store_true", help="List available case_ids and exit.")
    parser.add_argument("--trajectory-points", type=Path, default=DEFAULT_TRAJECTORY_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--radius", type=float, default=35.0, help="Neighbor proximity threshold (m) to the ego path.")
    parser.add_argument("--n-initial", type=int, default=10, help="Random evaluations before the GP starts.")
    parser.add_argument("--n-iterations", type=int, default=40, help="Bayesian-optimization iterations.")
    parser.add_argument("--solver-iterations", type=int, default=60, help="ILQR iterations per evaluation.")
    parser.add_argument("--error-samples", type=int, default=100, help="Arclength resample count for the error metric.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip-y", action="store_true", help="Use y := -y for ego data, replayed neighbors, and curbs.")
    parser.add_argument("--plot", action="store_true", help="Save an ego/neighbor trajectory plot.")
    parser.add_argument("--use-boundaries", action="store_true", help="Add real Foggy Bottom street curbs to the simulation.")
    parser.add_argument("--boundary-radius", type=float, default=40.0)
    parser.add_argument("--boundary-spacing", type=float, default=2.0)
    parser.add_argument("--boundary-weight", type=float, default=1000.0)
    return parser.parse_args()


def _boundary_options(args: argparse.Namespace) -> dict:
    return {
        "use_boundaries": args.use_boundaries,
        "radius_m": args.boundary_radius,
        "spacing_m": args.boundary_spacing,
        "weight": args.boundary_weight,
    }


def _save_plot(
    scene: NeighborScene,
    best_params: EgoParameters,
    output_dir: Path,
    boundary_options: dict | None = None,
) -> tuple[Path, Path] | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Plot skipped (matplotlib unavailable): {exc}")
        return None

    case = scene.ego
    curbs = None
    if boundary_options and boundary_options.get("use_boundaries"):
        curbs = case.local_curb_points(
            spacing_m=boundary_options.get("spacing_m", 2.0),
            radius_m=boundary_options.get("radius_m", 40.0),
        )
        scenario = case.to_scenario(
            min_horizon=scene.horizon_steps,
            boundary_obstacles=curbs,
            boundary_weight=boundary_options.get("weight", 1000.0),
        )
    else:
        scenario = case.to_scenario(min_horizon=scene.horizon_steps)

    result = solve_ego(scenario, scene.positions_per_step, best_params, max_iterations=120)
    sim = result.states[:, :2]
    obs = case.path_local

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    if curbs is not None and len(curbs):
        ax.scatter(curbs[:, 0], curbs[:, 1], s=8, color="#bbbbbb", label="real street curbs")

    for i, track in enumerate(scene.tracks):
        xy = track.xy_local
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.0, color="#999999", alpha=0.8,
                label="neighbors" if i == 0 else None)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=12, color="#999999")

    # Snapshot of neighbor positions at the closest-approach step (context).
    occupied = [k for k, p in enumerate(scene.positions_per_step) if len(p)]
    if occupied:
        k_mid = occupied[len(occupied) // 2]
        snap = scene.positions_per_step[k_mid]
        ax.scatter(snap[:, 0], snap[:, 1], s=40, facecolors="none",
                   edgecolors="#ff7f0e", label="neighbors @ mid-step")

    ax.plot(obs[:, 0], obs[:, 1], "o-", ms=3, lw=1.5, color="#1f77b4", label="ego observed")
    ax.plot(sim[:, 0], sim[:, 1], "-", lw=2.0, color="#d62728", label="ego ILQR (calibrated)")
    ax.scatter([obs[0, 0]], [obs[0, 1]], color="green", zorder=5, label="ego start")
    ax.scatter([obs[-1, 0]], [obs[-1, 1]], color="black", marker="x", zorder=5, label="ego observed end")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m] (ego local frame)")
    ax.set_ylabel("y [m] (ego local frame)")
    ax.set_title(f"Multi-agent calibration: {case.case_id}\n"
                 f"{case.movement_name}  ({scene.n_neighbors} neighbors)"
                 f"{'  [y flipped]' if case.y_flipped else ''}")
    ax.legend(loc="best", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case = safe_case_filename(case.case_id)
    suffix = "_flip_y" if case.y_flipped else ""
    path = output_dir / f"multi_agent_calibration_{safe_case}{suffix}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    speed_obs = np.zeros(len(obs), dtype=float)
    if len(obs) > 1:
        dt = np.diff(case.times)
        dt = np.where(dt > 0.0, dt, np.nan)
        seg_speed = np.linalg.norm(np.diff(obs, axis=0), axis=1) / dt
        speed_obs[0] = seg_speed[0] if np.isfinite(seg_speed[0]) else 0.0
        speed_obs[1:] = np.nan_to_num(seg_speed, nan=speed_obs[0])

    time_path = output_dir / f"multi_agent_calibration_{safe_case}{suffix}_timeseries.png"
    fig_ts, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(case.times, obs[:, 0], "o-", ms=3, lw=1.2, color="#1f77b4", label="observed")
    axes[0].plot(result.times, sim[:, 0], "-", lw=1.8, color="#d62728", label="ILQR")
    axes[0].set_ylabel("x [m]")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(case.times, obs[:, 1], "o-", ms=3, lw=1.2, color="#1f77b4")
    axes[1].plot(result.times, sim[:, 1], "-", lw=1.8, color="#d62728")
    axes[1].set_ylabel("y [m]")

    axes[2].plot(case.times, speed_obs, "o-", ms=3, lw=1.2, color="#1f77b4")
    axes[2].plot(result.times, result.states[:, 2], "-", lw=1.8, color="#d62728")
    axes[2].set_ylabel("speed [m/s]")
    axes[2].set_xlabel("time [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig_ts.suptitle(
        f"Multi-agent calibration time series: {case.case_id}"
        f"{'  [y flipped]' if case.y_flipped else ''}"
    )
    fig_ts.tight_layout()
    fig_ts.savefig(time_path, dpi=150)
    plt.close(fig_ts)
    return path, time_path


def main() -> None:
    args = parse_args()

    if args.list:
        for case_id in list_cases(args.trajectory_points):
            print(case_id)
        return

    if args.max_cases is not None:
        case_ids = list_cases(args.trajectory_points)[: args.max_cases]
    elif args.case_id is not None:
        case_ids = [args.case_id]
    else:
        available = list_cases(args.trajectory_points)
        raise SystemExit(
            "No --case-id provided. Use --list to see options, e.g.\n  "
            f"python -m Calibration.run_multi_agent_calibration --case-id {available[0]!r}"
        )

    boundary_options = _boundary_options(args)
    batch_results = []
    for i, case_id in enumerate(case_ids, start=1):
        print("")
        print(f"=== Case {i}/{len(case_ids)}: {case_id} ===")
        result, scene, best_params, bo, space = calibrate_ego_case(
            case_id,
            trajectory_points_path=args.trajectory_points,
            space=ParameterSpace(defs=EGO_PARAMETER_DEFS),
            radius_m=args.radius,
            n_initial=args.n_initial,
            n_iterations=args.n_iterations,
            solver_iterations=args.solver_iterations,
            error_samples=args.error_samples,
            seed=args.seed,
            boundary_options=boundary_options,
            flip_y=args.flip_y,
        )

        result_path = save_result(result, args.output_dir, bo=bo, space=space)
        batch_results.append(result)
        print("")
        print(f"Calibrated ego case: {result.case_id} ({result.movement_name})")
        print(f"Neighbors present: {result.n_neighbors}")
        print(f"Best RMS trajectory error: {result.best_score:.3f} m over {result.n_evaluations} evaluations")
        print(f"Best parameters: {result.best_params}")
        print(f"Saved result: {result_path}")
        suffix = "_flip_y" if result.y_flipped else ""
        safe_case = safe_case_filename(result.case_id)
        print(f"Saved evaluations CSV: {args.output_dir / f'multi_agent_calibration_{safe_case}{suffix}_evaluations.csv'}")
        print(f"Saved parameter distributions: {args.output_dir / f'multi_agent_calibration_{safe_case}{suffix}_param_distributions.png'}")

        if args.plot:
            plot_paths = _save_plot(scene, best_params, args.output_dir, boundary_options=boundary_options)
            if plot_paths is not None:
                traj_path, time_path = plot_paths
                print(f"Saved plot: {traj_path}")
                print(f"Saved time-series plot: {time_path}")

    if len(batch_results) > 1:
        summary_suffix = "_flip_y" if args.flip_y else ""
        summary_path = save_batch_summary_csv(
            batch_results,
            args.output_dir / f"multi_agent_calibration_summary{summary_suffix}.csv",
        )
        print("")
        print(f"Saved batch summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
