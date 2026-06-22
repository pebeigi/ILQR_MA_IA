"""CLI: calibrate one agent's ILQR cost weights to an observed left-turn case.

Example:
    python -m Calibration.run_calibration --case-id 719_1_2_5_I_WB_->_23_SB_middle
    python -m Calibration.run_calibration --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .calibrate_agent import calibrate_case, save_result
from .ilqr_interface import AgentParameters, solve_single_agent
from .observed_cases import DEFAULT_TRAJECTORY_POINTS, list_cases, load_case
from .parameters import ParameterSpace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Calibration" / "outputs" / "agent_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory-level ILQR calibration via Bayesian optimization.")
    parser.add_argument("--case-id", type=str, default=None, help="Observed case_id to calibrate.")
    parser.add_argument("--list", action="store_true", help="List available case_ids and exit.")
    parser.add_argument("--trajectory-points", type=Path, default=DEFAULT_TRAJECTORY_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-initial", type=int, default=8, help="Random evaluations before the GP starts.")
    parser.add_argument("--n-iterations", type=int, default=30, help="Bayesian-optimization iterations.")
    parser.add_argument("--solver-iterations", type=int, default=60, help="ILQR iterations per evaluation.")
    parser.add_argument("--error-samples", type=int, default=100, help="Arclength resample count for the error metric.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip-y", action="store_true", help="Use y := -y for the observed case and curbs.")
    parser.add_argument("--plot", action="store_true", help="Save an observed-vs-simulated trajectory plot.")
    parser.add_argument("--use-boundaries", action="store_true", help="Add real Foggy Bottom street curbs to the simulation.")
    parser.add_argument("--boundary-radius", type=float, default=40.0, help="Keep curbs within this distance (m) of the case start.")
    parser.add_argument("--boundary-spacing", type=float, default=2.0, help="Spacing (m) between sampled curb obstacle points.")
    parser.add_argument("--boundary-weight", type=float, default=1000.0, help="Repulsion weight for curb obstacles.")
    return parser.parse_args()


def _boundary_options(args: argparse.Namespace) -> dict:
    return {
        "use_boundaries": args.use_boundaries,
        "radius_m": args.boundary_radius,
        "spacing_m": args.boundary_spacing,
        "weight": args.boundary_weight,
    }


def _save_plot(
    case_id: str,
    trajectory_points: Path,
    best_params: AgentParameters,
    output_dir: Path,
    boundary_options: dict | None = None,
    flip_y: bool = False,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Plot skipped (matplotlib unavailable): {exc}")
        return None

    case = load_case(case_id, trajectory_points, flip_y=flip_y)
    curbs = None
    if boundary_options and boundary_options.get("use_boundaries"):
        curbs = case.local_curb_points(
            spacing_m=boundary_options.get("spacing_m", 2.0),
            radius_m=boundary_options.get("radius_m", 40.0),
        )
        scenario = case.to_scenario(
            boundary_obstacles=curbs, boundary_weight=boundary_options.get("weight", 1000.0)
        )
    else:
        scenario = case.to_scenario()
    result = solve_single_agent(scenario, best_params, max_iterations=120)
    sim = result.states[:, :2]
    obs = case.path_local

    fig, ax = plt.subplots(figsize=(7, 7))
    if curbs is not None and len(curbs):
        ax.scatter(curbs[:, 0], curbs[:, 1], s=10, color="#999999", label="real street curbs")
    ax.plot(obs[:, 0], obs[:, 1], "o-", ms=3, lw=1.5, color="#1f77b4", label="observed")
    ax.plot(sim[:, 0], sim[:, 1], "-", lw=2.0, color="#d62728", label="ILQR (calibrated)")
    ax.scatter([obs[0, 0]], [obs[0, 1]], color="green", zorder=5, label="start")
    ax.scatter([obs[-1, 0]], [obs[-1, 1]], color="black", marker="x", zorder=5, label="observed end")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m] (local frame)")
    ax.set_ylabel("y [m] (local frame)")
    ax.set_title(f"Calibration: {case_id}\n{case.movement_name}{'  [y flipped]' if case.y_flipped else ''}")
    ax.legend(loc="best", fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case = case_id.replace("/", "_")
    suffix = "_flip_y" if case.y_flipped else ""
    path = output_dir / f"calibration_{safe_case}{suffix}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()

    if args.list:
        for case_id in list_cases(args.trajectory_points):
            print(case_id)
        return

    if args.case_id is None:
        available = list_cases(args.trajectory_points)
        raise SystemExit(
            "No --case-id provided. Use --list to see options, e.g.\n  "
            f"python -m Calibration.run_calibration --case-id {available[0]!r}"
        )

    boundary_options = _boundary_options(args)
    result, bo, space = calibrate_case(
        args.case_id,
        trajectory_points_path=args.trajectory_points,
        space=ParameterSpace(),
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        solver_iterations=args.solver_iterations,
        error_samples=args.error_samples,
        seed=args.seed,
        boundary_options=boundary_options,
        flip_y=args.flip_y,
    )

    result_path = save_result(result, args.output_dir, bo=bo, space=space)
    print("")
    print(f"Calibrated case: {result.case_id} ({result.movement_name})")
    print(f"Best RMS trajectory error: {result.best_score:.3f} m over {result.n_evaluations} evaluations")
    print(f"Best parameters: {result.best_params}")
    print(f"Saved result: {result_path}")
    safe_case = result.case_id.replace("/", "_")
    print(f"Saved evaluations CSV: {args.output_dir / f'calibration_{safe_case}_evaluations.csv'}")
    print(f"Saved parameter distributions: {args.output_dir / f'calibration_{safe_case}_param_distributions.png'}")

    if args.plot:
        plot_path = _save_plot(
            args.case_id,
            args.trajectory_points,
            AgentParameters(**result.best_params),
            args.output_dir,
            boundary_options=boundary_options,
            flip_y=args.flip_y,
        )
        if plot_path is not None:
            print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
