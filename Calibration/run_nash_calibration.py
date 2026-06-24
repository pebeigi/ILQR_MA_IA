"""CLI: joint Nash-game calibration of per-agent ILQR weights.

Every co-present vehicle is a real ILQR player; all are optimized together as a
Nash game and calibrated jointly (per-agent weights, all-agents objective).

Example:
    python -m Calibration.run_nash_calibration \
        --case-id "719_1_2_5_I_WB_->_23_SB_middle" --flip-y --use-boundaries --plot
    python -m Calibration.run_nash_calibration --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .nash_calibrate import (
    NASH_PARAMETER_DEFS,
    NashCalibrationResult,
    calibrate_nash_case,
    save_result,
)
from .nash_cases import GameScene
from .nash_interface import NashAgentParameters, solve_nash
from .observed_cases import DEFAULT_TRAJECTORY_POINTS, list_cases
from .paths import safe_case_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Calibration" / "outputs" / "nash_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint Nash-game (all agents) ILQR calibration.")
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None, help="Calibrate the first N case_ids.")
    parser.add_argument("--list", action="store_true", help="List available case_ids and exit.")
    parser.add_argument("--trajectory-points", type=Path, default=DEFAULT_TRAJECTORY_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--radius", type=float, default=35.0, help="Neighbor proximity threshold (m).")
    parser.add_argument("--max-agents", type=int, default=6, help="Max game players (ego + neighbors).")
    parser.add_argument("--min-presence", type=float, default=0.5, help="Min fraction of horizon an agent must be present.")
    parser.add_argument("--min-travel", type=float, default=5.0, help="Min path length (m) for a neighbor to join the game; shorter tracks are ignored (plotted only).")
    parser.add_argument("--rounds", type=int, default=2, help="Coordinate-descent sweeps over agents.")
    parser.add_argument("--n-initial", type=int, default=6, help="Random evals per agent BO subproblem.")
    parser.add_argument("--n-iterations", type=int, default=15, help="BO iterations per agent subproblem.")
    parser.add_argument("--solver-iterations", type=int, default=50, help="ILQR iterations per game solve.")
    parser.add_argument("--error-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip-y", action="store_true", help="Use y := -y for data and curbs.")
    parser.add_argument("--plot", action="store_true", help="Save trajectory + parameter-distribution plots.")
    parser.add_argument("--use-boundaries", action="store_true", help="Add real street curbs to the game.")
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


def _curbs(scene: GameScene, boundary_options: dict | None):
    if not boundary_options or not boundary_options.get("use_boundaries"):
        return None
    from .real_boundaries import local_boundary_points

    return local_boundary_points(
        scene.ego_origin,
        spacing_m=boundary_options.get("spacing_m", 2.0),
        radius_m=boundary_options.get("radius_m", 40.0),
        flip_y=scene.y_flipped,
    )


def _save_plots(
    scene: GameScene,
    params_list: list[NashAgentParameters],
    result: NashCalibrationResult,
    output_dir: Path,
    boundary_options: dict | None,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Plot skipped (matplotlib unavailable): {exc}")
        return []

    curbs = _curbs(scene, boundary_options)
    sims = solve_nash(
        scene,
        params_list,
        boundary_obstacles=curbs,
        boundary_weight=(boundary_options or {}).get("weight", 1000.0),
        max_iterations=120,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case = safe_case_filename(result.case_id)
    suffix = "_flip_y" if scene.y_flipped else ""
    stem = f"nash_calibration_{safe_case}{suffix}"
    paths: list[Path] = []

    # 1) Trajectory plot: observed (solid) vs simulated (dashed) per agent.
    fig, ax = plt.subplots(figsize=(8, 8))
    if curbs is not None and len(curbs):
        ax.scatter(curbs[:, 0], curbs[:, 1], s=6, color="#cccccc", label="street curbs")

    cmap = plt.get_cmap("tab10")
    for i, (track, res) in enumerate(zip(scene.agents, sims)):
        color = cmap(i % 10)
        obs = track.observed_path
        sim = res.states[track.entry_step : track.exit_step + 1, :2]
        label_obs = "observed" if i == 0 else None
        label_sim = "simulated (Nash)" if i == 0 else None
        ax.plot(obs[:, 0], obs[:, 1], "-", lw=1.6, color=color, label=label_obs)
        ax.plot(sim[:, 0], sim[:, 1], "--", lw=1.8, color=color, label=label_sim)
        ax.scatter([obs[0, 0]], [obs[0, 1]], color=color, s=30, zorder=5)
        tag = "ego" if track.is_ego else track.name
        ax.annotate(tag, (obs[0, 0], obs[0, 1]), fontsize=7, color=color)

    # Stopped / ignored neighbors: observed only, gray, labeled.
    for j, track in enumerate(scene.ignored_agents):
        obs = track.observed_path
        if obs.shape[0] < 1:
            continue
        label_ign = "ignored (stopped)" if j == 0 else None
        ax.plot(obs[:, 0], obs[:, 1], "-", lw=1.2, color="#888888", alpha=0.75, label=label_ign)
        ax.scatter([obs[0, 0]], [obs[0, 1]], color="#888888", s=24, zorder=4, alpha=0.85)
        ax.annotate(
            f"{track.name}\n(ignored, {track.path_length_m:.1f} m)",
            (obs[0, 0], obs[0, 1]),
            fontsize=6,
            color="#666666",
            alpha=0.9,
        )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m] (ego local frame)")
    ax.set_ylabel("y [m] (ego local frame)")
    ignored_note = f", {scene.n_ignored} ignored" if scene.n_ignored else ""
    ax.set_title(
        f"Nash-game calibration: {result.case_id}\n"
        f"{scene.n_agents} active agents{ignored_note}, mean error {result.mean_error:.2f} m"
        f"{'  [y flipped]' if scene.y_flipped else ''}"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    traj_path = output_dir / f"{stem}.png"
    fig.savefig(traj_path, dpi=150)
    plt.close(fig)
    paths.append(traj_path)

    # 2) Distribution of each calibrated parameter across agents.
    names = [d.name for d in NASH_PARAMETER_DEFS]
    n_cols = min(4, len(names))
    n_rows = int(np.ceil(len(names) / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for idx, name in enumerate(names):
        ax = axes[idx]
        values = np.array([a.params[name] for a in result.agents], dtype=float)
        ax.hist(values, bins=min(10, max(3, scene.n_agents)), color="#4c72b0",
                alpha=0.85, edgecolor="white")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig2.suptitle(f"Per-agent parameter distribution: {result.case_id} (N={scene.n_agents})", fontsize=11)
    fig2.tight_layout()
    dist_path = output_dir / f"{stem}_param_distributions.png"
    fig2.savefig(dist_path, dpi=150)
    plt.close(fig2)
    paths.append(dist_path)
    return paths


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
            f"python -m Calibration.run_nash_calibration --case-id {available[0]!r}"
        )

    boundary_options = _boundary_options(args)
    for n, case_id in enumerate(case_ids, start=1):
        print("")
        print(f"=== Case {n}/{len(case_ids)}: {case_id} ===")
        result, scene, best_params = calibrate_nash_case(
            case_id,
            trajectory_points_path=args.trajectory_points,
            radius_m=args.radius,
            max_agents=args.max_agents,
            min_presence_frac=args.min_presence,
            min_travel_m=args.min_travel,
            rounds=args.rounds,
            n_initial=args.n_initial,
            n_iterations=args.n_iterations,
            solver_iterations=args.solver_iterations,
            error_samples=args.error_samples,
            seed=args.seed,
            boundary_options=boundary_options,
            flip_y=args.flip_y,
        )
        result_path = save_result(result, args.output_dir)
        print("")
        print(f"Calibrated case: {result.case_id}")
        print(
            f"Active agents: {result.n_agents}  |  ignored: {result.n_ignored} "
            f"(travel < {result.min_travel_m:.1f} m)  |  "
            f"mean trajectory error: {result.mean_error:.3f} m over {result.n_evaluations} game solves"
        )
        for a in result.agents:
            print(f"  {a.name:>10s} {'(ego)' if a.is_ego else '     '}  error={a.error:.3f} m")
        for a in result.ignored_agents:
            print(f"  {a.name:>10s} (ignored)  path={a.path_length_m:.2f} m  — {a.reason}")
        print(f"Saved result: {result_path}")

        if args.plot:
            for p in _save_plots(scene, best_params, result, args.output_dir, boundary_options):
                print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
