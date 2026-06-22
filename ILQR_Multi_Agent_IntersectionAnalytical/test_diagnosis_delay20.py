"""Single-case diagnosis for the delayed-turn warm start.

This runs only the switch-2 "turns_delayed_20" case and reports whether
left_turn_northbound and follower_left_turn merely get close or physically
overlap under the 4 m x 2 m vehicle footprint used by main.py.

Examples:
    .venv/bin/python test_diagnosis_delay20.py --use-existing-csv
    .venv/bin/python test_diagnosis_delay20.py --max-iterations 500
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "outputs" / "diagnosis_delay20"
CACHE_DIR = OUTPUT_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import main as experiment
import sensitivity_analysis as sensitivity
from ilq.ilq_solver import ILQSolver


DELAY_STEPS = 20
AGENT_A = "left_turn_northbound"
AGENT_B = "follower_left_turn"
EXISTING_CSV = (
    Path(__file__).parent
    / "outputs"
    / "sensitivity_analysis"
    / "switch_2_feedback"
    / "guess_turns_delayed_20.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run/diagnose only the delayed-20 left-turn platoon case."
    )
    parser.add_argument("--delay-steps", type=int, default=DELAY_STEPS)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--min-iterations", type=int, default=20)
    parser.add_argument("--convergence-tol", type=float, default=experiment.CONVERGENCE_TOL)
    parser.add_argument("--alpha-scaling", type=float, default=experiment.ALPHA_SCALING)
    parser.add_argument(
        "--no-alpha-line-search",
        action="store_true",
        help="Use the fixed --alpha-scaling value instead of backtracking line search.",
    )
    parser.add_argument("--alpha-line-search-min", type=float, default=experiment.ALPHA_LINE_SEARCH_MIN)
    parser.add_argument("--alpha-line-search-shrink", type=float, default=experiment.ALPHA_LINE_SEARCH_SHRINK)
    parser.add_argument(
        "--alpha-line-search-max-growth",
        type=float,
        default=experiment.ALPHA_LINE_SEARCH_MAX_GROWTH,
    )
    parser.add_argument(
        "--alpha-line-search-start-iteration",
        type=int,
        default=experiment.ALPHA_LINE_SEARCH_START_ITERATION,
    )
    parser.add_argument("--lq-solver-type", choices=["feedback", "open_loop"], default=experiment.LQ_SOLVER_TYPE)
    parser.add_argument(
        "--use-existing-csv",
        action="store_true",
        help="Only analyze the saved sensitivity CSV instead of solving again.",
    )
    return parser.parse_args()


def vehicle_polygon(x: float, y: float, heading: float) -> np.ndarray:
    return experiment.vehicle_corners(np.array([x, y], dtype=float), heading)


def polygon_axes(poly: np.ndarray) -> list[np.ndarray]:
    axes = []
    for i in range(len(poly)):
        edge = poly[(i + 1) % len(poly)] - poly[i]
        axis = np.array([-edge[1], edge[0]], dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-12:
            axes.append(axis / norm)
    return axes


def projection(poly: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = poly @ axis
    return float(values.min()), float(values.max())


def polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    for axis in polygon_axes(poly_a) + polygon_axes(poly_b):
        a_min, a_max = projection(poly_a, axis)
        b_min, b_max = projection(poly_b, axis)
        if a_max < b_min or b_max < a_min:
            return False
    return True


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(segment @ segment)
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    t = max(0.0, min(1.0, float((point - start) @ segment) / denom))
    return float(np.linalg.norm(point - (start + t * segment)))


def polygon_gap(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    if polygons_overlap(poly_a, poly_b):
        return 0.0
    best = float("inf")
    for point in poly_a:
        for i in range(len(poly_b)):
            best = min(best, point_segment_distance(point, poly_b[i], poly_b[(i + 1) % len(poly_b)]))
    for point in poly_b:
        for i in range(len(poly_a)):
            best = min(best, point_segment_distance(point, poly_a[i], poly_a[(i + 1) % len(poly_a)]))
    return best


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save_convergence_csv(history, iteration_delta_xs, alpha_history, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "total_cost", "max_delta_x", "accepted_alpha"])
        for idx, costs in enumerate(history):
            dxs = iteration_delta_xs[idx] if idx < len(iteration_delta_xs) else []
            alpha = alpha_history[idx] if idx < len(alpha_history) else ""
            writer.writerow(
                [
                    idx + 1,
                    float(sum(costs)),
                    float(max(dxs)) if dxs else 0.0,
                    alpha,
                ]
            )
    return path


def state_from_row(row: dict[str, str], agents: list[experiment.AgentSpec]) -> np.ndarray:
    values = []
    for agent in agents:
        prefix = agent.name
        values.extend(
            [
                float(row[f"{prefix}_x_m"]),
                float(row[f"{prefix}_y_m"]),
                float(row[f"{prefix}_speed_mps"]),
                float(row[f"{prefix}_heading_rad"]),
            ]
        )
    return np.asarray(values, dtype=float)


def controls_from_row(row: dict[str, str], agents: list[experiment.AgentSpec]) -> list[np.ndarray]:
    controls = []
    for agent in agents:
        prefix = agent.name
        controls.append(
            np.asarray(
                [
                    float(row[f"{prefix}_kappa_radpm"]),
                    float(row[f"{prefix}_a_mps2"]),
                ],
                dtype=float,
            )
        )
    return controls


def pair_record(row: dict[str, str]) -> dict[str, object]:
    ax = float(row[f"{AGENT_A}_x_m"])
    ay = float(row[f"{AGENT_A}_y_m"])
    ah = float(row[f"{AGENT_A}_heading_rad"])
    av = float(row[f"{AGENT_A}_speed_mps"])
    bx = float(row[f"{AGENT_B}_x_m"])
    by = float(row[f"{AGENT_B}_y_m"])
    bh = float(row[f"{AGENT_B}_heading_rad"])
    bv = float(row[f"{AGENT_B}_speed_mps"])

    poly_a = vehicle_polygon(ax, ay, ah)
    poly_b = vehicle_polygon(bx, by, bh)
    center_distance = math.hypot(ax - bx, ay - by)
    overlap = polygons_overlap(poly_a, poly_b)
    return {
        "step": int(row["step"]),
        "time_s": float(row["time_s"]),
        "center_distance": center_distance,
        "body_gap": polygon_gap(poly_a, poly_b),
        "overlap": overlap,
        "a_active": row[f"{AGENT_A}_active"],
        "b_active": row[f"{AGENT_B}_active"],
        "a_x": ax,
        "a_y": ay,
        "a_speed": av,
        "a_heading": ah,
        "b_x": bx,
        "b_y": by,
        "b_speed": bv,
        "b_heading": bh,
    }


def overlap_runs(records: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    overlaps = [record for record in records if record["overlap"]]
    if not overlaps:
        return []
    runs = []
    start = overlaps[0]
    previous = overlaps[0]
    for current in overlaps[1:]:
        if int(current["step"]) == int(previous["step"]) + 1:
            previous = current
        else:
            runs.append((start, previous))
            start = previous = current
    runs.append((start, previous))
    return runs


def relevant_pair_terms(game, owner_index: int, other_index: int, x: np.ndarray, us: list[np.ndarray], k: int):
    rows = []
    for term in game.player_costs[owner_index].cost_terms:
        term_name = type(term).__name__
        references_other = getattr(term, "other_agent_index", None) == other_index
        references_leader = getattr(term, "leader_index", None) == other_index
        if not references_other and not references_leader:
            continue
        rows.append(
            {
                "term": term_name,
                "weight": getattr(term, "weight", None),
                "value": float(term.evaluate(x, us, k)),
            }
        )
    return rows


def print_record(label: str, record: dict[str, object]) -> None:
    print(f"\n{label}")
    print(
        f"  step={record['step']}  t={record['time_s']:.1f}s  "
        f"center_distance={record['center_distance']:.4f} m  "
        f"body_gap={record['body_gap']:.4f} m  overlap={record['overlap']}"
    )
    print(
        f"  {AGENT_A}: ({record['a_x']:.3f}, {record['a_y']:.3f}), "
        f"v={record['a_speed']:.3f}, heading={record['a_heading']:.3f}, "
        f"active={record['a_active']}"
    )
    print(
        f"  {AGENT_B}: ({record['b_x']:.3f}, {record['b_y']:.3f}), "
        f"v={record['b_speed']:.3f}, heading={record['b_heading']:.3f}, "
        f"active={record['b_active']}"
    )


def analyze_csv(csv_path: Path, agents: list[experiment.AgentSpec], game) -> None:
    rows = load_csv_rows(csv_path)
    records = [pair_record(row) for row in rows]
    closest_center = min(records, key=lambda record: record["center_distance"])
    closest_body = min(records, key=lambda record: record["body_gap"])
    runs = overlap_runs(records)

    print(f"\nAnalyzing: {csv_path.resolve()}")
    print(f"Pair: {AGENT_A} vs {AGENT_B}")
    print("Vehicle footprint: 4.0 m x 2.0 m oriented rectangle")
    print_record("Closest center distance", closest_center)
    print_record("Smallest body gap", closest_body)

    if runs:
        print("\nBody overlap windows:")
        for start, end in runs:
            print(
                f"  steps {start['step']}..{end['step']}  "
                f"t={start['time_s']:.1f}s..{end['time_s']:.1f}s"
            )
    else:
        print("\nNo body overlap detected.")

    idx_a = next(i for i, agent in enumerate(agents) if agent.name == AGENT_A)
    idx_b = next(i for i, agent in enumerate(agents) if agent.name == AGENT_B)
    closest_row = rows[int(closest_center["step"])]
    x = state_from_row(closest_row, agents)
    us = controls_from_row(closest_row, agents)
    k = int(closest_center["step"])

    print("\nPair-related cost terms at closest center distance:")
    for owner_idx, other_idx, owner_name, other_name in [
        (idx_a, idx_b, AGENT_A, AGENT_B),
        (idx_b, idx_a, AGENT_B, AGENT_A),
    ]:
        terms = relevant_pair_terms(game, owner_idx, other_idx, x, us, k)
        print(f"  terms in {owner_name}'s cost that reference {other_name}:")
        if not terms:
            print("    none")
            continue
        has_proximity_speed = False
        for term in terms:
            has_proximity_speed = has_proximity_speed or term["term"] == "AgentProximitySpeedCost"
            weight = "None" if term["weight"] is None else f"{float(term['weight']):.6g}"
            print(f"    {term['term']}: weight={weight}, value={term['value']:.6g}")
        if not has_proximity_speed:
            print("    no AgentProximitySpeedCost term for this pair")


def run_case(args, agents: list[experiment.AgentSpec], game) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_prefix = OUTPUT_DIR / f"turns_delayed_{args.delay_steps}"
    solver = ILQSolver(
        game,
        max_iterations=args.max_iterations,
        convergence_tol=args.convergence_tol,
        alpha_scaling=args.alpha_scaling,
        use_euler=True,
        lq_solver_type=args.lq_solver_type,
        min_iterations=args.min_iterations,
        alpha_line_search=not args.no_alpha_line_search,
        alpha_line_search_min=args.alpha_line_search_min,
        alpha_line_search_shrink=args.alpha_line_search_shrink,
        alpha_line_search_max_growth=args.alpha_line_search_max_growth,
        alpha_line_search_start_iteration=args.alpha_line_search_start_iteration,
    )

    guess_agents = sensitivity.with_turning_warm_start_delay(agents, args.delay_steps)
    initial_operating_point = experiment.make_initial_nominal_trajectory(game, guess_agents)
    initial_xs = initial_operating_point[0]
    solver.current_operating_point = initial_operating_point

    print(
        f"Running one diagnosis case: turns_delayed_{args.delay_steps}, "
        f"solver={args.lq_solver_type}, max_iterations={args.max_iterations}"
    )
    start = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - start
    print(f"Finished solve in {elapsed:.1f}s with {len(result['history'])} iterations.")

    saved_paths = {}
    convergence_csv = save_convergence_csv(
        result["history"],
        result["iteration_delta_xs"],
        result.get("alpha_history", []),
        Path(f"{case_prefix}_convergence.csv"),
    )
    saved_paths["convergence_csv"] = str(convergence_csv.resolve())
    saved_paths.update(
        sensitivity.save_pillow_iteration_plots(
            result["history"],
            result["iteration_trajectories"],
            result["iteration_delta_xs"],
            agents,
            prefix=case_prefix,
        )
    )
    trajectory_per_agent_path = sensitivity._save_pillow_per_agent_evolution(
        result["iteration_trajectories"],
        initial_xs,
        agents,
        Path(f"{case_prefix}_trajectory_per_agent_per_iteration.png"),
    )
    if trajectory_per_agent_path is not None:
        saved_paths["trajectory_per_agent_plot"] = str(Path(trajectory_per_agent_path).resolve())

    trajectory_evolution_gif = sensitivity._save_pillow_optimization_evolution_animation(
        result["iteration_trajectories"],
        initial_xs,
        agents,
        OUTPUT_DIR / f"turns_delayed_{args.delay_steps}_trajectory_evolution.gif",
        fps=3,
    )
    if trajectory_evolution_gif is not None:
        saved_paths["trajectory_evolution_gif"] = str(Path(trajectory_evolution_gif).resolve())

    xs, us = experiment.freeze_arrived_agents(result["xs"], result["us"], agents)
    csv_path = Path(f"{case_prefix}.csv")
    experiment.save_multi_agent_csv(xs, us, agents, path=csv_path)
    plot_path = sensitivity._save_pillow_final_trajectory(
        xs,
        agents,
        Path(f"{case_prefix}.png"),
    )
    animation_path = sensitivity._save_pillow_animation(
        xs,
        us,
        agents,
        OUTPUT_DIR / f"turns_delayed_{args.delay_steps}_animation.gif",
        fps=10,
    )

    print(f"Saved CSV: {csv_path.resolve()}")
    if plot_path is not None:
        print(f"Saved plot: {Path(plot_path).resolve()}")
    if animation_path is not None:
        print(f"Saved animation GIF: {Path(animation_path).resolve()}")
    for label, path in saved_paths.items():
        print(f"Saved {label}: {path}")
    return csv_path


def main():
    args = parse_args()
    agents = sensitivity.copy_agents(experiment.AGENTS)
    game = experiment.build_multi_agent_game(agents)

    if args.use_existing_csv:
        csv_path = EXISTING_CSV
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
    else:
        csv_path = run_case(args, agents, game)

    analyze_csv(csv_path, agents, game)


if __name__ == "__main__":
    main()
