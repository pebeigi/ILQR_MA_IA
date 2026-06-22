"""Check that changing dynamic inter-agent repulsion weight changes trajectories.

This script imports the extendable setup from main.py, changes only the
other_agent_repulsion coefficient, and writes a comparison CSV/PNG.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import numpy as np

import main as experiment
from ilq.ilq_solver import ILQSolver


def run_case(pairwise_weight: float):
    agents = deepcopy(experiment.AGENTS)
    for agent in agents:
        agent.cost.other_agent_repulsion = float(pairwise_weight)

    game = experiment.build_multi_agent_game(agents)
    solver = ILQSolver(
        game,
        max_iterations=8,
        convergence_tol=experiment.CONVERGENCE_TOL,
        alpha_scaling=experiment.ALPHA_SCALING,
    )
    solver.current_operating_point = experiment.make_initial_nominal_trajectory(game, agents)
    result = solver.solve()
    arrivals = experiment.first_arrival_steps(result["xs"], agents)
    final_report_step = max(arrivals)

    pairwise_min = None
    if len(agents) >= 2:
        pairwise_min = min(
            float(np.linalg.norm(experiment.agent_position(result["xs"][k], 0) - experiment.agent_position(result["xs"][k], 1)))
            for k in range(final_report_step + 1)
        )

    return {
        "weight": float(pairwise_weight),
        "agents": agents,
        "result": result,
        "arrivals": arrivals,
        "final_report_step": final_report_step,
        "pairwise_min_distance": pairwise_min,
    }


def main():
    weights = [0.0, 25.0, 100.0]
    cases = [run_case(w) for w in weights]

    summary_path = Path("agent_repulsion_weight_sensitivity_summary.csv")
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["other_agent_repulsion_weight", "min_distance_agent_0_to_agent_1_m", "arrival_step_agent_0", "arrival_step_agent_1"])
        for case in cases:
            writer.writerow([
                case["weight"],
                case["pairwise_min_distance"],
                case["arrivals"][0] if len(case["arrivals"]) > 0 else None,
                case["arrivals"][1] if len(case["arrivals"]) > 1 else None,
            ])

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        for case in cases:
            xs = case["result"]["xs"]
            agents = case["agents"]
            for i, agent in enumerate(agents):
                stop = case["arrivals"][i] + 1
                xy = np.asarray([x[experiment.agent_slice(i)][:2] for x in xs[:stop]])
                plt.plot(xy[:, 0], xy[:, 1], marker="o", markersize=1.5, label=f"w={case['weight']}, {agent.name}")
        for obstacle in experiment.STATIC_OBSTACLES:
            obstacle = np.asarray(obstacle, dtype=float)
            plt.scatter([obstacle[0]], [obstacle[1]], marker="s", label="static obstacle")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Sensitivity to dynamic inter-agent repulsion weight")
        plt.legend(fontsize=6)
        plt.grid(True)
        plot_path = Path("agent_repulsion_weight_sensitivity.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        plot_path = None

    print(f"Saved summary: {summary_path.resolve()}")
    if plot_path is not None:
        print(f"Saved plot: {plot_path.resolve()}")
    for case in cases:
        print(
            f"weight={case['weight']:>6g}, "
            f"min pairwise distance={case['pairwise_min_distance']:.3f} m, "
            f"arrival steps={case['arrivals']}"
        )


if __name__ == "__main__":
    main()
