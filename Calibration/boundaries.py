"""Inspect and plot the boundaries of the ILQR simulation scene.

Two important facts this module makes explicit:

1. The *calibration* runner (`ilqr_interface.solve_single_agent`) has **no**
   boundaries: static obstacles and lane-keeping are intentionally omitted so an
   isolated observed case is fit only by the agent's own behavioral weights.
2. The full ILQR experiment (`main.py`) **does** have boundaries: a set of
   `STATIC_OBSTACLES` repulsion points describing a synthetic intersection.

It also provides an overlay to visualize what goes wrong when the simulation
geometry and the observed data are not on the same scale / not registered.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ILQR_ROOT = PROJECT_ROOT / "ILQR_Multi_Agent_IntersectionAnalytical"
if str(ILQR_ROOT) not in sys.path:
    sys.path.insert(0, str(ILQR_ROOT))


def load_scene():
    """Import the full ILQR scene and return its boundary geometry.

    Importing ``main`` only runs module-level setup (no solve, which is guarded
    by ``__main__``); we still silence its import-time output for cleanliness.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            import main  # type: ignore

    return {
        "static_obstacles": np.asarray(main.STATIC_OBSTACLES, dtype=float),
        "intersection": {
            "x_min": main.INTERSECTION_X_MIN,
            "x_max": main.INTERSECTION_X_MAX,
            "y_min": main.INTERSECTION_Y_MIN,
            "y_max": main.INTERSECTION_Y_MAX,
            "ns_divider_x": main.NS_DIVIDER_X,
            "ew_lane_y": main.EW_LANE_Y,
        },
        "arms": {
            "west_x_min": main.WEST_ARM_X_MIN,
            "east_x_max": main.EAST_ARM_X_MAX,
            "south_y_min": main.ROAD_SOUTH_Y_MIN,
            "north_y_max": main.ROAD_NORTH_Y_MAX,
        },
    }


def plot_simulation_boundaries(output_path: Path) -> Path:
    """Plot the synthetic static-obstacle boundaries of the full ILQR scene."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    scene = load_scene()
    obstacles = scene["static_obstacles"]
    box = scene["intersection"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        obstacles[:, 0],
        obstacles[:, 1],
        s=40,
        color="#383838",
        marker="s",
        label="static obstacle points (boundaries)",
    )
    ax.add_patch(
        mpatches.Rectangle(
            (box["x_min"], box["y_min"]),
            box["x_max"] - box["x_min"],
            box["y_max"] - box["y_min"],
            fill=False,
            edgecolor="#1f77b4",
            lw=1.5,
            linestyle="--",
            label="intersection box",
        )
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("ILQR simulation boundaries (full main.py scene)")
    ax.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_scale_overlay(case_id: str, output_path: Path) -> Path:
    """Overlay an observed case onto the synthetic scene to show scale mismatch.

    The observed path is translated so its first point sits at the synthetic
    intersection center; this highlights that the hand-placed boundaries do not
    correspond to where the real vehicle actually drove.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .observed_cases import load_case

    scene = load_scene()
    obstacles = scene["static_obstacles"]
    box = scene["intersection"]

    case = load_case(case_id)
    center = np.array(
        [
            0.5 * (box["x_min"] + box["x_max"]),
            0.5 * (box["y_min"] + box["y_max"]),
        ]
    )
    observed = case.path_local + center  # place observed entry at scene center

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(
        obstacles[:, 0], obstacles[:, 1], s=30, color="#888888", marker="s",
        label="simulation boundaries",
    )
    ax.plot(
        observed[:, 0], observed[:, 1], "o-", ms=3, lw=1.8, color="#d62728",
        label=f"observed case (real scale): {case.movement_name}",
    )
    ax.scatter([observed[0, 0]], [observed[0, 1]], color="green", zorder=5, label="observed start")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        "Scale / registration check: observed turn vs synthetic boundaries\n"
        "(real turn extent does not match hand-placed intersection geometry)"
    )
    ax.legend(loc="best", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "Calibration" / "outputs" / "boundaries"
    b = plot_simulation_boundaries(out_dir / "simulation_boundaries.png")
    print(f"Saved: {b}")
    s = plot_scale_overlay("719_1_2_5_I_WB_->_23_SB_middle", out_dir / "scale_overlay.png")
    print(f"Saved: {s}")
