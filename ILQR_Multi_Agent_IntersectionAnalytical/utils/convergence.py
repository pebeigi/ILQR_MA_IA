"""Utilities for saving ILQ convergence histories.

The ILQ solver stores ``history`` as a list with one row per ILQ iteration and
one column per agent/player.  This module saves that history as both a CSV table
and a PNG curve so the convergence of each agent's cost is visible after a run.
"""
from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable, Sequence

import numpy as np


def _default_agent_names(num_agents: int) -> list[str]:
    return [f"agent_{i}" for i in range(num_agents)]


def _clean_history(history) -> np.ndarray:
    """Convert an ILQ history list to a 2-D float array.

    Rows are ILQ iterations and columns are agents.  A single-agent history may
    arrive as a 1-D list, so it is reshaped to ``(num_iterations, 1)``.
    """
    arr = np.asarray(history, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("history must be a 1-D or 2-D array-like object")
    return arr


def save_convergence_history(
    history,
    agent_names: Sequence[str] | None = None,
    *,
    csv_path: str | Path = "convergence_cost_history.csv",
    plot_path: str | Path = "convergence_cost_history.png",
):
    """Save per-agent cost versus ILQ iteration.

    Parameters
    ----------
    history:
        Solver history where ``history[k][i]`` is agent/player ``i``'s total
        trajectory cost at ILQ iteration ``k``.
    agent_names:
        Optional names for the agents.  If omitted, names are generated as
        ``agent_0``, ``agent_1``, ...
    csv_path, plot_path:
        Output paths for the numeric table and convergence plot.

    Returns
    -------
    tuple[Path, Path | None]
        The CSV path and the plot path.  The plot path is ``None`` if
        matplotlib is unavailable.
    """
    arr = _clean_history(history)
    num_iterations, num_agents = arr.shape

    if agent_names is None:
        names = _default_agent_names(num_agents)
    else:
        names = [str(name) for name in agent_names]
        if len(names) != num_agents:
            raise ValueError(
                f"expected {num_agents} agent names, got {len(names)}"
            )

    csv_path = Path(csv_path)
    plot_path = Path(plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", *[f"cost_{name}" for name in names], "total_cost"])
        for iteration, row in enumerate(arr, start=1):
            writer.writerow([iteration, *[float(v) for v in row], float(np.sum(row))])

    saved_plot = None
    try:
        import matplotlib.pyplot as plt

        iterations = np.arange(1, num_iterations + 1)
        fig, ax = plt.subplots()
        for i, name in enumerate(names):
            ax.plot(iterations, arr[:, i], marker="o", linewidth=1.5, label=name)
        if num_agents > 1:
            ax.plot(iterations, np.sum(arr, axis=1), marker="s", linewidth=1.5, linestyle="--", label="total")
        ax.set_xlabel("ILQ iteration")
        ax.set_ylabel("trajectory cost")
        ax.set_title("Convergence curve: cost per agent vs iteration")
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        saved_plot = plot_path
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Convergence plot skipped because matplotlib is unavailable: {exc}")

    return csv_path, saved_plot
