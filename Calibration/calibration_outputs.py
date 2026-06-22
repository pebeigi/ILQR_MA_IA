"""CSV exports and parameter-distribution plots for calibration runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bayes_opt import BOResult
from .parameters import ParameterSpace


def evaluations_dataframe(
    bo: BOResult,
    space: ParameterSpace,
    base: Any | None = None,
) -> pd.DataFrame:
    """One row per Bayesian-optimization evaluation with decoded parameters."""
    best_idx = int(np.argmin(bo.ys))
    rows: list[dict] = []
    for i, (x, y) in enumerate(zip(bo.xs, bo.ys)):
        params = asdict(space.to_params(x, base=base))
        rows.append(
            {
                "evaluation_id": i,
                "score": float(y),
                "is_best": i == best_idx,
                **params,
            }
        )
    return pd.DataFrame(rows)


def save_evaluations_csv(
    bo: BOResult,
    space: ParameterSpace,
    path: Path,
    *,
    base: Any | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluations_dataframe(bo, space, base=base).to_csv(path, index=False)
    return path


def plot_parameter_distributions(
    bo: BOResult,
    space: ParameterSpace,
    output_path: Path,
    *,
    base: Any | None = None,
    title: str = "",
) -> Path | None:
    """Histogram of sampled values for each calibrated parameter."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    names = space.names
    n_params = len(names)
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    best_params = asdict(space.to_params(bo.best_x, base=base))
    df = evaluations_dataframe(bo, space, base=base)

    for idx, name in enumerate(names):
        ax = axes.flat[idx]
        values = df[name].to_numpy(dtype=float)
        ax.hist(values, bins=min(20, max(5, len(values) // 3)), color="#4c72b0", alpha=0.85, edgecolor="white")
        ax.axvline(best_params[name], color="#d62728", lw=2.0, label="best")
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="best", fontsize=7)

    for ax in axes.flat[n_params:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def result_summary_row(result: Any) -> dict:
    """Flatten a calibration result dataclass into one summary-table row."""
    if not is_dataclass(result):
        raise TypeError("result must be a dataclass instance")
    data = asdict(result)
    row: dict = {}
    for key, value in data.items():
        if key == "best_params" and isinstance(value, dict):
            row.update(value)
        elif key == "error_breakdown" and isinstance(value, dict):
            for ek, ev in value.items():
                row[f"error_{ek}"] = ev
        elif key == "history":
            continue
        else:
            row[key] = value
    return row


def save_batch_summary_csv(results: list[Any], path: Path) -> Path:
    """Write one row per calibrated case (best params + scores)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result_summary_row(r) for r in results]).to_csv(path, index=False)
    return path


def save_calibration_artifacts(
    result: Any,
    bo: BOResult,
    space: ParameterSpace,
    output_dir: Path,
    *,
    prefix: str,
    suffix: str = "",
    base: Any | None = None,
    plot_title: str = "",
) -> dict[str, Path]:
    """Save JSON (via caller), per-case evaluation CSV, and parameter distributions."""
    stem = f"{prefix}{suffix}"
    eval_path = save_evaluations_csv(
        bo, space, output_dir / f"{stem}_evaluations.csv", base=base
    )
    dist_path = plot_parameter_distributions(
        bo,
        space,
        output_dir / f"{stem}_param_distributions.png",
        base=base,
        title=plot_title or getattr(result, "case_id", ""),
    )
    artifacts = {"evaluations_csv": eval_path}
    if dist_path is not None:
        artifacts["param_distributions_png"] = dist_path
    return artifacts
