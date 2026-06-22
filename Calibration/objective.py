from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights and scaling used to compare simulated and observed movement targets."""

    feature_weights: dict[str, float]
    min_scale: float = 1e-6


DEFAULT_OBJECTIVE_CONFIG = ObjectiveConfig(
    feature_weights={
        "duration_s_mean": 1.0,
        "path_length_m_mean": 1.0,
        "mean_speed_mps_mean": 1.0,
        "max_speed_mps_mean": 0.5,
        "end_x_m_mean": 0.5,
        "end_y_m_mean": 0.5,
    }
)


def score_movement_targets(
    observed_targets: pd.DataFrame,
    simulated_targets: pd.DataFrame,
    config: ObjectiveConfig = DEFAULT_OBJECTIVE_CONFIG,
) -> dict[str, float | dict[str, float]]:
    """Return a normalized weighted squared-error score.

    Both inputs should be movement-level feature tables, usually produced by
    ``aggregate_movement_targets``. Movement names are used as the join key.
    """

    if "movement_name" not in observed_targets.columns:
        raise ValueError("observed_targets must include movement_name")
    if "movement_name" not in simulated_targets.columns:
        raise ValueError("simulated_targets must include movement_name")

    merged = observed_targets.merge(
        simulated_targets,
        on="movement_name",
        how="inner",
        suffixes=("_observed", "_simulated"),
    )
    if merged.empty:
        raise ValueError("no overlapping movement_name values between observed and simulated targets")

    feature_scores: dict[str, float] = {}
    weighted_errors = []
    for feature, weight in config.feature_weights.items():
        observed_col = f"{feature}_observed"
        simulated_col = f"{feature}_simulated"
        if observed_col not in merged.columns or simulated_col not in merged.columns:
            continue

        observed = merged[observed_col].to_numpy(dtype=float)
        simulated = merged[simulated_col].to_numpy(dtype=float)
        scale = np.maximum(np.abs(observed), config.min_scale)
        normalized_error = (simulated - observed) / scale
        mse = float(np.nanmean(normalized_error**2))
        feature_scores[feature] = mse
        weighted_errors.append(float(weight) * mse)

    if not weighted_errors:
        available = ", ".join(sorted(merged.columns))
        raise ValueError(f"none of the configured objective features were found. Available columns: {available}")

    return {
        "score": float(np.sum(weighted_errors)),
        "matched_movements": int(len(merged)),
        "feature_scores": feature_scores,
    }

