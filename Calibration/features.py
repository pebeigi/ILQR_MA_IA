from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TRAJECTORY_COLUMNS = {
    "case_id",
    "time",
    "xloc_kf",
    "yloc_kf",
    "movement_name",
}

DEFAULT_FEATURE_COLUMNS = (
    "duration_s",
    "path_length_m",
    "displacement_m",
    "mean_speed_mps",
    "max_speed_mps",
    "start_x_m",
    "start_y_m",
    "end_x_m",
    "end_y_m",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Settings used when converting trajectory points into calibration targets."""

    min_points_per_case: int = 3
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)


def load_trajectory_points(path: Path) -> pd.DataFrame:
    """Load prepared trajectory points and validate the columns used here."""

    df = pd.read_csv(path)
    missing = REQUIRED_TRAJECTORY_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_list}")

    df = df.sort_values(["case_id", "time"]).reset_index(drop=True)
    return df


def _case_features(case_id: str, group: pd.DataFrame) -> dict[str, float | str]:
    group = group.sort_values("time")
    x = group["xloc_kf"].to_numpy(dtype=float)
    y = group["yloc_kf"].to_numpy(dtype=float)
    t = group["time"].to_numpy(dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    valid_dt = dt > 0.0
    segment_lengths = np.hypot(dx, dy)
    speeds = np.divide(
        segment_lengths,
        dt,
        out=np.full_like(segment_lengths, np.nan, dtype=float),
        where=valid_dt,
    )

    duration_s = float(t[-1] - t[0])
    path_length_m = float(np.nansum(segment_lengths))
    displacement_m = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))

    return {
        "case_id": case_id,
        "vehicle_id": group["id"].iloc[0] if "id" in group.columns else "",
        "intersection_id": group["intersection_id"].iloc[0] if "intersection_id" in group.columns else "",
        "intersection_name": group["intersection_name"].iloc[0] if "intersection_name" in group.columns else "",
        "movement_name": group["movement_name"].iloc[0],
        "n_points": int(len(group)),
        "start_time_s": float(t[0]),
        "end_time_s": float(t[-1]),
        "duration_s": duration_s,
        "path_length_m": path_length_m,
        "displacement_m": displacement_m,
        "mean_speed_mps": float(np.nanmean(speeds)) if speeds.size else 0.0,
        "max_speed_mps": float(np.nanmax(speeds)) if speeds.size else 0.0,
        "start_x_m": float(x[0]),
        "start_y_m": float(y[0]),
        "end_x_m": float(x[-1]),
        "end_y_m": float(y[-1]),
    }


def compute_case_features(
    trajectory_points: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute one feature row per observed left-turn case."""

    config = config or FeatureConfig()
    rows: list[dict[str, float | str]] = []

    for case_id, group in trajectory_points.groupby("case_id", sort=False):
        if len(group) < config.min_points_per_case:
            continue
        rows.append(_case_features(str(case_id), group))

    return pd.DataFrame(rows)


def aggregate_movement_targets(
    case_features: pd.DataFrame,
    config: FeatureConfig | None = None,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Aggregate observed features by movement for calibration objectives."""

    config = config or FeatureConfig()
    rows: list[dict[str, float | str]] = []

    for movement_name, group in case_features.groupby("movement_name", sort=True):
        row: dict[str, float | str] = {
            "movement_name": movement_name,
            "case_count": int(len(group)),
        }
        if "intersection_id" in group.columns:
            row["intersection_id"] = group["intersection_id"].iloc[0]
        if "intersection_name" in group.columns:
            row["intersection_name"] = group["intersection_name"].iloc[0]

        for column in feature_columns:
            values = group[column].dropna().astype(float)
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
            for q in config.quantiles:
                row[f"{column}_q{int(q * 100):02d}"] = float(values.quantile(q))

        rows.append(row)

    return pd.DataFrame(rows)


def write_feature_outputs(
    trajectory_points_path: Path,
    output_dir: Path,
    config: FeatureConfig | None = None,
) -> tuple[Path, Path]:
    """Load observed data and write case-level and movement-level feature files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    points = load_trajectory_points(trajectory_points_path)
    case_features = compute_case_features(points, config=config)
    movement_targets = aggregate_movement_targets(case_features, config=config)

    case_features_path = output_dir / "observed_case_features.csv"
    movement_targets_path = output_dir / "observed_movement_targets.csv"
    case_features.to_csv(case_features_path, index=False)
    movement_targets.to_csv(movement_targets_path, index=False)
    return case_features_path, movement_targets_path

