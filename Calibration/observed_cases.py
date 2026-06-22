"""Load a single observed left-turn case and turn it into an ILQR scenario.

Observed trajectories live in real intersection coordinates that differ per
movement.  We translate each case so its first point sits at the origin; the
ILQR dynamics are translation-invariant, so we can solve in this local frame
and compare simulated-vs-observed paths directly without mapping back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ilqr_interface import DEFAULT_DT, ScenarioSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_POINTS = (
    PROJECT_ROOT
    / "Data_Preparation"
    / "outputs"
    / "left_turn_movements"
    / "left_turn_intersection_zone_points.csv"
)


@dataclass(frozen=True)
class ObservedCase:
    case_id: str
    movement_name: str
    times: np.ndarray  # (N,) seconds, zero-based
    path_local: np.ndarray  # (N, 2) positions translated to start at origin
    origin: np.ndarray  # (2,) world coordinates of the first point
    initial_speed: float
    initial_heading: float
    terminal_heading: float
    vehicle_id: int | str = ""  # ego vehicle id in the raw dataset
    start_time_abs: float = 0.0  # absolute time of the first observed point
    y_flipped: bool = False

    def to_scenario(
        self,
        dt: float = DEFAULT_DT,
        min_horizon: int = 10,
        *,
        boundary_obstacles: np.ndarray | None = None,
        boundary_weight: float = 1000.0,
        boundary_epsilon: float = 0.1,
    ) -> ScenarioSpec:
        duration = float(self.times[-1] - self.times[0])
        horizon_steps = max(min_horizon, int(round(duration / dt)) + 1)
        return ScenarioSpec(
            initial_state=np.array(
                [0.0, 0.0, self.initial_speed, self.initial_heading], dtype=float
            ),
            destination=self.path_local[-1].copy(),
            terminal_heading=self.terminal_heading,
            horizon_steps=horizon_steps,
            dt=dt,
            boundary_obstacles=boundary_obstacles,
            boundary_weight=boundary_weight,
            boundary_epsilon=boundary_epsilon,
        )

    def local_curb_points(
        self,
        *,
        spacing_m: float = 2.0,
        radius_m: float | None = 40.0,
    ) -> np.ndarray:
        """Real street-curb obstacle points translated into this case's frame."""
        from .real_boundaries import local_boundary_points

        return local_boundary_points(
            self.origin,
            spacing_m=spacing_m,
            radius_m=radius_m,
            flip_y=self.y_flipped,
        )


def _heading_from_points(p0: np.ndarray, p1: np.ndarray) -> float:
    delta = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    return float(np.arctan2(delta[1], delta[0]))


def _initial_speed(path: np.ndarray, times: np.ndarray, window: int = 3) -> float:
    if len(path) < 2:
        return 0.0
    window = min(window, len(path) - 1)
    seg = np.diff(path[: window + 1], axis=0)
    dt = np.diff(times[: window + 1])
    dt = np.where(dt > 0.0, dt, np.nan)
    speeds = np.linalg.norm(seg, axis=1) / dt
    speed = float(np.nanmean(speeds))
    return speed if np.isfinite(speed) else 0.0


def load_case(
    case_id: str,
    trajectory_points_path: Path = DEFAULT_TRAJECTORY_POINTS,
    *,
    flip_y: bool = False,
) -> ObservedCase:
    df = pd.read_csv(trajectory_points_path)
    case_df = df[df["case_id"] == case_id].sort_values("time").reset_index(drop=True)
    if case_df.empty:
        raise ValueError(f"case_id {case_id!r} not found in {trajectory_points_path}")
    return _case_from_frame(case_id, case_df, flip_y=flip_y)


def _case_from_frame(case_id: str, case_df: pd.DataFrame, *, flip_y: bool = False) -> ObservedCase:
    path_world = case_df[["xloc_kf", "yloc_kf"]].to_numpy(dtype=float)
    if flip_y:
        path_world[:, 1] *= -1.0
    abs_times = case_df["time"].to_numpy(dtype=float)
    times = abs_times - abs_times[0]
    origin = path_world[0].copy()
    path_local = path_world - origin

    head_window = min(3, len(path_local) - 1)
    initial_heading = _heading_from_points(path_local[0], path_local[head_window])
    terminal_heading = _heading_from_points(path_local[-2], path_local[-1])

    vehicle_id = case_df["id"].iloc[0] if "id" in case_df.columns else ""

    return ObservedCase(
        case_id=str(case_id),
        movement_name=str(case_df["movement_name"].iloc[0]),
        times=times,
        path_local=path_local,
        origin=origin,
        initial_speed=_initial_speed(path_local, times),
        initial_heading=initial_heading,
        terminal_heading=terminal_heading,
        vehicle_id=vehicle_id,
        start_time_abs=float(abs_times[0]),
        y_flipped=flip_y,
    )


def list_cases(trajectory_points_path: Path = DEFAULT_TRAJECTORY_POINTS) -> list[str]:
    df = pd.read_csv(trajectory_points_path, usecols=["case_id"])
    return list(dict.fromkeys(df["case_id"].astype(str).tolist()))
