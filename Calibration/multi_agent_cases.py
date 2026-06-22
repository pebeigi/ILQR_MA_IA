"""Reconstruct the multi-agent scene around an ego left-turn case.

Neighbors are vehicles from the raw TGSIM dataset that (a) overlap the ego's
time window and (b) come within a distance threshold of the ego path.  Each
neighbor is replayed from its real trajectory, expressed in the ego's local
frame (translated by the ego origin) and resampled onto the ego time grid.

The result is a per-timestep list of neighbor positions that the moving-obstacle
costs consume, plus the full neighbor tracks for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .observed_cases import ObservedCase, load_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "Data_Preparation"
    / "Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv"
)
RAW_COLUMNS = ["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "type_most_common"]
MOTOR_VEHICLE_TYPES = (3, 4, 5, 6, 7)


@dataclass
class NeighborTrack:
    vehicle_id: int
    type_code: int
    times_abs: np.ndarray  # (N,) absolute seconds
    xy_local: np.ndarray  # (N, 2) positions in ego local frame


@dataclass
class NeighborScene:
    ego: ObservedCase
    dt: float
    horizon_steps: int
    positions_per_step: list[np.ndarray]  # len == horizon_steps; each (M_k, 2)
    tracks: list[NeighborTrack] = field(default_factory=list)

    @property
    def n_neighbors(self) -> int:
        return len(self.tracks)


def _read_window(
    raw_path: Path,
    t_lo: float,
    t_hi: float,
    bbox: tuple[float, float, float, float],
    ego_id,
    flip_y: bool = False,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Stream the raw CSV and keep only rows in the ego time/space window."""
    x_lo, x_hi, y_lo, y_hi = bbox
    kept = []
    for chunk in pd.read_csv(raw_path, usecols=RAW_COLUMNS, chunksize=chunksize):
        y_filter = -chunk["yloc_kf"] if flip_y else chunk["yloc_kf"]
        m = (
            (chunk["time"] >= t_lo)
            & (chunk["time"] <= t_hi)
            & (chunk["xloc_kf"] >= x_lo)
            & (chunk["xloc_kf"] <= x_hi)
            & (y_filter >= y_lo)
            & (y_filter <= y_hi)
            & (chunk["type_most_common"].isin(MOTOR_VEHICLE_TYPES))
            & (chunk["id"] != ego_id)
        )
        if m.any():
            kept.append(chunk.loc[m, RAW_COLUMNS])
    if not kept:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.concat(kept, ignore_index=True)


def _min_distance_to_path(points_xy: np.ndarray, path_xy: np.ndarray) -> float:
    diff = points_xy[:, None, :] - path_xy[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.min(d))


def build_neighbor_scene(
    case_id: str,
    *,
    trajectory_points_path: Path | None = None,
    raw_path: Path = RAW_DATA_PATH,
    dt: float = 0.1,
    radius_m: float = 35.0,
    time_buffer_s: float = 2.0,
    min_horizon: int = 10,
    flip_y: bool = False,
) -> NeighborScene:
    if trajectory_points_path is None:
        ego = load_case(case_id, flip_y=flip_y)
    else:
        ego = load_case(case_id, trajectory_points_path, flip_y=flip_y)

    duration = float(ego.times[-1] - ego.times[0])
    horizon_steps = max(min_horizon, int(round(duration / dt)) + 1)
    t_abs = ego.start_time_abs + np.arange(horizon_steps) * dt

    path_world = ego.path_local + ego.origin
    x_lo, x_hi = path_world[:, 0].min() - radius_m, path_world[:, 0].max() + radius_m
    y_lo, y_hi = path_world[:, 1].min() - radius_m, path_world[:, 1].max() + radius_m

    raw = _read_window(
        raw_path,
        t_lo=ego.start_time_abs - time_buffer_s,
        t_hi=t_abs[-1] + time_buffer_s,
        bbox=(x_lo, x_hi, y_lo, y_hi),
        ego_id=ego.vehicle_id,
        flip_y=flip_y,
    )

    positions_per_step: list[np.ndarray] = [np.empty((0, 2)) for _ in range(horizon_steps)]
    tracks: list[NeighborTrack] = []

    if not raw.empty:
        for vehicle_id, vdf in raw.groupby("id", sort=False):
            vdf = vdf.sort_values("time")
            v_times = vdf["time"].to_numpy(dtype=float)
            v_xy_world = vdf[["xloc_kf", "yloc_kf"]].to_numpy(dtype=float)
            if flip_y:
                v_xy_world[:, 1] *= -1.0
            v_xy_local = v_xy_world - ego.origin

            # Qualify by proximity to the ego path.
            if _min_distance_to_path(v_xy_local, ego.path_local) > radius_m:
                continue

            tracks.append(
                NeighborTrack(
                    vehicle_id=int(vehicle_id),
                    type_code=int(vdf["type_most_common"].iloc[0]),
                    times_abs=v_times,
                    xy_local=v_xy_local,
                )
            )

            # Sample onto the ego time grid where the neighbor exists.
            in_span = (t_abs >= v_times[0]) & (t_abs <= v_times[-1])
            if not in_span.any():
                continue
            xi = np.interp(t_abs, v_times, v_xy_local[:, 0])
            yi = np.interp(t_abs, v_times, v_xy_local[:, 1])
            for k in np.nonzero(in_span)[0]:
                positions_per_step[k] = (
                    np.vstack([positions_per_step[k], [xi[k], yi[k]]])
                    if positions_per_step[k].size
                    else np.array([[xi[k], yi[k]]])
                )

    return NeighborScene(
        ego=ego,
        dt=dt,
        horizon_steps=horizon_steps,
        positions_per_step=positions_per_step,
        tracks=tracks,
    )
