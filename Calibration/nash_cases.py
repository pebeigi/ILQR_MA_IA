"""Reconstruct a full multi-agent scene for Nash-game calibration.

Unlike :mod:`Calibration.multi_agent_cases` (which replays neighbors as moving
obstacles), here *every* co-present vehicle becomes a real ILQR player.  Each
agent is seeded from data with its own initial state, destination, and terminal
heading, all expressed in the ego's local frame on a shared time grid.

The game is then solved jointly (a true Nash game), and each agent's simulated
trajectory is compared against its observed trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .multi_agent_cases import (
    RAW_DATA_PATH,
    _min_distance_to_path,
    _read_window,
)
from .observed_cases import load_case


DEFAULT_MIN_TRAVEL_M = 5.0


@dataclass
class AgentTrack:
    """One agent's observed motion on the shared ego time grid (local frame)."""

    name: str
    vehicle_id: int | str
    is_ego: bool
    obs_xy: np.ndarray  # (H, 2), NaN where the agent is absent
    valid_mask: np.ndarray  # (H,) bool
    entry_step: int
    exit_step: int  # inclusive last valid step
    initial_state: np.ndarray  # [px, py, speed, heading]
    destination: np.ndarray  # (2,)
    terminal_heading: float
    mean_speed: float
    path_length_m: float = 0.0

    @property
    def observed_path(self) -> np.ndarray:
        """Observed positions over the agent's valid span (entry..exit)."""
        return self.obs_xy[self.entry_step : self.exit_step + 1]


@dataclass
class GameScene:
    agents: list[AgentTrack]  # active game players (calibrated)
    dt: float
    horizon_steps: int
    ego_origin: np.ndarray
    y_flipped: bool = False
    times: np.ndarray = field(default_factory=lambda: np.empty(0))
    ignored_agents: list[AgentTrack] = field(default_factory=list)  # stopped / not in game
    min_travel_m: float = DEFAULT_MIN_TRAVEL_M

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def n_neighbors(self) -> int:
        return sum(0 if a.is_ego else 1 for a in self.agents)

    @property
    def n_ignored(self) -> int:
        return len(self.ignored_agents)


def path_length_m(xy: np.ndarray) -> float:
    """Total arclength of a polyline (m)."""
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def _heading_from_points(p0: np.ndarray, p1: np.ndarray) -> float:
    delta = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    if float(np.hypot(delta[0], delta[1])) < 1e-9:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def _estimate_state(xy: np.ndarray, times: np.ndarray, idx: int) -> tuple[float, float]:
    """Return (speed, heading) near sample ``idx`` from finite differences."""
    n = len(xy)
    if n < 2:
        return 0.0, 0.0
    j = min(idx + 1, n - 1)
    i = max(j - 1, 0)
    dt = float(times[j] - times[i])
    if dt <= 0.0:
        return 0.0, _heading_from_points(xy[i], xy[j])
    step = xy[j] - xy[i]
    speed = float(np.hypot(step[0], step[1]) / dt)
    return speed, _heading_from_points(xy[i], xy[j])


def _resample_to_grid(
    xy: np.ndarray, src_times: np.ndarray, grid_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a track onto ``grid_times``; mask steps outside its span."""
    valid = (grid_times >= src_times[0]) & (grid_times <= src_times[-1])
    out = np.full((len(grid_times), 2), np.nan, dtype=float)
    if valid.any():
        out[valid, 0] = np.interp(grid_times[valid], src_times, xy[:, 0])
        out[valid, 1] = np.interp(grid_times[valid], src_times, xy[:, 1])
    return out, valid


def _track_from_grid(
    name: str,
    vehicle_id: int | str,
    is_ego: bool,
    obs_xy: np.ndarray,
    valid_mask: np.ndarray,
    grid_times: np.ndarray,
    *,
    initial_speed: float | None = None,
    initial_heading: float | None = None,
    terminal_heading: float | None = None,
) -> AgentTrack:
    valid_idx = np.nonzero(valid_mask)[0]
    entry, exit_ = int(valid_idx[0]), int(valid_idx[-1])
    span_xy = obs_xy[entry : exit_ + 1]
    span_t = grid_times[entry : exit_ + 1]

    est_speed, est_heading = _estimate_state(span_xy, span_t, 0)
    speed0 = float(initial_speed if initial_speed is not None else est_speed)
    heading0 = float(initial_heading if initial_heading is not None else est_heading)

    if terminal_heading is None and len(span_xy) >= 2:
        terminal_heading = _heading_from_points(span_xy[-2], span_xy[-1])
    terminal_heading = float(terminal_heading or heading0)

    # Mean speed over the valid span (used to seed desired_speed).
    if len(span_xy) >= 2 and (span_t[-1] - span_t[0]) > 0:
        dists = np.linalg.norm(np.diff(span_xy, axis=0), axis=1)
        mean_speed = float(np.sum(dists) / (span_t[-1] - span_t[0]))
    else:
        mean_speed = speed0

    initial_state = np.array([span_xy[0, 0], span_xy[0, 1], speed0, heading0], dtype=float)
    return AgentTrack(
        name=name,
        vehicle_id=vehicle_id,
        is_ego=is_ego,
        obs_xy=obs_xy,
        valid_mask=valid_mask,
        entry_step=entry,
        exit_step=exit_,
        initial_state=initial_state,
        destination=span_xy[-1].copy(),
        terminal_heading=terminal_heading,
        mean_speed=mean_speed,
        path_length_m=path_length_m(span_xy),
    )


def build_game_scene(
    case_id: str,
    *,
    trajectory_points_path: Path | None = None,
    raw_path: Path = RAW_DATA_PATH,
    dt: float = 0.1,
    radius_m: float = 35.0,
    time_buffer_s: float = 2.0,
    min_horizon: int = 10,
    min_presence_frac: float = 0.5,
    min_travel_m: float = DEFAULT_MIN_TRAVEL_M,
    max_agents: int | None = None,
    flip_y: bool = False,
) -> GameScene:
    """Build a Nash-game scene: ego + qualifying neighbors as real players.

    Neighbors that travel less than ``min_travel_m`` over their observed span
    (e.g. waiting at a red signal) are excluded from the game but returned in
    ``ignored_agents`` for plotting.
    """
    if trajectory_points_path is None:
        ego = load_case(case_id, flip_y=flip_y)
    else:
        ego = load_case(case_id, trajectory_points_path, flip_y=flip_y)

    duration = float(ego.times[-1] - ego.times[0])
    horizon_steps = max(min_horizon, int(round(duration / dt)) + 1)
    grid_times = np.arange(horizon_steps) * dt  # ego-relative seconds
    t_abs = ego.start_time_abs + grid_times

    # Ego track on the shared grid.
    ego_obs, ego_valid = _resample_to_grid(ego.path_local, ego.times, grid_times)
    if not ego_valid.any():
        raise ValueError(f"ego case {case_id!r} has no valid samples on the grid")
    agents: list[AgentTrack] = [
        _track_from_grid(
            "ego",
            ego.vehicle_id,
            True,
            ego_obs,
            ego_valid,
            grid_times,
            initial_speed=ego.initial_speed,
            initial_heading=ego.initial_heading,
            terminal_heading=ego.terminal_heading,
        )
    ]

    # Neighbors from the raw dataset.
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

    min_valid_steps = max(2, int(round(min_presence_frac * horizon_steps)))
    active_candidates: list[tuple[int, AgentTrack]] = []
    ignored_agents: list[AgentTrack] = []
    if not raw.empty:
        for vehicle_id, vdf in raw.groupby("id", sort=False):
            vdf = vdf.sort_values("time")
            v_times = vdf["time"].to_numpy(dtype=float) - ego.start_time_abs  # ego-relative
            v_xy_world = vdf[["xloc_kf", "yloc_kf"]].to_numpy(dtype=float)
            if flip_y:
                v_xy_world[:, 1] *= -1.0
            v_xy_local = v_xy_world - ego.origin

            if _min_distance_to_path(v_xy_local, ego.path_local) > radius_m:
                continue

            obs_xy, valid = _resample_to_grid(v_xy_local, v_times, grid_times)
            if int(valid.sum()) < min_valid_steps:
                continue

            track = _track_from_grid(
                f"veh_{int(vehicle_id)}", int(vehicle_id), False, obs_xy, valid, grid_times
            )
            if track.path_length_m < min_travel_m:
                ignored_agents.append(track)
            else:
                active_candidates.append((int(valid.sum()), track))

    # Keep the longest-present moving neighbors first (most informative for the game).
    active_candidates.sort(key=lambda t: t[0], reverse=True)
    if max_agents is not None:
        active_candidates = active_candidates[: max(0, max_agents - 1)]
    agents.extend(track for _, track in active_candidates)

    return GameScene(
        agents=agents,
        dt=dt,
        horizon_steps=horizon_steps,
        ego_origin=ego.origin.copy(),
        y_flipped=flip_y,
        times=grid_times,
        ignored_agents=ignored_agents,
        min_travel_m=min_travel_m,
    )
