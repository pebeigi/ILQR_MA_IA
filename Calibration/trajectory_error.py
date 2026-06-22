"""Pointwise trajectory error between a simulated path and an observed path.

Observed and simulated trajectories have different point counts and time
parameterizations, so we resample both by normalized arclength onto a common
grid and compare position pointwise.  Arclength (rather than time) makes the
metric robust to speed/duration differences while still penalizing shape and
endpoint mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryError:
    mean_distance_m: float
    rms_distance_m: float
    max_distance_m: float
    endpoint_distance_m: float

    @property
    def score(self) -> float:
        """Default scalar objective: RMS pointwise distance in meters."""
        return self.rms_distance_m


def _cumulative_arclength(path: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s


def resample_by_arclength(path: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample a polyline to ``n_samples`` points evenly spaced by arclength."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (N, 2)")
    if len(path) == 1:
        return np.repeat(path, n_samples, axis=0)

    s = _cumulative_arclength(path)
    total = s[-1]
    if total <= 0.0:
        return np.repeat(path[:1], n_samples, axis=0)

    targets = np.linspace(0.0, total, n_samples)
    x = np.interp(targets, s, path[:, 0])
    y = np.interp(targets, s, path[:, 1])
    return np.column_stack([x, y])


def trajectory_error(
    simulated_path: np.ndarray,
    observed_path: np.ndarray,
    n_samples: int = 100,
) -> TrajectoryError:
    sim = resample_by_arclength(simulated_path, n_samples)
    obs = resample_by_arclength(observed_path, n_samples)
    distances = np.linalg.norm(sim - obs, axis=1)
    endpoint = float(np.linalg.norm(simulated_path[-1] - observed_path[-1]))
    return TrajectoryError(
        mean_distance_m=float(np.mean(distances)),
        rms_distance_m=float(np.sqrt(np.mean(distances**2))),
        max_distance_m=float(np.max(distances)),
        endpoint_distance_m=endpoint,
    )
