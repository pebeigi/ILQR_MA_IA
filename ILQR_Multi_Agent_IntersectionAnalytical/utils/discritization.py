# utils/discritization.py
from __future__ import annotations

import math
from dataclasses import dataclass
from config import DT, HORIZON

@dataclass(frozen=True)
class TimeDiscretization:
    """
        Minimal time/discretization utilities.

        This is the Python equivalent of the C++ combo:
          - ilqgames/utils/types.h: defines dt, horizon, num_steps
          - ilqgames/utils/relative_time_tracker.h: provides time-index conversions
    """
    dt: float
    horizon: float
    eps: float = 1e-9

    @property
    def num_steps(self) -> int:
        # Compute how many discrete steps fit in the horizon.
        # The eps is there to avoid floating point issues like:
        # (10.0 / 0.1) becoming 99.999999 -> int(...) = 99 (wrong)
        return int((self.horizon + self.eps) / self.dt)

    def relative_time(self, k: int) -> float:
        # Converts a step index k to time since start (t - t0).
        # Example: k=3, dt=0.1 -> 0.3 seconds.
        return float(k) * self.dt

    def absolute_time(self, t0: float, k: int) -> float:
        # Converts a step index k to absolute time.
        # Example: t0=5.0, k=3, dt=0.1 -> 5.3 seconds.
        return float(t0) + float(k) * self.dt

    def time_index(self, t: float) -> int:
        """
                Maps a continuous time 't' to the discrete time index 'k'.

                We do floor((t + eps)/dt) so that times just below a grid point
                still map correctly due to floating point noise.

                Then we clamp between [0, num_steps] so we never go out of range.
        """
        k = int(math.floor((t + self.eps) / self.dt))
        return max(0, min(k, self.num_steps))


TIME = TimeDiscretization(dt=DT, horizon=HORIZON)