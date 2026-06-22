"""Calibration parameter space for one agent's ILQR cost weights.

Each parameter is searched in a transformed space so the Bayesian optimizer
works on well-scaled, unconstrained-ish coordinates:

* ``log10`` weights span several orders of magnitude (e.g. q_speed, betas,
  terminal weights), so they are optimized in log10 space.
* ``desired_speed`` is a physical quantity optimized linearly.

A :class:`ParameterSpace` maps a normalized/raw search vector to and from an
:class:`~Calibration.ilqr_interface.AgentParameters` instance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .ilqr_interface import AgentParameters


@dataclass(frozen=True)
class ParameterDef:
    name: str
    low: float
    high: float
    transform: str = "log10"  # "log10" or "linear"

    def to_search(self, value: float) -> float:
        if self.transform == "log10":
            return float(np.log10(value))
        return float(value)

    def from_search(self, search_value: float) -> float:
        if self.transform == "log10":
            return float(10.0**search_value)
        return float(search_value)

    @property
    def search_bounds(self) -> tuple[float, float]:
        if self.transform == "log10":
            return (float(np.log10(self.low)), float(np.log10(self.high)))
        return (float(self.low), float(self.high))


DEFAULT_PARAMETER_DEFS: tuple[ParameterDef, ...] = (
    ParameterDef("q_speed", 1.0, 1000.0, "log10"),
    ParameterDef("desired_speed", 1.0, 15.0, "linear"),
    ParameterDef("beta_2", 0.1, 100.0, "log10"),
    ParameterDef("beta_3", 1.0, 500.0, "log10"),
    ParameterDef("running_destination", 0.1, 50.0, "log10"),
    ParameterDef("terminal_position_weight", 10.0, 2000.0, "log10"),
    ParameterDef("terminal_speed_weight", 1.0, 500.0, "log10"),
    ParameterDef("terminal_heading_weight", 1.0, 500.0, "log10"),
)


@dataclass(frozen=True)
class ParameterSpace:
    defs: tuple[ParameterDef, ...] = DEFAULT_PARAMETER_DEFS

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.defs]

    @property
    def search_bounds(self) -> np.ndarray:
        """(D, 2) array of [low, high] bounds in search space."""
        return np.array([d.search_bounds for d in self.defs], dtype=float)

    def to_params(self, search_vector: np.ndarray, base: AgentParameters | None = None) -> AgentParameters:
        base = base or AgentParameters()
        values = {d.name: d.from_search(float(v)) for d, v in zip(self.defs, search_vector)}
        return replace(base, **values)

    def to_search_vector(self, params: AgentParameters) -> np.ndarray:
        return np.array(
            [d.to_search(getattr(params, d.name)) for d in self.defs], dtype=float
        )

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        bounds = self.search_bounds
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, len(self.defs)))
