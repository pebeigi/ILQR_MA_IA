"""Minimal Bayesian optimization (GP + Expected Improvement).

Implemented on top of scikit-learn's ``GaussianProcessRegressor`` so we avoid an
extra dependency.  Designed for low-dimensional, expensive, possibly noisy
black-box objectives like "run ILQR and measure trajectory error".

Workflow:
    1. Evaluate ``n_initial`` random points (Latin-ish uniform) for a prior.
    2. Fit a GP to observed (x, y).
    3. Maximize Expected Improvement over many random candidates to pick next x.
    4. Repeat for ``n_iterations``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class BOResult:
    best_x: np.ndarray
    best_y: float
    xs: np.ndarray
    ys: np.ndarray
    history: list[float] = field(default_factory=list)


def _expected_improvement(
    candidates: np.ndarray,
    gp: GaussianProcessRegressor,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    mu, sigma = gp.predict(candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    # Minimization: improvement is reduction below current best.
    improvement = best_y - mu - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def minimize(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    *,
    n_initial: int = 8,
    n_iterations: int = 30,
    n_candidates: int = 2000,
    seed: int = 0,
    verbose: bool = True,
) -> BOResult:
    """Minimize ``objective`` over box ``bounds`` (shape (D, 2)) via GP-EI."""

    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    rng = np.random.default_rng(seed)

    def sample_uniform(n: int) -> np.ndarray:
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, dim))

    xs = sample_uniform(n_initial)
    ys = np.array([float(objective(x)) for x in xs], dtype=float)

    best_idx = int(np.argmin(ys))
    best_x, best_y = xs[best_idx].copy(), float(ys[best_idx])
    history = [best_y]
    if verbose:
        print(f"[BO] init best score={best_y:.4f} after {n_initial} random evals")

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e4), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )

    for it in range(n_iterations):
        y_mean = float(np.mean(ys))
        y_std = float(np.std(ys)) or 1.0
        ys_norm = (ys - y_mean) / y_std

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gp.fit(xs, ys_norm)

        candidates = sample_uniform(n_candidates)
        best_y_norm = (best_y - y_mean) / y_std
        ei = _expected_improvement(candidates, gp, best_y_norm)
        next_x = candidates[int(np.argmax(ei))]

        next_y = float(objective(next_x))
        xs = np.vstack([xs, next_x])
        ys = np.append(ys, next_y)

        if next_y < best_y:
            best_x, best_y = next_x.copy(), next_y
        history.append(best_y)
        if verbose:
            print(
                f"[BO] iter {it + 1:3d}/{n_iterations}  "
                f"eval={next_y:.4f}  best={best_y:.4f}"
            )

    return BOResult(best_x=best_x, best_y=best_y, xs=xs, ys=ys, history=history)
