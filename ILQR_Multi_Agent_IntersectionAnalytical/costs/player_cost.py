from __future__ import annotations
from typing import List, Optional
import numpy as np
from .base_cost import BaseCost, QuadraticApprox

class PlayerCost:
    """Container for all cost terms of a single player.

    This mirrors the role of ``PlayerCost`` in ilqgames-master while staying
    lightweight and NumPy-only. Costs may be evaluated directly or combined
    into a single local quadratic approximation.
    """

    def __init__(self, name: str, state_regularization: float = 0.0, control_regularization: float = 0.0):
        self.name = name
        self.cost_terms: List[BaseCost] = []
        self.state_regularization = float(state_regularization)
        self.control_regularization = float(control_regularization)

    def add_cost(self, cost: BaseCost) -> None:
        self.cost_terms.append(cost)

    def evaluate(self, x: np.ndarray, us: list[np.ndarray], k: Optional[int] = None) -> float:
        total = 0.0
        for cost in self.cost_terms:
            total += float(cost.evaluate(x, us, k))
        return total

    __call__ = evaluate

    def quadraticize(
        self,
        x: np.ndarray,
        us: list[np.ndarray],
        k: Optional[int] = None,
        *,
        include_const: bool = True,
    ) -> QuadraticApprox:
        x = np.asarray(x, dtype=float).reshape(-1)
        us = [np.asarray(u, dtype=float).reshape(-1) for u in us]
        nx = x.size
        Q = np.zeros((nx, nx), dtype=float)
        np.fill_diagonal(Q, self.state_regularization)
        R = {}
        r = {}
        for i, u in enumerate(us):
            Ri = np.zeros((u.size, u.size), dtype=float)
            np.fill_diagonal(Ri, self.control_regularization)
            R[i] = Ri
            r[i] = np.zeros(u.size, dtype=float)
        approx = QuadraticApprox(
            Q=Q,
            q=np.zeros(nx, dtype=float),
            R=R,
            r=r,
            S={},
            const=0.0,
        )
        for cost in self.cost_terms:
            accumulator = getattr(cost, "accumulate_quadratic", None)
            if accumulator is not None:
                approx.const += float(
                    accumulator(x, us, k, approx, compute_const=include_const)
                )
                continue
            term = cost.quadraticize(x, us, k)
            approx.Q += term.Q
            approx.q += term.q
            if include_const:
                approx.const += float(term.const)
            for i, Ri in term.R.items():
                if i in approx.R:
                    approx.R[i] += Ri
                else:
                    approx.R[i] = Ri.copy()
            for i, ri in term.r.items():
                if i in approx.r:
                    approx.r[i] += ri
                else:
                    approx.r[i] = ri.copy()
            for ij, Sij in term.S.items():
                if ij in approx.S:
                    approx.S[ij] += Sij
                else:
                    approx.S[ij] = Sij.copy()
        return approx
