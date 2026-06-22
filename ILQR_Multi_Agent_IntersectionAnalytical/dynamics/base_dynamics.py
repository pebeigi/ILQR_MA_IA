from __future__ import annotations
import numpy as np

class ConcatenatedDynamics:
    """Cartesian-product multi-player dynamics.

    This is the Python counterpart of the concatenated/product multi-player
    systems in ilqgames-master. Each subsystem receives its own slice of the
    state and its own control vector.
    """

    def __init__(self, subsystems, state_dims, control_dims=None):
        assert len(subsystems) == len(state_dims)
        self.subsystems = list(subsystems)
        self.state_dims = [int(d) for d in state_dims]
        self.control_dims = None if control_dims is None else [int(d) for d in control_dims]
        self.num_players = len(subsystems)
        self.start_dims = [0]
        for d in self.state_dims:
            self.start_dims.append(self.start_dims[-1] + d)
        self.xdim = self.start_dims[-1]

    def split_state(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return [x[self.start_dims[i]:self.start_dims[i+1]] for i in range(self.num_players)]

    def evaluate(self, t, x, us):
        x = np.asarray(x, dtype=float).reshape(-1)
        us = [np.asarray(u, dtype=float).reshape(-1) for u in us]
        xdot = np.zeros(self.xdim, dtype=float)
        for i, sys in enumerate(self.subsystems):
            start, end = self.start_dims[i], self.start_dims[i+1]
            xdot[start:end] = np.asarray(sys(x[start:end], us[i]), dtype=float).reshape(-1)
        return xdot

    def jacobians(self, t, x, us):
        """Return exact continuous-time Jacobians of the product dynamics."""
        x = np.asarray(x, dtype=float).reshape(-1)
        us = [np.asarray(u, dtype=float).reshape(-1) for u in us]
        if len(us) != self.num_players:
            raise ValueError(f"expected {self.num_players} controls, got {len(us)}")

        A = np.zeros((self.xdim, self.xdim), dtype=float)
        B = [np.zeros((self.xdim, u.size), dtype=float) for u in us]
        for i, sys in enumerate(self.subsystems):
            jacobian_fn = getattr(sys, "jacobians", None)
            if jacobian_fn is None:
                raise NotImplementedError(
                    f"subsystem {i} ({getattr(sys, '__name__', type(sys).__name__)}) "
                    "does not provide analytical Jacobians"
                )
            start, end = self.start_dims[i], self.start_dims[i + 1]
            Ai, Bi = jacobian_fn(x[start:end], us[i])
            A[start:end, start:end] = np.asarray(Ai, dtype=float)
            B[i][start:end, :] = np.asarray(Bi, dtype=float)
        return A, B
