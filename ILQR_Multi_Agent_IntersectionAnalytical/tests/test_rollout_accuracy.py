import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# tests/test_rollout_accuracy.py
import numpy as np

from dynamics.player_dynamics import unicycle_4d
from dynamics.base_dynamics import ConcatenatedDynamics
from ilq.rollout import integrate
from config import DT

np.set_printoptions(precision=6, suppress=True)


def main():
    print("DT =", DT)

    x0 = np.array([0.0, 0.0, 0.0, 5.0])  # [px, py, theta, v]
    u0 = np.array([1.0, 0.0])            # [omega, a]
    us = [u0]

    dynamics = ConcatenatedDynamics(
        subsystems=[unicycle_4d],
        state_dims=[4]
    )

    x_euler = integrate(dynamics.evaluate, 0.0, DT, x0, us, use_euler=True)
    x_rk4   = integrate(dynamics.evaluate, 0.0, DT, x0, us, use_euler=False)

    # Analytic solution for constant omega, a=0, constant v
    omega, a = u0
    px0, py0, theta0, v0 = x0

    if abs(omega) < 1e-12:
        # Straight line
        x_true = np.array([
            px0 + v0 * np.cos(theta0) * DT,
            py0 + v0 * np.sin(theta0) * DT,
            theta0,
            v0
        ])
    else:
        R = v0 / omega
        x_true = np.array([
            px0 + R * (np.sin(theta0 + omega * DT) - np.sin(theta0)),
            py0 - R * (np.cos(theta0 + omega * DT) - np.cos(theta0)),
            theta0 + omega * DT,
            v0
        ])

    print("\nInitial state:\n", x0)
    print("\nEuler result:\n", x_euler)
    print("\nRK4 result:\n", x_rk4)
    print("\nAnalytic solution:\n", x_true)

    print("\nEuler error:\n", x_euler - x_true)
    print("\nRK4 error:\n", x_rk4 - x_true)


if __name__ == "__main__":
    main()
