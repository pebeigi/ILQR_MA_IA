import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

import main
from costs.base_cost import QuadraticApprox


def test_main_cost_accumulators_match_standalone_quadraticizations():
    game = main.build_multi_agent_game(main.AGENTS)
    xs, us, _ = main.make_initial_nominal_trajectory(game, main.AGENTS)

    for k in (0, game.horizon_steps - 1):
        x = xs[k]
        us_k = [us[i][k] for i in range(game.num_players)]
        for player_cost in game.player_costs:
            for cost in player_cost.cost_terms:
                accumulator = getattr(cost, "accumulate_quadratic", None)
                if accumulator is None:
                    continue

                expected = cost.quadraticize(x, us_k, k)
                actual = QuadraticApprox(
                    Q=np.zeros_like(expected.Q),
                    q=np.zeros_like(expected.q),
                    R={i: np.zeros((u.size, u.size)) for i, u in enumerate(us_k)},
                    r={i: np.zeros(u.size) for i, u in enumerate(us_k)},
                    S={},
                    const=0.0,
                )
                actual.const = accumulator(x, us_k, k, actual)

                np.testing.assert_allclose(actual.Q, expected.Q, atol=1e-11, rtol=0.0)
                np.testing.assert_allclose(actual.q, expected.q, atol=1e-11, rtol=0.0)
                np.testing.assert_allclose(actual.const, expected.const, atol=1e-11, rtol=0.0)
                for i in expected.R:
                    np.testing.assert_allclose(actual.R[i], expected.R[i], atol=1e-11, rtol=0.0)
                for i in expected.r:
                    np.testing.assert_allclose(actual.r[i], expected.r[i], atol=1e-11, rtol=0.0)
