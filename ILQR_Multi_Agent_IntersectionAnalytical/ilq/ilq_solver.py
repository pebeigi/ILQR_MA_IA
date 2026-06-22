from __future__ import annotations
import time as _time
import numpy as np

try:
    from ..config import (
        MAX_ITERATIONS,
        CONVERGENCE_TOL,
        RELATIVE_COST_CONVERGENCE_TOL,
        CONVERGENCE_PATIENCE,
    )
    from ..ilq.rollout import integrate
    from ..ilq.linearization import linearize_discrete
    from ..ilq.quadraticization import pack_for_lq_game
    from ..lq_game.lq_feedback_solver import solve_feedback_lq_game
    from ..lq_game.lq_open_loop_solver import solve_lq_game as solve_open_loop_lq_game
except ImportError:  # pragma: no cover
    from config import (
        MAX_ITERATIONS,
        CONVERGENCE_TOL,
        RELATIVE_COST_CONVERGENCE_TOL,
        CONVERGENCE_PATIENCE,
    )
    from ilq.rollout import integrate
    from ilq.linearization import linearize_discrete
    from ilq.quadraticization import pack_for_lq_game
    from lq_game.lq_feedback_solver import solve_feedback_lq_game
    from lq_game.lq_open_loop_solver import solve_lq_game as solve_open_loop_lq_game

class ILQSolver:
    """Iterative LQ game solver using either feedback-Nash or open-loop Nash LQ subproblems."""

    def __init__(
        self,
        game,
        use_euler=False,
        alpha_scaling=0.05,
        max_iterations=MAX_ITERATIONS,
        convergence_tol=CONVERGENCE_TOL,
        relative_cost_convergence_tol=RELATIVE_COST_CONVERGENCE_TOL,
        convergence_patience=CONVERGENCE_PATIENCE,
        lq_solver_type="feedback",
        min_iterations=0,
        alpha_line_search=False,
        alpha_line_search_min=0.03125,
        alpha_line_search_shrink=0.5,
        alpha_line_search_max_growth=5.0,
        alpha_line_search_start_iteration=0,
        alpha_line_search_cost_tol=1e-9,
    ):
        self.game = game
        self.use_euler = bool(use_euler)
        self.alpha_scaling = float(alpha_scaling)
        self.current_alpha_scaling = float(alpha_scaling)
        self.max_iterations = int(max_iterations)
        self.convergence_tol = float(convergence_tol)
        self.relative_cost_convergence_tol = float(relative_cost_convergence_tol)
        self.convergence_patience = int(convergence_patience)
        self.min_iterations = int(min_iterations)
        self.lq_solver_type = lq_solver_type.lower()
        self.alpha_line_search = bool(alpha_line_search)
        self.alpha_line_search_min = float(alpha_line_search_min)
        self.alpha_line_search_shrink = float(alpha_line_search_shrink)
        self.alpha_line_search_max_growth = float(alpha_line_search_max_growth)
        self.alpha_line_search_start_iteration = int(alpha_line_search_start_iteration)
        self.alpha_line_search_cost_tol = float(alpha_line_search_cost_tol)
        if self.alpha_line_search_min <= 0.0:
            raise ValueError("alpha_line_search_min must be positive")
        if not 0.0 < self.alpha_line_search_shrink < 1.0:
            raise ValueError("alpha_line_search_shrink must be in (0, 1)")
        if self.alpha_line_search_max_growth <= 0.0:
            raise ValueError("alpha_line_search_max_growth must be positive")
        if self.alpha_line_search_start_iteration < 0:
            raise ValueError("alpha_line_search_start_iteration must be nonnegative")
        if self.alpha_line_search_cost_tol < 0.0:
            raise ValueError("alpha_line_search_cost_tol must be nonnegative")
        if self.convergence_tol <= 0.0:
            raise ValueError("convergence_tol must be positive")
        if self.relative_cost_convergence_tol <= 0.0:
            raise ValueError("relative_cost_convergence_tol must be positive")
        if self.convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive")
        self.num_players = game.num_players
        self.horizon_steps = int(game.horizon_steps)
        self.xdim = int(game.state_dim)
        self.udims = game.control_dims
        self.Ps = [[np.zeros((ud, self.xdim)) for _ in range(self.horizon_steps)] for ud in self.udims]
        self.alphas = [[np.zeros((ud, 1)) for _ in range(self.horizon_steps)] for ud in self.udims]
        self.current_operating_point = None
        self.last_operating_point = None
        self._pending_rollout = None

    def _rollout_with_strategy(self, alpha_scaling=None, Ps=None, alphas=None):
        alpha = self.current_alpha_scaling if alpha_scaling is None else float(alpha_scaling)
        Ps = self.Ps if Ps is None else Ps
        alphas = self.alphas if alphas is None else alphas
        xs = [self.game.x0.copy()]
        us = [[] for _ in range(self.num_players)]
        costs = [[] for _ in range(self.num_players)]
        t = 0.0
        for k in range(self.horizon_steps):
            xk = xs[-1]
            if self.current_operating_point is None:
                x_ref = np.zeros_like(xk)
                u_ref = [np.zeros((ud,)) for ud in self.udims]
            else:
                x_ref = self.current_operating_point[0][k]
                u_ref = [self.current_operating_point[1][i][k] for i in range(self.num_players)]
            u_k = []
            for i in range(self.num_players):
                ui = u_ref[i] - Ps[i][k] @ (xk - x_ref) - alpha * alphas[i][k].reshape(-1)
                ui = np.asarray(ui, dtype=float).reshape(-1)
                # Safety clamp used by the supplied vehicle examples. Guard the
                # indices so the solver still works for lower-dimensional controls.
                if ui.size >= 1:
                    steering = float(ui[0])
                    if steering < -0.6:
                        steering = -0.6
                    elif steering > 0.6:
                        steering = 0.6
                    ui[0] = steering
                if ui.size >= 2:
                    acceleration = float(ui[1])
                    if acceleration < -8.0:
                        acceleration = -8.0
                    elif acceleration > 8.0:
                        acceleration = 8.0
                    ui[1] = acceleration
                u_k.append(ui)
                us[i].append(ui)
            for i in range(self.num_players):
                costs[i].append(float(self.game.player_costs[i].evaluate(xk, u_k, k)))
            if k < self.horizon_steps - 1:
                xs.append(integrate(self.game.dynamics.evaluate, t, self.game.dt, xk, u_k, use_euler=self.use_euler))
                t += self.game.dt
        return xs, us, costs

    def _is_finite_rollout(self, xs, us, costs):
        for x in xs:
            if not np.all(np.isfinite(x)):
                return False
        for player_us in us:
            for u in player_us:
                if not np.all(np.isfinite(u)):
                    return False
        for player_costs in costs:
            if not np.all(np.isfinite(player_costs)):
                return False
        return True

    def _max_state_update_from_current(self, xs):
        if self.current_operating_point is None:
            return 0.0
        return max(
            float(np.linalg.norm(a - b))
            for a, b in zip(xs, self.current_operating_point[0])
        )

    @staticmethod
    def _player_total_costs(costs):
        return np.asarray([float(sum(ci)) for ci in costs], dtype=float)

    def _choose_alpha_scaling(self, previous_max_delta, iteration):
        """Backtracking line search on alpha using realized rollout cost.

        The nonlinear game does not have a true single objective, but the summed
        player cost is a useful globalization merit for these traffic examples.
        State-update size is kept only as a runaway guard; minimizing it directly
        can make the convergence plot look good by accepting tiny, stalled steps.
        """
        if not self.alpha_line_search:
            self.current_alpha_scaling = self.alpha_scaling
            return self.current_alpha_scaling, None
        if iteration < self.alpha_line_search_start_iteration:
            self.current_alpha_scaling = self.alpha_scaling
            return self.current_alpha_scaling, {
                "alpha": self.current_alpha_scaling,
                "max_delta_x": None,
                "total_cost": None,
                "skipped": "warmup",
            }

        reference_costs = None
        reference_merit = None
        if self.current_operating_point is not None:
            reference_costs = self._player_total_costs(self.current_operating_point[2])
            reference_merit = float(np.sum(reference_costs))

        growth_threshold = None
        if previous_max_delta > self.convergence_tol:
            growth_threshold = self.alpha_line_search_max_growth * previous_max_delta
        merit_tol = 0.0
        if reference_merit is not None:
            merit_tol = self.alpha_line_search_cost_tol * max(1.0, abs(reference_merit))

        eta = self.alpha_scaling
        best_guarded = None
        best_guarded_rollout = None
        least_updating_finite = None
        least_updating_finite_rollout = None
        trials = []
        while eta >= self.alpha_line_search_min - 1e-15:
            xs, us, costs = self._rollout_with_strategy(alpha_scaling=eta)
            if self._is_finite_rollout(xs, us, costs):
                max_delta = self._max_state_update_from_current(xs)
                player_costs = self._player_total_costs(costs)
                merit = float(np.sum(player_costs))
                merit_change = None if reference_merit is None else merit - reference_merit
                within_delta_guard = growth_threshold is None or max_delta <= growth_threshold
                candidate = {
                    "alpha": float(eta),
                    "max_delta_x": float(max_delta),
                    "player_costs": player_costs.tolist(),
                    "total_cost": merit,
                    "merit": merit,
                    "merit_change": merit_change,
                    "within_delta_guard": bool(within_delta_guard),
                }
                trials.append(candidate)
                if (
                    least_updating_finite is None
                    or max_delta < least_updating_finite["max_delta_x"]
                ):
                    least_updating_finite = candidate
                    least_updating_finite_rollout = (xs, us, costs)
                if within_delta_guard and (
                    best_guarded is None or merit < best_guarded["merit"]
                ):
                    best_guarded = candidate
                    best_guarded_rollout = (xs, us, costs)
                improves_merit = (
                    reference_merit is None
                    or merit <= reference_merit - merit_tol
                )
                if within_delta_guard and improves_merit:
                    candidate = dict(candidate)
                    candidate["accepted"] = True
                    candidate["reason"] = "merit_decrease"
                    candidate["reference_total_cost"] = reference_merit
                    if reference_costs is not None:
                        candidate["reference_player_costs"] = reference_costs.tolist()
                    candidate["trials"] = trials
                    self.current_alpha_scaling = float(eta)
                    self._pending_rollout = (xs, us, costs)
                    return self.current_alpha_scaling, candidate
            else:
                trials.append({
                    "alpha": float(eta),
                    "max_delta_x": float("inf"),
                    "total_cost": float("inf"),
                    "finite": False,
                })
            eta *= self.alpha_line_search_shrink

        # If no candidate clears the merit test, keep the least-cost guarded
        # rollout so the solver can keep moving, and record that this was not a
        # proper line-search acceptance.
        if best_guarded is not None:
            self.current_alpha_scaling = best_guarded["alpha"]
            self._pending_rollout = best_guarded_rollout
            best_guarded = dict(best_guarded)
            best_guarded["accepted"] = False
            best_guarded["fallback"] = True
            best_guarded["reason"] = "best_guarded_merit"
            best_guarded["reference_total_cost"] = reference_merit
            if reference_costs is not None:
                best_guarded["reference_player_costs"] = reference_costs.tolist()
            best_guarded["trials"] = trials
            return self.current_alpha_scaling, best_guarded

        # If every finite candidate violated the delta guard, use the least
        # updating finite rollout rather than accepting an obvious runaway.
        if least_updating_finite is not None:
            self.current_alpha_scaling = least_updating_finite["alpha"]
            self._pending_rollout = least_updating_finite_rollout
            least_updating_finite = dict(least_updating_finite)
            least_updating_finite["accepted"] = False
            least_updating_finite["fallback"] = True
            least_updating_finite["reason"] = "least_updating_finite"
            least_updating_finite["reference_total_cost"] = reference_merit
            if reference_costs is not None:
                least_updating_finite["reference_player_costs"] = reference_costs.tolist()
            least_updating_finite["trials"] = trials
            return self.current_alpha_scaling, least_updating_finite

        self.current_alpha_scaling = self.alpha_line_search_min
        return self.current_alpha_scaling, {
            "alpha": self.current_alpha_scaling,
            "max_delta_x": float("inf"),
            "total_cost": float("inf"),
            "merit": float("inf"),
            "accepted": False,
            "fallback": True,
            "reason": "no_finite_rollout",
            "reference_total_cost": reference_merit,
            "reference_player_costs": None if reference_costs is None else reference_costs.tolist(),
            "trials": trials,
        }

    def _convergence_metrics(self, previous_total_cost, current_total_cost):
        if self.last_operating_point is None or self.current_operating_point is None:
            return float("inf"), float("inf")
        prev_xs = self.last_operating_point[0]
        curr_xs = self.current_operating_point[0]
        max_delta_x = max(float(np.linalg.norm(a - b)) for a, b in zip(prev_xs, curr_xs))
        relative_delta_cost = abs(current_total_cost - previous_total_cost) / max(
            1.0,
            abs(previous_total_cost),
        )
        return max_delta_x, float(relative_delta_cost)

    def _update_convergence_streak(self, current_streak, max_delta_x, relative_delta_cost):
        meets_criteria = (
            max_delta_x < self.convergence_tol
            and relative_delta_cost < self.relative_cost_convergence_tol
        )
        return current_streak + 1 if meets_criteria else 0

    def solve(self):
        history = []
        iteration_trajectories = []
        iteration_delta_xs = []
        alpha_history = []
        line_search_history = []
        relative_delta_costs = []
        convergence_streak_history = []
        convergence_streak = 0
        did_converge = False

        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=self.max_iterations, desc=f"ILQR ({self.lq_solver_type})", unit="iter", ncols=80)
            _have_tqdm = True
        except ImportError:
            _have_tqdm = False
            pbar = None

        t0 = _time.perf_counter()

        for iteration in range(self.max_iterations):
            if self._pending_rollout is None:
                xs, us, costs = self._rollout_with_strategy()
            else:
                xs, us, costs = self._pending_rollout
                self._pending_rollout = None

            # Track trajectory per iteration
            iteration_trajectories.append(xs)

            # Track delta_x per iteration (relative to last operating point)
            if self.current_operating_point is not None:
                dxs = [float(np.linalg.norm(a - b)) for a, b in zip(xs, self.current_operating_point[0])]
                iteration_delta_xs.append(dxs)
            else:
                iteration_delta_xs.append([0.0] * len(xs))
            previous_max_delta = max(iteration_delta_xs[-1]) if iteration_delta_xs[-1] else 0.0

            self.last_operating_point = self.current_operating_point
            self.current_operating_point = (xs, us, costs)
            total_costs = [sum(ci) for ci in costs]
            history.append(total_costs)
            if len(history) > 1:
                previous_total_cost = float(sum(history[-2]))
                current_total_cost = float(sum(history[-1]))
                max_delta_x, relative_delta_cost = self._convergence_metrics(
                    previous_total_cost,
                    current_total_cost,
                )
                convergence_streak = self._update_convergence_streak(
                    convergence_streak,
                    max_delta_x,
                    relative_delta_cost,
                )
            else:
                relative_delta_cost = float("inf")
                convergence_streak = 0
            relative_delta_costs.append(relative_delta_cost)
            convergence_streak_history.append(convergence_streak)
            converged = (
                len(history) >= self.min_iterations
                and convergence_streak >= self.convergence_patience
            )

            elapsed = _time.perf_counter() - t0
            if _have_tqdm:
                pbar.set_postfix(cost=f"{sum(total_costs):.0f}", t=f"{elapsed:.0f}s")
                pbar.update(1)
            else:
                print(
                    f"\r  iter {iteration + 1:3d}/{self.max_iterations}  "
                    f"cost={sum(total_costs):9.0f}  t={elapsed:.0f}s  ",
                    end="", flush=True,
                )

            if converged:
                did_converge = True
                break

            As = []
            Bs = [[] for _ in range(self.num_players)]
            t = 0.0
            for k in range(self.horizon_steps):
                us_k = [us[i][k] for i in range(self.num_players)]
                A, B = linearize_discrete(
                    self.game.dynamics.evaluate,
                    t,
                    xs[k],
                    us_k,
                    self.game.dt,
                    use_euler=self.use_euler,
                )
                As.append(A)
                for i in range(self.num_players):
                    Bs[i].append(B[i])
                t += self.game.dt

            Qs, ls, Rs, rs = pack_for_lq_game(self.game.player_costs, xs, us)
            
            if self.lq_solver_type == "feedback":
                self.Ps, self.alphas = solve_feedback_lq_game(As, Bs, Qs, ls, Rs, rs)
            elif self.lq_solver_type == "open_loop":
                self.Ps, self.alphas = solve_open_loop_lq_game(As, Bs, Qs, ls, Rs, rs)
            else:
                raise ValueError(f"Unknown lq_solver_type: {self.lq_solver_type}. Use 'feedback' or 'open_loop'.")

            accepted_alpha, line_search_info = self._choose_alpha_scaling(
                previous_max_delta,
                iteration,
            )
            alpha_history.append(float(accepted_alpha))
            line_search_history.append(line_search_info)

        if _have_tqdm:
            pbar.close()
        else:
            print()

        total_time = _time.perf_counter() - t0
        n_iters = len(history)
        status = "converged" if did_converge else "max iterations reached"
        print(f"ILQR: {n_iters} iterations in {total_time:.1f}s ({status})")

        return {
            "xs": self.current_operating_point[0],
            "us": self.current_operating_point[1],
            "costs": self.current_operating_point[2],
            "Ps": self.Ps,
            "alphas": self.alphas,
            "history": history,
            "iteration_trajectories": iteration_trajectories,
            "iteration_delta_xs": iteration_delta_xs,
            "relative_delta_costs": relative_delta_costs,
            "convergence_streak_history": convergence_streak_history,
            "converged": did_converge,
            "convergence_tolerances": {
                "max_delta_x": self.convergence_tol,
                "relative_delta_cost": self.relative_cost_convergence_tol,
                "patience": self.convergence_patience,
            },
            "alpha_history": alpha_history,
            "line_search_history": line_search_history,
            "lq_solver": self.lq_solver_type,
        }
