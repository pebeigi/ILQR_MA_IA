# Feedback-Nash Adaptation Notes

This package keeps the updated Python cost functions from `ILQR_Multi_Agent` and changes the local LQ subproblem used by `ILQSolver` to a feedback Nash equilibrium.

## Main Python changes

- `lq_game/lq_feedback_solver.py` was added. It computes closed-loop Nash strategies of the form
  `du_i[k] = -P_i[k] dx[k] - alpha_i[k]`.
- `ilq/ilq_solver.py` now imports `solve_feedback_lq_game` and calls it in the ILQ iteration.
- `lq_game/__init__.py` exports `solve_feedback_lq_game`.
- The previous `lq_game/lq_open_loop_solver.py` is retained only for comparison/backward compatibility; the iterative solver no longer uses it.

## C++ files adapted from `ilqgames-master`

| New/modified Python file | C++ source file(s) adapted | What was adapted |
|---|---|---|
| `lq_game/lq_feedback_solver.py` | `include/ilqgames/solver/lq_feedback_solver.h` | Solver interface, feedback strategy notation, allocation structure for `Ps`, `alphas`, `Zs`, and `zetas`. |
| `lq_game/lq_feedback_solver.py` | `src/lq_feedback_solver.cpp` | Backward dynamic-programming recursion, construction of the coupled Nash matrix `S`, block right-hand side `Y`, solution of `S X = Y`, extraction of feedback gains and feedforward terms, closed-loop `F`/`beta`, and value-function recursion. |
| `lq_game/lq_feedback_solver.py` | `include/ilqgames/utils/strategy.h` | Strategy convention `u = u_ref - P * delta_x - alpha`. |
| `ilq/ilq_solver.py` | `src/ilq_solver.cpp` | Use of feedback strategies during rollout: `current_u = last_u - P * delta_x - alpha`, with optional alpha scaling. |
| `ilq/ilq_solver.py` | `include/ilqgames/solver/lq_solver.h` | The LQ solver abstraction: ILQ repeatedly linearizes dynamics, quadraticizes costs, solves an LQ game, and rolls out the new strategy. |

## Important implementation details

- The final time sample is treated as terminal cost, matching `LQFeedbackSolver::Solve` in C++.
- The feedback solver uses the coupled block matrix equation `S X = Y` from the C++ implementation.
- The optional Gershgorin-style diagonal regularization from the C++ solver is implemented in `_regularize_gershgorin`.
- The solver returns both feedback gains `Ps` and feedforward terms `alphas`.
- All code is NumPy-only Python; no C++ bindings are required.
