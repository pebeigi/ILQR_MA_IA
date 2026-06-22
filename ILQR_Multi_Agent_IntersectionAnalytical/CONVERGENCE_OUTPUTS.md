# Convergence outputs

This feedback-Nash package now saves the ILQ convergence curve when `python main.py` is run.

Generated files:

- `multi_agent_trajectory_convergence.csv`
  - one row per ILQ iteration
  - one cost column per agent
  - one `total_cost` column containing the sum across agents

- `multi_agent_trajectory_convergence.png`
  - cost of each agent as a function of ILQ iteration
  - total cost curve is also shown when more than one agent is present

Implementation changes:

- Added `utils/convergence.py` with `save_convergence_history(...)`.
- Updated `main.py` to call `save_convergence_history(result["history"], ...)` immediately after `solver.solve()`.
