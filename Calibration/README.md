# Calibration

This folder starts the calibration workflow for fitting the ILQR model to the
prepared trajectory data in `Data_Preparation/outputs/left_turn_movements`.

The first step is to convert observed trajectory points into comparable
movement-level targets:

```bash
python Calibration/calibrate.py
```

By default this reads:

```text
Data_Preparation/outputs/left_turn_movements/left_turn_intersection_zone_points.csv
```

and writes:

```text
Calibration/outputs/observed_case_features.csv
Calibration/outputs/observed_movement_targets.csv
```

The current objective compares movement-level features such as duration, path
length, mean speed, max speed, and end position.

## Trajectory-level calibration (Bayesian optimization)

The main calibrator fits one agent's ILQR cost weights so the simulated path
matches an observed left-turn case pointwise.

```bash
# list available observed cases
python -m Calibration.run_calibration --list

# calibrate one case (with an observed-vs-simulated plot)
python -m Calibration.run_calibration \
    --case-id "719_1_2_5_I_WB_->_23_SB_middle" \
    --n-initial 8 --n-iterations 30 --solver-iterations 60 --plot
```

### How it works

| Step | Module | Description |
|------|--------|-------------|
| Forward model | `ilqr_interface.py` | Builds + solves a single-agent ILQR scenario, returns the simulated trajectory. |
| Observed case | `observed_cases.py` | Loads one case, translates it to a local frame, derives initial state / destination / horizon. |
| Objective | `trajectory_error.py` | Arclength-resamples both paths and returns RMS pointwise position error. |
| Search space | `parameters.py` | Per-agent cost weights with log/linear bounds and vector<->params mapping. |
| Optimizer | `bayes_opt.py` | Gaussian-process Bayesian optimization with Expected Improvement (uses scikit-learn). |
| Driver | `calibrate_agent.py`, `run_calibration.py` | Tie everything together and save results. |

### Inputs and outputs

- **Calibrated parameters (`theta`):** `q_speed`, `desired_speed`, `beta_2`,
  `beta_3`, `running_destination`, and terminal position/speed/heading weights.
- **Model input:** observed initial state (position, speed, heading),
  destination, terminal heading, horizon, and a candidate `theta`.
- **Model output:** simulated trajectory `[px, py, speed, heading]` per step.
- **Calibration output:** the fitted `theta*` minimizing RMS trajectory error,
  plus an error breakdown, written to
  `Calibration/outputs/agent_calibration/`.

## Real street boundaries

The simulation can use the real Foggy Bottom street boundaries instead of the
synthetic intersection. Polygons come from `Foggy_Bottom_boundaries.txt`
(pixels) and are converted with `1 px = 0.0186613838586 m` into the same frame
as the trajectory data (verified — no axis flip).

`real_boundaries.py` keeps only the **side curbs** of each street: lanes are
grouped per street (23 ST / 22 ST as N-S; I ST / H ST as E-W), unioned to drop
lane dividers, and only the edges running *along* the street axis are kept (end
caps and junction boxes are dropped, so intersections stay open). The sidewalk
strips (ids 35–49) that fill the gap between each street end and the junction
are added the same way — each strip's axis is its shorter dimension — so the
curb lines continue across the sidewalk up to the intersection.

```bash
# plot the real curbs over all observed trajectories
python -m Calibration.real_boundaries

# calibrate a case with the real street boundaries enabled
python -m Calibration.run_calibration \
    --case-id "719_1_2_5_I_WB_->_23_SB_middle" --use-boundaries --plot
```

Curbs are attached to each case in its local frame (translated to the observed
start) and added to the ILQR game as soft repulsion obstacles. They describe the
scene only and are **not** part of the calibrated cost weights.

## Multi-agent (ego) calibration

The single-agent pipeline above is unchanged. A parallel pipeline calibrates the
**ego** agent while the other vehicles that were actually present during the
maneuver are **replayed from the raw data** as moving obstacles (only the ego is
optimized by ILQR; the neighbors follow their real trajectories).

Pipeline:

- `multi_agent_cases.py` — streams the raw TGSIM CSV and selects neighbors that
  (a) overlap the ego's time window and (b) come within `--radius` metres of the
  ego path. Each neighbor is translated into the ego local frame and resampled
  onto the ego time grid, producing per-step obstacle positions.
- `moving_obstacles.py` — time-varying versions of the ILQR static-obstacle and
  proximity-speed costs (`MovingObstacleRepulsionCost`,
  `MovingProximitySpeedCost`); the neighbor positions are constants at each step,
  so the problem stays single-player.
- `multi_agent_interface.py` — `EgoParameters` (behavioral weights + interaction
  weights) and `build_ego_game`/`solve_ego`, which reuse the single-agent base
  costs and add the moving-obstacle terms.
- `multi_agent_calibrate.py` — Bayesian optimization over the ego's behavioral
  **and** interaction weights (`neighbor_repulsion`, `neighbor_proximity_speed`,
  `neighbor_activation_distance`), minimizing the ego sim-vs-observed pointwise
  error exactly as in the single-agent case.

```bash
# calibrate the ego among the real co-present vehicles, with curbs + plot
python -m Calibration.run_multi_agent_calibration \
    --case-id "719_1_2_5_I_WB_->_23_SB_middle" --use-boundaries --plot

# batch-calibrate the first 10 case ids
python -m Calibration.run_multi_agent_calibration \
    --max-cases 10 --flip-y --use-boundaries --plot
```

Use `--flip-y` to run the same calibration in a globally mirrored coordinate
frame (`y := -y`). The ego data, replayed neighbors, and real curb obstacles are
all flipped consistently; output files receive a `_flip_y` suffix.

Results and plots are written to `Calibration/outputs/multi_agent_calibration/`.
With `--plot`, each case gets a spatial trajectory plot plus a `_timeseries.png`
diagnostic plot comparing observed vs simulated `x(t)`, `y(t)`, and speed.

### Scope / current assumptions

- Single-agent: one cost-weight set per agent, no inter-agent interaction.
- Multi-agent: only the ego is optimized; neighbors are **replayed** from data
  (not re-simulated), so the ego reacts to the real environment.
- Each observed case is solved in its own local frame (start translated to the
  origin); the ILQR dynamics are translation-invariant.
- Street boundaries are optional (`--use-boundaries`) and use real curbs only
  (no lane structure).

