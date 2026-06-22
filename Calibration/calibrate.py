from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .features import FeatureConfig, write_feature_outputs
    from .objective import score_movement_targets
except ImportError:  # pragma: no cover - supports `python Calibration/calibrate.py`
    from features import FeatureConfig, write_feature_outputs
    from objective import score_movement_targets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_POINTS = (
    PROJECT_ROOT
    / "Data_Preparation"
    / "outputs"
    / "left_turn_movements"
    / "left_turn_intersection_zone_points.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Calibration" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build observed calibration targets from prepared left-turn trajectory data. "
            "Optionally compare those targets against a simulated movement-target CSV."
        )
    )
    parser.add_argument(
        "--trajectory-points",
        type=Path,
        default=DEFAULT_TRAJECTORY_POINTS,
        help="Prepared trajectory-points CSV from Data_Preparation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where calibration outputs are written.",
    )
    parser.add_argument(
        "--min-points-per-case",
        type=int,
        default=3,
        help="Drop observed cases with fewer points than this.",
    )
    parser.add_argument(
        "--simulated-targets",
        type=Path,
        default=None,
        help="Optional simulated movement-target CSV to score against observed targets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatureConfig(min_points_per_case=args.min_points_per_case)

    case_features_path, movement_targets_path = write_feature_outputs(
        trajectory_points_path=args.trajectory_points,
        output_dir=args.output_dir,
        config=config,
    )

    print(f"Wrote observed case features: {case_features_path}")
    print(f"Wrote observed movement targets: {movement_targets_path}")

    if args.simulated_targets is not None:
        observed_targets = pd.read_csv(movement_targets_path)
        simulated_targets = pd.read_csv(args.simulated_targets)
        score = score_movement_targets(observed_targets, simulated_targets)
        score_path = args.output_dir / "objective_score.json"
        score_path.write_text(json.dumps(score, indent=2), encoding="utf-8")
        print(f"Wrote objective score: {score_path}")
        print(f"Objective score: {score['score']:.6g}")


if __name__ == "__main__":
    main()

