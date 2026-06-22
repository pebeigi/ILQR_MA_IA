from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv")
OUTPUT_DIR = Path("outputs/left_turn_movements")
VEHICLE_TYPE_CODES = (4,)
VEHICLE_TYPE_NAMES = {
    3: "passenger car",
    4: "AV",
    5: "motorcycle",
    6: "bus",
    7: "truck",
}
METERS_PER_PIXEL = 0.0186613838586
BACKGROUND_ALPHA = 0.5
IMAGE_ORIGIN = "upper"


@dataclass(frozen=True)
class LaneRun:
    label: str
    lane: int
    start_pos: int
    end_pos: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class MovementRule:
    intersection_id: int
    intersection_name: str
    movement_name: str
    approach_group: str
    approach_label: str
    exit_group: str
    exit_label: str


RULES: tuple[MovementRule, ...] = (
    MovementRule(
        1,
        "I ST. & 23 ST.",
        "23 SB upper -> I EB",
        "23 ST. southbound upper",
        "23_SB_UPPER",
        "I ST. eastbound",
        "I_EB",
    ),
    MovementRule(
        1,
        "I ST. & 23 ST.",
        "23 SB upper -> I WB",
        "23 ST. southbound upper",
        "23_SB_UPPER",
        "I ST. westbound",
        "I_WB",
    ),
    MovementRule(
        1,
        "I ST. & 23 ST.",
        "I WB -> 23 SB middle",
        "I ST. westbound",
        "I_WB",
        "23 ST. southbound middle",
        "23_SB_MIDDLE",
    ),
    MovementRule(
        1,
        "I ST. & 23 ST.",
        "I WB -> 23 NB middle",
        "I ST. westbound",
        "I_WB",
        "23 ST. northbound middle",
        "23_NB_MIDDLE",
    ),
    MovementRule(
        2,
        "I ST. & 22 ST.",
        "22 NB middle -> I WB",
        "22 ST. northbound middle",
        "22_NB_MIDDLE",
        "I ST. westbound",
        "I_WB",
    ),
    MovementRule(
        2,
        "I ST. & 22 ST.",
        "22 NB middle -> I EB",
        "22 ST. northbound middle",
        "22_NB_MIDDLE",
        "I ST. eastbound",
        "I_EB",
    ),
    MovementRule(
        2,
        "I ST. & 22 ST.",
        "I38 -> I35",
        "lane_kf 38 approach",
        "I38_APPROACH",
        "lane_kf 35 exit",
        "I35_APPROACH",
    ),
    MovementRule(
        2,
        "I ST. & 22 ST.",
        "I35 -> I WB",
        "lane_kf 35 approach",
        "I35_APPROACH",
        "I ST. westbound",
        "I_WB",
    ),
    MovementRule(
        2,
        "I ST. & 22 ST.",
        "I35 -> I EB",
        "lane_kf 35 approach",
        "I35_APPROACH",
        "I ST. eastbound",
        "I_EB",
    ),
    MovementRule(
        3,
        "H ST. & 23 ST.",
        "23 NB down -> H WB left",
        "23 ST. northbound down",
        "23_NB_DOWN",
        "H ST. westbound left",
        "H_WB_LEFT",
    ),
    MovementRule(
        3,
        "H ST. & 23 ST.",
        "23 NB down -> H EB left",
        "23 ST. northbound down",
        "23_NB_DOWN",
        "H ST. eastbound left",
        "H_EB_LEFT",
    ),
    MovementRule(
        3,
        "H ST. & 23 ST.",
        "23 SB middle -> H EB right",
        "23 ST. southbound middle",
        "23_SB_MIDDLE",
        "H ST. eastbound right",
        "H_EB_RIGHT",
    ),
    MovementRule(
        3,
        "H ST. & 23 ST.",
        "23 SB middle -> H WB right",
        "23 ST. southbound middle",
        "23_SB_MIDDLE",
        "H ST. westbound right",
        "H_WB_RIGHT",
    ),
    MovementRule(
        3,
        "H ST. & 23 ST.",
        "H EB left -> 23 NB middle",
        "H ST. eastbound left",
        "H_EB_LEFT",
        "23 ST. northbound middle",
        "23_NB_MIDDLE",
    ),
    MovementRule(
        3,
        "H ST. & 23 ST.",
        "H WB left -> 23 NB middle",
        "H ST. westbound left",
        "H_WB_LEFT",
        "23 ST. northbound middle",
        "23_NB_MIDDLE",
    ),
    MovementRule(
        3,
        "H ST. & 23 ST.",
        "H WB right -> 23 SB down",
        "H ST. westbound right",
        "H_WB_RIGHT",
        "23 ST. southbound down",
        "23_SB_DOWN",
    ),
    MovementRule(
        4,
        "H ST. & 22 ST.",
        "I44 -> 22 NB middle",
        "lane_kf 44 approach",
        "I44_APPROACH",
        "22 ST. northbound middle",
        "22_NB_MIDDLE",
    ),
    MovementRule(
        4,
        "H ST. & 22 ST.",
        "22 NB down -> H WB right",
        "22 ST. northbound down",
        "22_NB_DOWN",
        "H ST. westbound right",
        "H_WB_RIGHT",
    ),
    MovementRule(
        4,
        "H ST. & 22 ST.",
        "22 NB down -> H EB right",
        "22 ST. northbound down",
        "22_NB_DOWN",
        "H ST. eastbound right",
        "H_EB_RIGHT",
    ),
    MovementRule(
        4,
        "H ST. & 22 ST.",
        "H EB right -> 22 NB middle",
        "H ST. eastbound right",
        "H_EB_RIGHT",
        "22 ST. northbound middle",
        "22_NB_MIDDLE",
    ),
)


ROAD_GROUPS = {
    5: "23_NB_UPPER",
    8: "23_NB_UPPER",
    11: "23_NB_UPPER",
    6: "23_NB_MIDDLE",
    9: "23_NB_MIDDLE",
    12: "23_NB_MIDDLE",
    7: "23_NB_DOWN",
    10: "23_NB_DOWN",
    13: "23_NB_DOWN",
    14: "23_SB_UPPER",
    17: "23_SB_UPPER",
    20: "23_SB_UPPER",
    15: "23_SB_MIDDLE",
    18: "23_SB_MIDDLE",
    21: "23_SB_MIDDLE",
    16: "23_SB_DOWN",
    19: "23_SB_DOWN",
    22: "23_SB_DOWN",
    23: "22_NB_MIDDLE",
    25: "22_NB_MIDDLE",
    27: "22_NB_MIDDLE",
    24: "22_NB_DOWN",
    26: "22_NB_DOWN",
    28: "22_NB_DOWN",
    29: "I_EB",
    30: "I_WB",
    31: "H_WB_LEFT",
    32: "H_WB_RIGHT",
    33: "H_EB_LEFT",
    34: "H_EB_RIGHT",
    35: "I35_APPROACH",
    38: "I38_APPROACH",
    43: "22_NB_DOWN",
    44: "I44_APPROACH",
}

JUNCTION_ZONES = {
    1: frozenset({1, 39, 40, 41}),
    2: frozenset({2, 35, 36, 37, 38}),
    3: frozenset({3, 46, 47, 48, 49}),
    4: frozenset({4, 42, 43, 44, 45}),
}

LANE_TO_JUNCTION = {
    lane: f"J{intersection_id}"
    for intersection_id, lanes in JUNCTION_ZONES.items()
    for lane in lanes
}

LABEL_TO_LANES: dict[str, tuple[int, ...]] = {}
for lane, label in ROAD_GROUPS.items():
    LABEL_TO_LANES.setdefault(label, tuple())
    LABEL_TO_LANES[label] = tuple(sorted((*LABEL_TO_LANES[label], lane)))


def format_lanes(label: str) -> str:
    lanes = LABEL_TO_LANES.get(label)
    if lanes is None:
        return label
    return ",".join(str(lane) for lane in lanes)


def movement_legend_label(rule: MovementRule, case_count: int) -> str:
    return (
        f"{rule.movement_name} "
        f"[from {format_lanes(rule.approach_label)} -> to {format_lanes(rule.exit_label)}] "
        f"(n={case_count})"
    )


def lane_label(lane: int) -> str:
    return ROAD_GROUPS.get(lane) or LANE_TO_JUNCTION.get(lane) or f"OTHER_{lane}"


def build_lane_runs(vehicle: pd.DataFrame) -> list[LaneRun]:
    lanes = vehicle["lane_kf"].to_numpy()
    times = vehicle["time"].to_numpy()
    labels = [lane_label(int(lane)) for lane in lanes]
    runs: list[LaneRun] = []
    run_start = 0

    for pos in range(1, len(vehicle)):
        if labels[pos] != labels[run_start]:
            runs.append(
                LaneRun(
                    label=labels[run_start],
                    lane=int(lanes[run_start]),
                    start_pos=run_start,
                    end_pos=pos - 1,
                    start_time=float(times[run_start]),
                    end_time=float(times[pos - 1]),
                )
            )
            run_start = pos

    runs.append(
        LaneRun(
            label=labels[run_start],
            lane=int(lanes[run_start]),
            start_pos=run_start,
            end_pos=len(vehicle) - 1,
            start_time=float(times[run_start]),
            end_time=float(times[-1]),
        )
    )
    return runs


def extract_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_records: list[dict[str, object]] = []
    trajectory_segments: list[pd.DataFrame] = []

    for vehicle_id, vehicle in df.groupby("id", sort=False):
        vehicle = vehicle.sort_values("time").reset_index(drop=True)
        runs = build_lane_runs(vehicle)

        for rule in RULES:
            search_start_idx = 0
            while search_start_idx < len(runs):
                approach_idx = next(
                    (idx for idx in range(search_start_idx, len(runs)) if runs[idx].label == rule.approach_label),
                    None,
                )
                if approach_idx is None:
                    break

                exit_idx = next(
                    (idx for idx in range(approach_idx + 1, len(runs)) if runs[idx].label == rule.exit_label),
                    None,
                )
                if exit_idx is None:
                    break

                approach_run = runs[approach_idx]
                exit_run = runs[exit_idx]
                segment = vehicle.iloc[approach_run.start_pos : exit_run.end_pos + 1].copy()
                junction_lanes = segment.loc[
                    segment["lane_kf"].isin(JUNCTION_ZONES[rule.intersection_id]), "lane_kf"
                ].drop_duplicates()
                junction_runs = [
                    run for run in runs[approach_idx + 1 : exit_idx] if run.label == f"J{rule.intersection_id}"
                ]

                case_id = (
                    f"{vehicle_id}_{rule.intersection_id}_{approach_idx}_{exit_idx}_"
                    f"{rule.movement_name.replace(' ', '_')}"
                )
                segment["case_id"] = case_id
                segment["intersection_id"] = rule.intersection_id
                segment["intersection_name"] = rule.intersection_name
                segment["movement_name"] = rule.movement_name
                trajectory_segments.append(segment)

                case_records.append(
                    {
                        "case_id": case_id,
                        "id": vehicle_id,
                        "intersection_id": rule.intersection_id,
                        "intersection_name": rule.intersection_name,
                        "movement_name": rule.movement_name,
                        "approach_group": rule.approach_group,
                        "approach_lane_kf": approach_run.lane,
                        "junction_lane_kf_first": junction_runs[0].lane if junction_runs else pd.NA,
                        "junction_lane_kf_sequence": ",".join(map(str, junction_lanes)),
                        "exit_group": rule.exit_group,
                        "exit_lane_kf": exit_run.lane,
                        "start_time": approach_run.start_time,
                        "intersection_entry_time": junction_runs[0].start_time if junction_runs else pd.NA,
                        "intersection_exit_time": junction_runs[-1].end_time if junction_runs else pd.NA,
                        "end_time": exit_run.end_time,
                        "duration_seconds": exit_run.end_time - approach_run.start_time,
                        "matching_method": "approach_before_exit",
                        "type_most_common": int(vehicle["type_most_common"].mode().iat[0]),
                    }
                )
                search_start_idx = exit_idx + 1

    cases = pd.DataFrame(case_records)
    if trajectory_segments:
        trajectories = pd.concat(trajectory_segments, ignore_index=True)
    else:
        trajectories = pd.DataFrame()
    return cases, trajectories


def filter_to_intersection_zone(trajectories: pd.DataFrame) -> pd.DataFrame:
    if trajectories.empty:
        return trajectories.copy()

    filtered_segments = []
    for intersection_id, intersection_trajectories in trajectories.groupby("intersection_id", sort=False):
        zone_lanes = JUNCTION_ZONES[int(intersection_id)]
        filtered_segments.append(intersection_trajectories[intersection_trajectories["lane_kf"].isin(zone_lanes)])

    if not filtered_segments:
        return trajectories.iloc[0:0].copy()

    return pd.concat(filtered_segments, ignore_index=True)


def add_reference_background(ax: plt.Axes, reference_image_path: Path | None) -> tuple[float | None, float | None]:
    if reference_image_path is None:
        return None, None

    image = mpimg.imread(reference_image_path)
    image_height_px, image_width_px = image.shape[:2]
    image_width_m = image_width_px * METERS_PER_PIXEL
    image_height_m = image_height_px * METERS_PER_PIXEL
    ax.imshow(
        image,
        extent=[0, image_width_m, 0, image_height_m],
        # Keep the reference image in its original orientation, then flip
        # trajectory y-values into that image coordinate frame before plotting.
        origin=IMAGE_ORIGIN,
        alpha=BACKGROUND_ALPHA,
    )
    return image_width_m, image_height_m


def y_for_plot(y_values: pd.Series, image_height_m: float | None) -> pd.Series:
    if image_height_m is None:
        return y_values
    return image_height_m - y_values


def plot_intersection(
    intersection_id: int,
    cases: pd.DataFrame,
    trajectories: pd.DataFrame,
    reference_image_path: Path | None,
) -> Path | None:
    intersection_cases = cases[cases["intersection_id"] == intersection_id]
    if intersection_cases.empty:
        return None

    intersection_name = intersection_cases["intersection_name"].iat[0]
    intersection_trajectories = trajectories[trajectories["intersection_id"] == intersection_id]

    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    image_width_m, image_height_m = add_reference_background(ax, reference_image_path)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    movements = sorted(intersection_cases["movement_name"].unique())
    rules_by_movement = {
        rule.movement_name: rule
        for rule in RULES
        if rule.intersection_id == intersection_id
    }

    for color_idx, movement in enumerate(movements):
        movement_cases = intersection_cases[intersection_cases["movement_name"] == movement]
        color = color_cycle[color_idx % len(color_cycle)]

        for case_id in movement_cases["case_id"]:
            segment = intersection_trajectories[intersection_trajectories["case_id"] == case_id]
            y_values = y_for_plot(segment["yloc_kf"], image_height_m)
            ax.plot(segment["xloc_kf"], y_values, color=color, alpha=0.22, linewidth=0.9)

        first_segments = intersection_trajectories[
            intersection_trajectories["case_id"].isin(movement_cases["case_id"])
        ].groupby("case_id", sort=False).head(1)
        last_segments = intersection_trajectories[
            intersection_trajectories["case_id"].isin(movement_cases["case_id"])
        ].groupby("case_id", sort=False).tail(1)
        first_y = y_for_plot(first_segments["yloc_kf"], image_height_m)
        last_y = y_for_plot(last_segments["yloc_kf"], image_height_m)
        ax.scatter(first_segments["xloc_kf"], first_y, color=color, s=8, alpha=0.35, marker="o")
        ax.scatter(last_segments["xloc_kf"], last_y, color=color, s=10, alpha=0.45, marker="x")
        rule = rules_by_movement[movement]
        ax.plot([], [], color=color, linewidth=3, label=movement_legend_label(rule, len(movement_cases)))

    ax.set_title(f"Extracted left-turn trajectories: {intersection_id} - {intersection_name}")
    ax.set_xlabel("xloc_kf (m)")
    ax.set_ylabel("image-space y (m)" if image_height_m is not None else "yloc_kf (m)")
    if image_width_m is not None and image_height_m is not None:
        ax.set_xlim(0, image_width_m)
        ax.set_ylim(0, image_height_m)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    suffix = "_on_reference" if reference_image_path is not None else ""
    output_path = OUTPUT_DIR / f"intersection_{intersection_id}_left_turn_trajectories{suffix}.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_intersection_zone(
    intersection_id: int,
    cases: pd.DataFrame,
    intersection_zone_trajectories: pd.DataFrame,
    reference_image_path: Path | None,
) -> Path | None:
    intersection_trajectories = intersection_zone_trajectories[
        intersection_zone_trajectories["intersection_id"] == intersection_id
    ]
    if intersection_trajectories.empty:
        return None

    case_ids_with_points = set(intersection_trajectories["case_id"].unique())
    intersection_cases = cases[
        (cases["intersection_id"] == intersection_id) & (cases["case_id"].isin(case_ids_with_points))
    ]
    if intersection_cases.empty:
        return None

    intersection_name = intersection_cases["intersection_name"].iat[0]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    image_width_m, image_height_m = add_reference_background(ax, reference_image_path)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    movements = sorted(intersection_cases["movement_name"].unique())
    rules_by_movement = {
        rule.movement_name: rule
        for rule in RULES
        if rule.intersection_id == intersection_id
    }

    for color_idx, movement in enumerate(movements):
        movement_cases = intersection_cases[intersection_cases["movement_name"] == movement]
        movement_case_ids = set(movement_cases["case_id"])
        movement_trajectories = intersection_trajectories[
            intersection_trajectories["case_id"].isin(movement_case_ids)
        ]
        color = color_cycle[color_idx % len(color_cycle)]

        for case_id, segment in movement_trajectories.groupby("case_id", sort=False):
            ax.plot(
                segment["xloc_kf"],
                y_for_plot(segment["yloc_kf"], image_height_m),
                color=color,
                alpha=0.28,
                linewidth=1.0,
            )

        first_segments = movement_trajectories.groupby("case_id", sort=False).head(1)
        last_segments = movement_trajectories.groupby("case_id", sort=False).tail(1)
        ax.scatter(
            first_segments["xloc_kf"],
            y_for_plot(first_segments["yloc_kf"], image_height_m),
            color=color,
            s=8,
            alpha=0.4,
            marker="o",
        )
        ax.scatter(
            last_segments["xloc_kf"],
            y_for_plot(last_segments["yloc_kf"], image_height_m),
            color=color,
            s=10,
            alpha=0.5,
            marker="x",
        )
        rule = rules_by_movement[movement]
        ax.plot([], [], color=color, linewidth=3, label=movement_legend_label(rule, len(movement_cases)))

    zone_lanes = ",".join(str(lane) for lane in sorted(JUNCTION_ZONES[intersection_id]))
    ax.set_title(
        f"Intersection-zone left-turn trajectories: {intersection_id} - {intersection_name}\n"
        f"kept lane_kf: {zone_lanes}"
    )
    ax.set_xlabel("xloc_kf (m)")
    ax.set_ylabel("image-space y (m)" if image_height_m is not None else "yloc_kf (m)")
    if image_width_m is not None and image_height_m is not None:
        ax.set_xlim(0, image_width_m)
        ax.set_ylim(0, image_height_m)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    suffix = "_on_reference" if reference_image_path is not None else ""
    output_path = OUTPUT_DIR / f"intersection_{intersection_id}_left_turn_intersection_zone{suffix}.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_all_vehicle_trajectories(df: pd.DataFrame, reference_image_path: Path | None) -> Path:
    fig, ax = plt.subplots(figsize=(12, 10), dpi=180)
    image_width_m, image_height_m = add_reference_background(ax, reference_image_path)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color_idx, type_code in enumerate(VEHICLE_TYPE_CODES):
        type_df = df[df["type_most_common"] == type_code]
        if type_df.empty:
            continue

        color = color_cycle[color_idx % len(color_cycle)]
        for _, vehicle in type_df.groupby("id", sort=False):
            ax.plot(
                vehicle["xloc_kf"],
                y_for_plot(vehicle["yloc_kf"], image_height_m),
                color=color,
                linestyle="-",
                marker=None,
                alpha=0.18,
                linewidth=0.8,
            )

        vehicle_count = type_df["id"].nunique()
        ax.plot(
            [],
            [],
            color=color,
            linewidth=2,
            label=f"{type_code}: {VEHICLE_TYPE_NAMES[type_code]} (n={vehicle_count})",
        )

    ax.set_title("All AV trajectories by vehicle ID")
    ax.set_xlabel("xloc_kf (m)")
    ax.set_ylabel("image-space y (m)" if image_height_m is not None else "yloc_kf (m)")
    if image_width_m is not None and image_height_m is not None:
        ax.set_xlim(0, image_width_m)
        ax.set_ylim(0, image_height_m)
    ax.axis("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    suffix = "_on_reference" if reference_image_path is not None else ""
    output_path = OUTPUT_DIR / f"all_motor_vehicle_trajectories{suffix}.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract motor-vehicle left-turn trajectories and plot one figure per intersection."
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help=(
            "Optional background image to overlay below trajectories. "
            f"Uses {METERS_PER_PIXEL} meters per pixel, {BACKGROUND_ALPHA:.0%} transparency, "
            f"and image origin={IMAGE_ORIGIN!r}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_image_path = args.reference_image
    if reference_image_path is not None and not reference_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

    df = pd.read_csv(
        DATA_PATH,
        usecols=["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "type_most_common"],
    )
    df = df[df["type_most_common"].isin(VEHICLE_TYPE_CODES)]
    df = df.sort_values(["id", "time"]).reset_index(drop=True)

    cases, trajectories = extract_cases(df)
    intersection_zone_trajectories = filter_to_intersection_zone(trajectories)
    cases_path = OUTPUT_DIR / "left_turn_cases.csv"
    trajectories_path = OUTPUT_DIR / "left_turn_trajectory_points.csv"
    cases.to_csv(cases_path, index=False)
    trajectories.to_csv(trajectories_path, index=False)
    intersection_zone_path = OUTPUT_DIR / "left_turn_intersection_zone_points.csv"
    intersection_zone_trajectories.to_csv(intersection_zone_path, index=False)

    for intersection_id in sorted(JUNCTION_ZONES):
        intersection_zone_file = OUTPUT_DIR / f"intersection_{intersection_id}_left_turn_intersection_zone_points.csv"
        intersection_zone_trajectories[
            intersection_zone_trajectories["intersection_id"] == intersection_id
        ].to_csv(intersection_zone_file, index=False)

    if cases.empty:
        summary = pd.DataFrame(columns=["intersection_id", "intersection_name", "movement_name", "case_count"])
    else:
        summary = (
            cases.groupby(["intersection_id", "intersection_name", "movement_name"])
            .size()
            .reset_index(name="case_count")
            .sort_values(["intersection_id", "movement_name"])
        )
    summary_path = OUTPUT_DIR / "left_turn_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_paths = []
    for intersection_id in sorted({rule.intersection_id for rule in RULES}):
        plot_path = plot_intersection(intersection_id, cases, trajectories, reference_image_path)
        if plot_path is not None:
            plot_paths.append(plot_path)
        zone_plot_path = plot_intersection_zone(
            intersection_id,
            cases,
            intersection_zone_trajectories,
            reference_image_path,
        )
        if zone_plot_path is not None:
            plot_paths.append(zone_plot_path)
    plot_paths.append(plot_all_vehicle_trajectories(df, reference_image_path))

    print(f"Wrote {len(cases)} cases to {cases_path}")
    print(f"Wrote trajectory points to {trajectories_path}")
    print(f"Wrote intersection-zone trajectory points to {intersection_zone_path}")
    print(f"Wrote summary to {summary_path}")
    for plot_path in plot_paths:
        print(f"Wrote plot {plot_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
