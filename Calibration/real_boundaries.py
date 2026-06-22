"""Real Foggy Bottom street boundaries (clean street side-curbs) for ILQR.

Source: ``Foggy_Bottom_boundaries.txt`` — one polygon per ``lane-kf`` id in image
pixels.  Conversion to meters uses ``METERS_PER_PIXEL`` with the same orientation
as the trajectory data (verified: ``xloc_kf/yloc_kf == pixel * mpp``, no flip).

Boundary philosophy (matches the previous synthetic scene):
    * For each *street* we keep only the two long side lines that run **along**
      the street axis (the curbs).  Lane dividers are removed by unioning the
      lanes; street end-caps are removed by dropping edges perpendicular to the
      street axis.
    * Intersections / junction boxes (ids 1-4) and crosswalks (ids 35-49) are
      **excluded**, so each street is an open corridor with a gap at the
      junctions -- just like the old simulation boundaries.

``lane-kf`` id groups (from the dataset dictionary):
    1-4    intersection / junction boxes        (excluded)
    5-22   23 ST. lanes (N-S)
    23-28  22 ST. lanes (N-S)
    29-30  I ST. (E-W)
    31-34  H ST. (E-W)
    35-49  crosswalks / approach markings       (excluded)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOUNDARY_FILE = PROJECT_ROOT / "Foggy_Bottom_boundaries.txt"
METERS_PER_PIXEL = 0.0186613838586

# Street -> (lane-kf ids, axis).  axis "NS" = runs along y, "EW" = runs along x.
STREET_GROUPS: dict[str, tuple[tuple[int, ...], str]] = {
    "23 ST.": (tuple(range(5, 23)), "NS"),
    "22 ST.": (tuple(range(23, 29)), "NS"),
    "I ST.": ((29, 30), "EW"),
    "H ST.": ((31, 32, 33, 34), "EW"),
}

# Sidewalk / approach strips that fill the gap between each street end and the
# junction box.  Processed individually; each strip's axis is its shorter bbox
# dimension (it is elongated across the street it continues), so its along-axis
# side edges extend the corresponding street curbs up to the intersection.
SIDEWALK_IDS: tuple[int, ...] = tuple(range(35, 50))

_LINE_RE = re.compile(r"^\s*(\d+)\s*=\s*(\[.*\])\s*$")
_MIN_EDGE_LEN_M = 1.0


def parse_boundary_polygons(
    path: Path = DEFAULT_BOUNDARY_FILE,
    meters_per_pixel: float = METERS_PER_PIXEL,
) -> dict[int, np.ndarray]:
    """Return ``{lane_kf_id: (N, 2) polygon in meters}``."""
    polygons: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        lane_id = int(match.group(1))
        raw = match.group(2).replace("(", "[").replace(")", "]")
        pts = np.asarray(ast.literal_eval(raw), dtype=float)
        polygons[lane_id] = pts * meters_per_pixel
    return polygons


@dataclass
class StreetCurbs:
    """Curb side-lines for all streets, in meters."""

    segments: list[np.ndarray]  # each (2, 2): [[x0, y0], [x1, y1]]
    by_street: dict[str, list[np.ndarray]]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        allpts = np.vstack(self.segments)
        return (
            float(allpts[:, 0].min()),
            float(allpts[:, 1].min()),
            float(allpts[:, 0].max()),
            float(allpts[:, 1].max()),
        )


def _along_axis_edges(ring: np.ndarray, axis: str) -> list[np.ndarray]:
    """Keep ring edges that run along the street axis (the side curbs)."""
    segments: list[np.ndarray] = []
    for a, b in zip(ring[:-1], ring[1:]):
        d = b - a
        length = float(np.hypot(d[0], d[1]))
        if length < _MIN_EDGE_LEN_M:
            continue
        along = abs(d[1]) >= abs(d[0]) if axis == "NS" else abs(d[0]) >= abs(d[1])
        if along:
            segments.append(np.array([a, b], dtype=float))
    return segments


def _polygon_axis(pts: np.ndarray) -> str:
    """A strip's axis is its shorter bounding-box dimension."""
    width = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 1].max() - pts[:, 1].min())
    return "NS" if height <= width else "EW"


def build_street_curbs(
    path: Path = DEFAULT_BOUNDARY_FILE,
    meters_per_pixel: float = METERS_PER_PIXEL,
) -> StreetCurbs:
    """Extract the two side curb lines for every street (no caps, no junctions)."""
    polygons = parse_boundary_polygons(path, meters_per_pixel)

    by_street: dict[str, list[np.ndarray]] = {}
    all_segments: list[np.ndarray] = []

    for street, (ids, axis) in STREET_GROUPS.items():
        polys = []
        for lane_id in ids:
            if lane_id not in polygons:
                continue
            poly = Polygon(polygons[lane_id])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                polys.append(poly)
        if not polys:
            continue

        merged = unary_union(polys)
        blocks = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]

        street_segments: list[np.ndarray] = []
        for block in blocks:
            ring = np.asarray(block.exterior.coords, dtype=float)
            street_segments.extend(_along_axis_edges(ring, axis))

        by_street[street] = street_segments
        all_segments.extend(street_segments)

    # Sidewalk strips: extend each street's curbs across the sidewalk gap toward
    # the intersection.  Each strip is handled on its own with a per-strip axis.
    sidewalk_segments: list[np.ndarray] = []
    for lane_id in SIDEWALK_IDS:
        if lane_id not in polygons:
            continue
        pts = polygons[lane_id]
        ring = np.vstack([pts, pts[:1]]) if not np.allclose(pts[0], pts[-1]) else pts
        sidewalk_segments.extend(_along_axis_edges(ring, _polygon_axis(pts)))
    if sidewalk_segments:
        by_street["sidewalk"] = sidewalk_segments
        all_segments.extend(sidewalk_segments)

    return StreetCurbs(segments=all_segments, by_street=by_street)


def curb_points(
    curbs: StreetCurbs | None = None,
    spacing_m: float = 2.0,
) -> np.ndarray:
    """Sample obstacle points along the curb side-lines (global meter frame)."""
    if curbs is None:
        curbs = build_street_curbs()
    pts: list[np.ndarray] = []
    for seg in curbs.segments:
        length = float(np.hypot(*(seg[1] - seg[0])))
        n = max(2, int(np.ceil(length / spacing_m)))
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(seg[0] + t * (seg[1] - seg[0]))
    return np.vstack(pts) if pts else np.empty((0, 2))


# Backward-compatible alias used by callers.
def boundary_points(curbs: StreetCurbs | None = None, spacing_m: float = 2.0) -> np.ndarray:
    return curb_points(curbs, spacing_m=spacing_m)


def local_boundary_points(
    origin: np.ndarray,
    *,
    spacing_m: float = 2.0,
    radius_m: float | None = 40.0,
    curbs: StreetCurbs | None = None,
    flip_y: bool = False,
) -> np.ndarray:
    """Curb obstacle points in a case-local frame (translated by ``-origin``)."""
    pts = curb_points(curbs, spacing_m=spacing_m)
    if pts.size == 0:
        return pts
    if flip_y:
        pts = pts.copy()
        pts[:, 1] *= -1.0
    local = pts - np.asarray(origin, dtype=float).reshape(2)
    if radius_m is not None:
        keep = np.hypot(local[:, 0], local[:, 1]) <= radius_m
        local = local[keep]
    return local


def plot_boundary(
    output_path: Path,
    *,
    overlay_trajectories: bool = True,
    flip_y: bool = False,
) -> Path:
    """Plot the clean street side-curbs, optionally overlaying observed paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curbs = build_street_curbs()
    colors = {
        "23 ST.": "#383838",
        "22 ST.": "#383838",
        "I ST.": "#1f77b4",
        "H ST.": "#1f77b4",
        "sidewalk": "#e08200",
    }

    fig, ax = plt.subplots(figsize=(9, 12))
    labelled: set[str] = set()
    for street, segs in curbs.by_street.items():
        for seg in segs:
            draw_seg = seg.copy()
            if flip_y:
                draw_seg[:, 1] *= -1.0
            ax.plot(
                draw_seg[:, 0],
                draw_seg[:, 1],
                "-",
                color=colors.get(street, "#383838"),
                lw=2.0,
                label=f"{street} curbs" if street not in labelled else None,
            )
            labelled.add(street)

    if overlay_trajectories:
        try:
            import pandas as pd

            from .observed_cases import DEFAULT_TRAJECTORY_POINTS

            df = pd.read_csv(DEFAULT_TRAJECTORY_POINTS)
            first = True
            for _, seg in df.groupby("case_id", sort=False):
                y = -seg["yloc_kf"] if flip_y else seg["yloc_kf"]
                ax.plot(
                    seg["xloc_kf"], y, "-", color="#d62728", lw=0.8, alpha=0.5,
                    label="observed trajectories" if first else None,
                )
                first = False
        except Exception as exc:  # pragma: no cover
            print(f"Trajectory overlay skipped: {exc}")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    suffix = " [y flipped]" if flip_y else ""
    ax.set_title(f"Foggy Bottom street boundaries (side curbs only, open intersections){suffix}")
    ax.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "Calibration" / "outputs" / "boundaries"
    p = plot_boundary(out_dir / "real_street_boundaries.png", overlay_trajectories=True)
    pf = plot_boundary(out_dir / "real_street_boundaries_flip_y.png", overlay_trajectories=True, flip_y=True)
    print(f"Saved: {p}")
    print(f"Saved: {pf}")
    c = build_street_curbs()
    print(f"Streets: {list(c.by_street)}")
    print(f"Curb segments: {len(c.segments)}  bounds (m): {tuple(round(v, 1) for v in c.bounds)}")
    print(f"Sampled curb points @2m: {curb_points(c, 2.0).shape[0]}")
