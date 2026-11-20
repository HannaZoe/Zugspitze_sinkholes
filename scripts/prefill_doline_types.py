"""Suggest a type for each mapped doline point, only where the data are clear.

Evidence per point (EPSG:32632):
- Field data (Aug 2026): nearest field point within FIELD_M (QField GPS is good
  to a few metres). "Cave Doline" -> cave.
- Aug 2025 LiDAR: the closed depression (make_depressions.py) containing the
  point, or the nearest one within DEPR_M: the 0.1 m outline, or if that is a
  whole basin (diameter > MAX_DIAM_M) the 0.3 m, then the 1.0 m outline
  around the point. From it: d/D = depth below that outline / equivalent
  diameter. Classic doline shapes (Cvijic 1893):
  bowls are about ten times wider than deep, funnels about 2-3 times,
  well/shaft dolines deeper than wide.
  Elongation = long / short side of the minimum rotated rectangle.
  Sub-pits = separate patches of the next deeper level (>= 1 m2) inside it.

Rules, first match wins, everything else stays empty (no guessing):
  cave             field says cave, or d/D >= 1 (deeper than wide)
  multi-sinkhole   >= 2 sub-pits and this is the only mapped point in the outline
  sinkhole - elongated  elongation >= 3
  sinkhole - funnel     0.25 <= d/D < 1
  sinkhole - bowl       d/D < 0.12
Between 0.12 and 0.25, without a depression, or where field and LiDAR disagree
(field depth/width says funnel but LiDAR says bowl, or vice versa): left empty.

Output: data/processed/doline_type_suggestions.csv (id, type, basis); written
into the point layer by the QGIS step (only into empty `type` fields).
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from master_thesis.paths import FIELD_DATA, MANUAL, PROCESSED

CRS = "EPSG:32632"
FIELD_M = 5.0
DEPR_M = 2.0
MAX_DIAM_M = 30.0  # field widths are 1.6-16 m; anything bigger is a basin
LEVELS = [0.1, 0.3, 1.0]
OUT = PROCESSED / "doline_type_suggestions.csv"


def shape_class(d_over_d: float) -> str | None:
    if d_over_d >= 1:
        return "cave"
    if d_over_d >= 0.25:
        return "sinkhole - funnel"
    if d_over_d < 0.12:
        return "sinkhole - bowl"
    return None


def elongation(geom) -> float:
    rect = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)
    sides = np.hypot(*np.diff(rect[:3], axis=0).T)
    return float(sides.max() / max(sides.min(), 1e-6))


def depression_at(pt, dep: gpd.GeoDataFrame):
    """Smallest-level outline around the point that is still doline-sized."""
    for k, level in enumerate(LEVELS):
        lv = dep[dep.level_m == level]
        dist = lv.distance(pt)
        if not len(lv) or dist.min() > DEPR_M:
            return None
        row = lv.loc[dist.idxmin()]
        diam = 2 * np.sqrt(row.geometry.area / np.pi)
        if diam <= MAX_DIAM_M:
            deeper = LEVELS[k + 1] if k + 1 < len(LEVELS) else None
            sub = dep[(dep.level_m == deeper) & (dep.area_m2 >= 1)] if deeper else dep.iloc[:0]
            n_sub = int(sub.within(row.geometry.buffer(0.01)).sum())
            return row.geometry, row.max_depth_m - level, diam, n_sub
    return None


def main() -> None:
    pts = gpd.read_file(MANUAL / "sinkhole_points.gpkg").to_crs(CRS)
    dep = gpd.read_file(PROCESSED / "depressions" / "depressions.gpkg", layer="20250820")
    field = gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").to_crs(CRS)

    near_field = gpd.sjoin_nearest(pts[["id", "geometry"]], field[["type", "width_m", "length_m",
                                   "depth_m", "geometry"]], max_distance=FIELD_M)  # fmt: skip
    near_field = near_field[~near_field.index.duplicated()]

    rows = []
    for i, p in pts.iterrows():
        typ, why = None, []
        f = near_field.loc[i] if i in near_field.index else None
        if f is not None and str(f["type"]) == "Cave Doline":
            typ, why = "cave", ["field: Cave Doline"]
        found = depression_at(p.geometry, dep) if typ is None else None
        if found is not None:
            outline, depth, diam, n_pits = found
            dd = depth / diam
            n_pts = int(pts.within(outline.buffer(0.01)).sum())
            elong = elongation(outline)
            why = [f"LiDAR depth {depth:.1f} m, diam {diam:.1f} m, d/D {dd:.2f}",
                   f"elong {elong:.1f}, sub-pits {n_pits}"]  # fmt: skip
            if dd >= 1:
                typ = "cave"
            elif n_pits >= 2 and n_pts == 1:
                typ = "multi-sinkhole"
            elif elong >= 3:
                typ = "sinkhole - elongated"
            else:
                typ = shape_class(dd)
            # field depth/width as a cross-check for funnel vs bowl
            if f is not None and typ in ("sinkhole - funnel", "sinkhole - bowl"):
                dims = [v for v in (f["width_m"], f["length_m"]) if pd.notna(v)]
                width = float(np.mean(dims)) if dims else np.nan
                if np.isfinite(width) and np.isfinite(f["depth_m"]) and width > 0:
                    field_typ = shape_class(f["depth_m"] / width)
                    why.append(f"field d/D {f['depth_m'] / width:.2f}")
                    if field_typ not in (None, typ, "cave"):
                        typ = None  # field and LiDAR disagree: leave it to Hanna
        rows.append({"id": int(p["id"]), "type": typ, "basis": "; ".join(why) or None})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out["type"].value_counts(dropna=False).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
