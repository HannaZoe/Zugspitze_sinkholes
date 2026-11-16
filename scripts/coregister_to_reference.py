"""Co-register every epoch to the 20250820 LiDAR DSM (star topology, no chains).

Works on the common 20 cm grid from resample_common_grid.py (EPSG:32632).
Each processing chain (one flight + one software run) is registered once via
its DSM; the resulting horizontal shift is then applied to the orthos that
came out of the same chain. Pipelines follow xdem's recommendations and
docs/thesis/lit_uav_coregistration.md:

- ppk     (Terra PPK LiDAR/RGB):        NuthKaab
- ppk_along (20241025, east-end bend):  NuthKaab + along-strip bias (binned median
                                        along the flight strip; notebook 07 holdout:
                                        NMAD 0.050 -> 0.030 m, bend +0.18 -> +0.02 m)
- rtk     (Agisoft RTK, no GCPs):       NuthKaab + Deramp(1)   (RTK tilt)
- rtk_big (20260827, metres off in z):  VerticalShift + Deramp(1) + NuthKaab
- datum   (DGM1, DHHN2016 vs ellipsoid): VerticalShift + NuthKaab

Stable ground = valid in both, >5 m from data edges, slope < 40 deg, outside
buffers around mapped features, not snow in the epoch's own RGB ortho. Each
fit runs twice: the second pass drops pixels whose residual is beyond
3 NMAD of the first, so leftover snow/changes don't drag the solution; the
first pass already starts from raw-dh inliers. A correction that increases
the stable-ground NMAD is rejected in favour of a vertical shift only
(column `applied` in the report).

The DGM1 is fitted over the LiDAR footprint and the correction applied to
the whole mosaic. Every chain is processed inside its own data footprint to
keep memory sane; outputs stay on the full grid.

Output: data/interim/coregistered_20cm/ (same filenames; DGM1 ends up in
ellipsoidal heights like the UAV data) plus coregistration_report.csv with
the shifts and before/after median/NMAD on stable ground.
The DOP20 orthophotos have no DSM and are not handled here.

Rerun single chains without touching anything else (e.g. while the orthos
are open in QGIS): `--only 20241025_lidar 20241025_rgb --dsm-only`. The report
rows of those chains are then replaced, the others kept.
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xdem
from rasterio.warp import Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from master_thesis.image_alignment import shift_raster
from master_thesis.paths import FIELD_DATA, INTERIM, MANUAL
from master_thesis.stable_ground import (
    erode_valid,
    feature_mask,
    inliers_of,
    median,
    nmad,
    slope_deg,
    snow_mask,
)  # fmt: skip

CRS = rasterio.crs.CRS.from_epsg(32632)
RES = 0.2
IN_DIR = INTERIM / "common_grid_20cm"
OUT_DIR = INTERIM / "coregistered_20cm"
REF = "20250820_terra_ppk_lidar_dsm"
MAX_SLOPE = 40
STRIP_ANGLE = -10  # deg, direction of the 20241025 flight strip (from east, ccw)

# chain: (dsm, products sharing its georeferencing, pipeline, ortho for the snow mask)
CHAINS = {
    "dgm1": ("bavaria_dgm1", [], "datum", None),
    "20241025_lidar": (
        "20241025_terra_ppk_lidar_dsm", [], "ppk_along", "20241025_terra_ppk_rgb_om",
    ),
    "20241025_rgb": (
        "20241025_terra_ppk_rgb_dsm", ["20241025_terra_ppk_rgb_om"], "ppk_along",
        "20241025_terra_ppk_rgb_om",
    ),
    "20241025_ms": (
        "20241025_agisoft_rtk_ms_dsm", ["20241025_agisoft_rtk_ms_om"], "rtk",
        "20241025_terra_ppk_rgb_om",
    ),
    "20250604_lidar": ("20250604_terra_ppk_lidar_dsm", [], "ppk", "20250604_terra_ppk_rgb_om"),
    "20250604_rgb": (
        "20250604_terra_ppk_rgb_dsm", ["20250604_terra_ppk_rgb_om"], "ppk",
        "20250604_terra_ppk_rgb_om",
    ),
    "20250604_ms": (
        "20250604_agisoft_rtk_ms_dsm", ["20250604_agisoft_rtk_ms_om"], "rtk",
        "20250604_terra_ppk_rgb_om",
    ),
    "20250820_rgb": (
        "20250820_terra_ppk_rgb_dsm", ["20250820_terra_ppk_rgb_om"], "ppk", None,
    ),
    "20260827_3m": (
        "20260827_terra_rtk_dsm_20cm_local_coregistration",
        ["20260827_terra_rtk_rgb_om_20cm_local_coregistration",
         "20260827_terra_rtk_ms_om_20cm_local_coregistration"],
        "rtk_big", None,
    ),
    "20260827_3t": (
        "20260827_terra_rtk_thermal_dsm_20cm_local_coregistration",
        ["20260827_terra_rtk_thermal_ir_om_20cm_local_coregistration",
         "20260827_terra_rtk_thermal_rgb_om_20cm_local_coregistration"],
        "rtk_big", None,
    ),
}  # fmt: skip
FOOTPRINT_PAD = 250  # px, 50 m around the data


def pipeline(kind: str) -> xdem.coreg.Coreg:
    nk, vs = xdem.coreg.NuthKaab(), xdem.coreg.VerticalShift()
    ramp = xdem.coreg.Deramp(poly_order=1)
    along = xdem.coreg.DirectionalBias(angle=STRIP_ANGLE, fit_or_bin="bin", bin_sizes=30)
    pipes = {
        "ppk": nk, "ppk_along": nk + along, "rtk": nk + ramp, "rtk_big": vs + ramp + nk,
        "datum": vs + nk,
    }  # fmt: skip
    return pipes.get(kind, vs)  # "vertical": fallback when the full pipeline makes it worse


def read(name: str, window: Window | None = None, folder=IN_DIR) -> tuple[np.ndarray, dict]:
    with rasterio.open(folder / f"{name}.tif") as src:
        arr = src.read(window=window)
        if src.nodata is not None and not np.isnan(src.nodata) and arr.dtype.kind == "f":
            arr[arr == src.nodata] = np.nan
        return arr, src.profile


def write(name: str, arr: np.ndarray, profile: dict, window: Window | None = None) -> None:
    """Write (a window of) a full-grid raster; the rest stays nodata."""
    nodata = profile.get("nodata")
    if arr.dtype.kind == "f" and nodata is not None and not np.isnan(nodata):
        arr = np.where(np.isnan(arr), nodata, arr)  # NaN would otherwise count as valid data
    with rasterio.open(OUT_DIR / f"{name}.tif", "w", **profile) as dst:
        dst.write(arr, window=window)
        dst.build_overviews([2, 4, 8, 16], Resampling.average)


def footprint(names: list[str], shape: tuple[int, int]) -> Window:
    """Bounding window of valid data over several rasters, padded."""
    rows, cols = [], []
    for name in names:
        with rasterio.open(IN_DIR / f"{name}.tif") as src:
            f = 20
            valid = src.dataset_mask(out_shape=(src.height // f, src.width // f)) > 0
        r, c = np.nonzero(valid)
        rows += [r.min() * f, (r.max() + 1) * f]
        cols += [c.min() * f, (c.max() + 1) * f]
    r0, c0 = max(min(rows) - FOOTPRINT_PAD, 0), max(min(cols) - FOOTPRINT_PAD, 0)
    r1, c1 = min(max(rows) + FOOTPRINT_PAD, shape[0]), min(max(cols) + FOOTPRINT_PAD, shape[1])
    return Window(c0, r0, c1 - c0, r1 - r0)


def horizontal_shift(coreg: xdem.coreg.Coreg) -> tuple[float, float]:
    steps = coreg.pipeline if isinstance(coreg, xdem.coreg.CoregPipeline) else [coreg]
    nk = next((s for s in steps if isinstance(s, xdem.coreg.NuthKaab)), None)
    if nk is None:
        return 0.0, 0.0
    m = nk.to_matrix()
    return float(m[0, 3]), float(m[1, 3])


def fit(ref, tba, mask, kind, transform):
    """Drop raw-dh outliers (snow, real change), fit, then refit without the
    residual outliers of that first fit."""
    kw = dict(transform=transform, crs=CRS, subsample=1_000_000, random_state=42)
    inliers = inliers_of(tba - ref, mask)
    coreg = pipeline(kind).fit(ref, tba, inlier_mask=inliers, **kw)
    inliers = inliers_of(coreg.apply(tba, transform=transform, crs=CRS)[0] - ref, mask)
    coreg = pipeline(kind).fit(ref, tba, inlier_mask=inliers, **kw)
    return coreg, inliers


def stable_base(ref: np.ndarray, transform, shape) -> np.ndarray:
    """Outside mapped features and not too steep, on the given (window) grid."""
    base = ~feature_mask(
        polygons=[gpd.read_file(MANUAL / "Suspicious_Areas.shp").geometry],
        points=[
            gpd.read_file(MANUAL / "Sinkholes.shp").geometry,
            gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").geometry,
        ],
        transform=transform, shape=shape, crs=CRS,
    )  # fmt: skip
    return base & (slope_deg(ref, RES) < MAX_SLOPE)


def main(only: list[str] | None = None, dsm_only: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(IN_DIR / f"{REF}.tif") as src:
        grid_profile, grid_shape = src.profile, src.shape
    if not only:
        write(REF, read(REF)[0], grid_profile)
    lidar_window = footprint([REF], grid_shape)

    rows = []
    for chain, (dsm_name, products, kind, snow_from) in CHAINS.items():
        if only and chain not in only:
            continue
        # the DGM1 is fitted where the LiDAR is but corrected everywhere
        win = lidar_window if chain == "dgm1" else footprint([dsm_name, *products], grid_shape)
        ref, ref_name = read(REF, win)[0][0], REF
        tf = window_transform(win, grid_profile["transform"])
        dsm, profile = read(dsm_name, win)
        dsm = dsm[0]

        mask = stable_base(ref, tf, ref.shape)
        mask &= erode_valid(np.isfinite(dsm) & np.isfinite(ref), RES)
        if snow_from:
            mask &= ~snow_mask(read(snow_from, win)[0], RES)
        if mask.sum() < 10_000:
            print(f"  {chain}: only {mask.sum()} stable px, skipped")
            continue

        before = dsm - ref
        used = kind
        coreg, inliers = fit(ref, dsm, mask, kind, tf)
        aligned = coreg.apply(dsm, transform=tf, crs=CRS)[0].astype("float32")
        if nmad((aligned - ref)[mask]) > nmad(before[mask]):
            # sanity rule: a correction that increases the scatter is not accepted
            used = "vertical"
            coreg, inliers = fit(ref, dsm, mask, used, tf)
            aligned = coreg.apply(dsm, transform=tf, crs=CRS)[0].astype("float32")
        dx, dy = horizontal_shift(coreg)
        after = aligned - ref
        rows.append({
            "chain": chain, "reference": ref_name, "pipeline": kind, "applied": used,
            "dx_m": dx, "dy_m": dy,
            "stable_px": int(mask.sum()), "inlier_frac": inliers.sum() / mask.sum(),
            "median_before_m": median(before[mask]), "nmad_before_m": nmad(before[mask]),
            "median_after_m": median(after[mask]), "nmad_after_m": nmad(after[mask]),
            "nmad_after_inliers_m": nmad(after[inliers]),
        })  # fmt: skip
        r = rows[-1]
        print(
            f"  {chain:16s} [{used}] dx {dx:+.2f} dy {dy:+.2f}  "
            f"NMAD {r['nmad_before_m']:.3f} -> {r['nmad_after_m']:.3f} m",
            flush=True,
        )
        del before, after, ref, mask, inliers

        if chain == "dgm1":  # apply to the whole DGM1, not just the LiDAR window
            full = read(dsm_name)[0][0]
            aligned = coreg.apply(full, transform=grid_profile["transform"], crs=CRS)[0]
            write(dsm_name, aligned.astype("float32")[None], profile)
        else:
            write(dsm_name, aligned[None], profile, win)
        for p in [] if dsm_only else products:
            arr, p_profile = read(p, win)
            nodata = p_profile["nodata"] if p_profile["nodata"] is not None else 0
            write(p, shift_raster(arr, tf, CRS, nodata, dx, dy), p_profile, win)

    report = pd.DataFrame(rows)
    report_path = OUT_DIR / "coregistration_report.csv"
    if only and report_path.exists():
        old = pd.read_csv(report_path)
        report = pd.concat([old[~old.chain.isin(report.chain)], report], ignore_index=True)
        report = report.set_index("chain").loc[[c for c in CHAINS if c in set(report.chain)]]
        report = report.reset_index()
    report.to_csv(OUT_DIR / "coregistration_report.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=list(CHAINS), help="chains to rerun")
    parser.add_argument("--dsm-only", action="store_true", help="don't rewrite the orthos")
    args = parser.parse_args()
    main(args.only, args.dsm_only)
