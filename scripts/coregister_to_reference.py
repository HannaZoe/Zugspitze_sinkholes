"""Co-register every epoch to the 20250820 LiDAR DSM (star topology, no chains).

Works on the common 20 cm grid from resample_common_grid.py (EPSG:32632).
Each processing chain (one flight + one software run) is registered once via
its DSM; the resulting horizontal shift is then applied to the orthos that
came out of the same chain. Pipelines follow xdem's recommendations and
docs/thesis/lit_uav_coregistration.md:

- ppk     (Terra PPK LiDAR/RGB):        NuthKaab
- rtk     (Agisoft RTK, no GCPs):       NuthKaab + Deramp(1)   (RTK tilt)
- rtk_big (20260827, metres off in z):  VerticalShift + Deramp(1) + NuthKaab
- datum   (DGM1, DHHN2016 vs ellipsoid): VerticalShift + NuthKaab

Stable ground = valid in both, >5 m from data edges, slope < 40 deg, outside
buffers around mapped features, not snow in the epoch's own RGB ortho. Each
fit runs twice: the second pass drops pixels whose residual is beyond
3 NMAD of the first, so leftover snow/changes don't drag the solution.

Output: data/interim/coregistered_20cm/ (same filenames; DGM1 ends up in
ellipsoidal heights like the UAV data) plus coregistration_report.csv with
the shifts and before/after median/NMAD on stable ground.
The DOP20 orthophotos have no DSM and are not handled here.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xdem
from rasterio.warp import Resampling, reproject

from master_thesis.paths import FIELD_DATA, INTERIM, MANUAL
from master_thesis.stable_ground import erode_valid, feature_mask, nmad, slope_deg, snow_mask

CRS = rasterio.crs.CRS.from_epsg(32632)
RES = 0.2
IN_DIR = INTERIM / "common_grid_20cm"
OUT_DIR = INTERIM / "coregistered_20cm"
REF = "20250820_terra_ppk_lidar_dsm"
MAX_SLOPE = 40

# chain: (dsm, products sharing its georeferencing, pipeline, ortho for the snow mask)
CHAINS = {
    "20241025_lidar": ("20241025_terra_ppk_lidar_dsm", [], "ppk", "20241025_terra_ppk_rgb_om"),
    "20241025_rgb": (
        "20241025_terra_ppk_rgb_dsm", ["20241025_terra_ppk_rgb_om"], "ppk",
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
    "20250606_rgb": (
        "20250606_agisoft_rtk_rgb_dsm", ["20250606_agisoft_rtk_rgb_om"], "rtk",
        "20250606_agisoft_rtk_rgb_om",
    ),
    "20250717_rgb": (
        "20250717_agisoft_rtk_rgb_dsm", ["20250717_agisoft_rtk_rgb_om"], "rtk",
        "20250717_agisoft_rtk_rgb_om",
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
    "dgm1": ("bavaria_dgm1", [], "datum", None),
}  # fmt: skip


def pipeline(kind: str) -> xdem.coreg.Coreg:
    nk, vs = xdem.coreg.NuthKaab(), xdem.coreg.VerticalShift()
    ramp = xdem.coreg.Deramp(poly_order=1)
    return {"ppk": nk, "rtk": nk + ramp, "rtk_big": vs + ramp + nk, "datum": vs + nk}[kind]


def read(name: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(IN_DIR / f"{name}.tif") as src:
        arr = src.read()
        if src.nodata is not None and not np.isnan(src.nodata) and arr.dtype.kind == "f":
            arr[arr == src.nodata] = np.nan
        return arr, src.profile


def write(name: str, arr: np.ndarray, profile: dict) -> None:
    with rasterio.open(OUT_DIR / f"{name}.tif", "w", **profile) as dst:
        dst.write(arr)
        dst.build_overviews([2, 4, 8, 16], Resampling.average)


def horizontal_shift(coreg: xdem.coreg.Coreg) -> tuple[float, float]:
    steps = coreg.pipeline if isinstance(coreg, xdem.coreg.CoregPipeline) else [coreg]
    m = next(s for s in steps if isinstance(s, xdem.coreg.NuthKaab)).to_matrix()
    return float(m[0, 3]), float(m[1, 3])


def shift_product(arr: np.ndarray, profile: dict, dx: float, dy: float) -> np.ndarray:
    """Move an ortho by (dx, dy) metres and resample it back onto the grid."""
    nodata = profile["nodata"] if profile["nodata"] is not None else 0
    out = np.full_like(arr, nodata)
    tf = profile["transform"]
    reproject(arr, out, src_transform=rasterio.Affine(tf.a, tf.b, tf.c + dx, tf.d, tf.e, tf.f + dy),
              src_crs=CRS, src_nodata=nodata, dst_transform=tf, dst_crs=CRS, dst_nodata=nodata,
              resampling=Resampling.bilinear)  # fmt: skip
    return out


def fit(ref, tba, mask, kind, transform):
    """Two-pass fit: second pass ignores >3 NMAD residuals of the first."""
    kw = dict(transform=transform, crs=CRS, subsample=1_000_000, random_state=42)
    coreg = pipeline(kind).fit(ref, tba, inlier_mask=mask, **kw)
    dh = coreg.apply(tba, transform=transform, crs=CRS)[0] - ref
    med, spread = np.nanmedian(dh[mask]), nmad(dh[mask])
    inliers = mask & (np.abs(dh - med) < 3 * max(spread, 0.05))
    coreg = pipeline(kind).fit(ref, tba, inlier_mask=inliers, **kw)
    return coreg, inliers


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref, ref_profile = read(REF)
    ref = ref[0]
    transform, shape = ref_profile["transform"], ref.shape
    write(REF, ref[None], ref_profile)

    base = ~feature_mask(
        polygons=[gpd.read_file(MANUAL / "Suspicious_Areas.shp").geometry],
        points=[
            gpd.read_file(MANUAL / "Sinkholes.shp").geometry,
            gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").geometry,
        ],
        transform=transform, shape=shape, crs=CRS,
    )  # fmt: skip
    base &= slope_deg(ref, RES) < MAX_SLOPE
    if (snowless_ref := ~snow_mask(read("20250820_terra_ppk_rgb_om")[0], RES)).mean() < 0.99:
        print(f"  note: {1 - snowless_ref.mean():.1%} of the reference ortho flagged as snow")

    rows = []
    for chain, (dsm_name, products, kind, snow_from) in CHAINS.items():
        dsm, profile = read(dsm_name)
        dsm = dsm[0]
        mask = base & erode_valid(np.isfinite(dsm) & np.isfinite(ref), RES)
        if snow_from:
            mask &= ~snow_mask(read(snow_from)[0], RES)

        coreg, inliers = fit(ref, dsm, mask, kind, transform)
        aligned = coreg.apply(dsm, transform=transform, crs=CRS)[0].astype("float32")
        dx, dy = horizontal_shift(coreg)
        write(dsm_name, aligned[None], profile)
        for p in products:
            arr, p_profile = read(p)
            write(p, shift_product(arr, p_profile, dx, dy), p_profile)

        before, after = (dsm - ref), (aligned - ref)
        rows.append({
            "chain": chain, "pipeline": kind, "dx_m": dx, "dy_m": dy,
            "stable_px": int(mask.sum()), "inlier_frac": inliers.sum() / max(mask.sum(), 1),
            "median_before_m": np.nanmedian(before[mask]), "nmad_before_m": nmad(before[mask]),
            "median_after_m": np.nanmedian(after[mask]), "nmad_after_m": nmad(after[mask]),
            "nmad_after_inliers_m": nmad(after[inliers]),
        })  # fmt: skip
        print(f"  {chain:16s} dx {dx:+.2f} dy {dy:+.2f}  NMAD {rows[-1]['nmad_before_m']:.3f}"
              f" -> {rows[-1]['nmad_after_m']:.3f} m")  # fmt: skip

    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "coregistration_report.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
