"""How small a height change can we trust? Per-pair uncertainty and LoD95 maps.

Follows Hugonnet et al. (2022), as recommended in
docs/thesis/lit_uav_coregistration.md, on the co-registered DSMs
(coregister_to_reference.py), for every pair of the epochs that line up to a
few cm (the three Terra LiDAR DSMs and the DGM1):

1. dh = later - earlier on stable ground: outside buffers around mapped
   features, >5 m from data edges, no snow in either epoch's RGB ortho,
   within 4 NMAD of the median (leftover snow, blunders). No slope limit:
   the error model has to cover the steep doline walls too.
2. Heteroscedasticity: the dh spread (NMAD) binned by slope and maximum
   curvature gives a per-pixel error sigma(slope, curvature). Steep and rough
   terrain is less certain than flat rock.
3. Spatial correlation: variogram of dh / sigma, fitted with a Gaussian +
   spherical model (short- and long-range correlated errors).
4. Outputs per pair:
   - LoD95 map = 1.96 * sigma, the smallest change at a single 1 m pixel that
     is significant at 95 %;
   - for the mean change over a circular area (a doline), the error shrinks
     with the number of effectively independent samples in it, which the
     variogram gives. Reported for 10 m and 30 m wide circles.

Everything runs at 1 m (5x5 average of the 20 cm grid): enough for metre-scale
dolines and keeps the variograms tractable. 20 cm pixels are noisier, so their
single-pixel LoD would be higher. EPSG:32632, ellipsoidal heights.
Output: data/processed/uncertainty_1m/lod95_<later>_<earlier>.tif, and
figures/uncertainty_lod.csv / .png.
"""

from itertools import combinations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xdem
from affine import Affine
from rasterio.enums import Resampling

from master_thesis.paths import FIELD_DATA, FIGURES, INTERIM, MANUAL, PROCESSED
from master_thesis.stable_ground import (
    erode_valid,
    feature_mask,
    inliers_of,
    median,
    nmad,
    snow_mask,
)

CRS = rasterio.crs.CRS.from_epsg(32632)
IN_DIR = INTERIM / "coregistered_20cm"
OUT_DIR = PROCESSED / "uncertainty_1m"
FACTOR = 5  # 20 cm -> 1 m
RES = 1.0
# label: (DSM, RGB ortho for the snow mask), oldest first; DGM1 flight date unknown
EPOCHS = {
    "dgm1": ("bavaria_dgm1", None),
    "20241025": ("20241025_terra_ppk_lidar_dsm", "20241025_terra_ppk_rgb_om"),
    "20250604": ("20250604_terra_ppk_lidar_dsm", "20250604_terra_ppk_rgb_om"),
    "20250820": ("20250820_terra_ppk_lidar_dsm", "20250820_terra_ppk_rgb_om"),
}
DIAMETERS_M = (10, 30)


def load(name: str) -> tuple[np.ndarray, Affine]:
    """Read a raster from the 20 cm grid, averaged to 1 m."""
    with rasterio.open(IN_DIR / f"{name}.tif") as src:
        shape = (src.count, src.height // FACTOR, src.width // FACTOR)
        arr = src.read(out_shape=shape, resampling=Resampling.average).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            arr[arr == src.nodata] = np.nan
        tf = src.transform * Affine.scale(src.width / shape[2], src.height / shape[1])
    return arr, tf


def stable_mask(dems: dict, snow: dict, tf: Affine) -> np.ndarray:
    shape = next(iter(dems.values())).shape
    features = feature_mask(
        polygons=[gpd.read_file(MANUAL / "Suspicious_Areas.shp").geometry],
        points=[
            gpd.read_file(MANUAL / "Sinkholes.shp").geometry,
            gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").geometry,
        ],
        transform=tf, shape=shape, crs=CRS,
    )  # fmt: skip
    return ~features & ~np.logical_or.reduce(list(snow.values()))


def analyse_pair(dh, dem_mean, stable, tf):
    slope, maxc = xdem.terrain.get_terrain_attribute(
        dem_mean, attribute=["slope", "max_curvature"], resolution=RES
    )
    # passed as a raster: xdem 0.2.3 takes the pixel size only from a raster here
    dh_r = xdem.DEM.from_array(dh, tf, CRS, nodata=-9999.0)
    sigma, binning, _ = xdem.spatialstats.infer_heteroscedasticity_from_stable(
        dh_r, [slope, maxc], stable_mask=stable, list_var_names=["slope", "maxc"]
    )
    sigma = sigma.get_nanarray() if hasattr(sigma, "get_nanarray") else sigma
    emp, params, _ = xdem.spatialstats.infer_spatial_correlation_from_stable(
        dh, ["Gaussian", "Spherical"], stable_mask=stable, errors=sigma, gsd=RES,
        subsample=2000, n_variograms=5, random_state=42,
    )  # fmt: skip
    return sigma, slope, binning, emp, params


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dems, snow = {}, {}
    for label, (dsm, ortho) in EPOCHS.items():
        dems[label], tf = load(dsm)
        dems[label] = dems[label][0]
        if ortho:
            snow[label] = snow_mask(load(ortho)[0], RES)
    base = stable_mask(dems, snow, tf)
    with rasterio.open(IN_DIR / f"{EPOCHS['20250820'][0]}.tif") as src:
        profile = src.profile | dict(
            width=dems["20250820"].shape[1], height=dems["20250820"].shape[0],
            transform=tf, count=1, dtype="float32", nodata=np.nan, crs=CRS,
        )  # fmt: skip

    rows, curves = [], {}
    for early, late in combinations(EPOCHS, 2):
        pair = f"{late}_{early}"
        dh = dems[late] - dems[early]
        valid = erode_valid(np.isfinite(dh), RES)
        stable = inliers_of(dh, base & valid, k=4.0)
        sigma, slope, binning, emp, params = analyse_pair(
            dh, np.nanmean([dems[late], dems[early]], axis=0), stable, tf
        )
        lod95 = np.where(valid, 1.96 * sigma, np.nan).astype("float32")
        with rasterio.open(OUT_DIR / f"lod95_{pair}.tif", "w", **profile) as dst:
            dst.write(lod95, 1)

        row = {
            "pair": pair, "stable_px": int(stable.sum()),
            "median_dh_m": median(dh[stable]), "nmad_dh_m": nmad(dh[stable]),
            "lod95_median_m": median(lod95[valid]),
            "lod95_flat_m": median(lod95[valid & (slope < 20)]),
            "lod95_steep_m": median(lod95[valid & (slope > 40)]),
        }  # fmt: skip
        # short- and long-range part; total sill ~1 if sigma explains all the scatter
        for tag, (_, p) in zip(
            ("short", "long"), params.sort_values("range").iterrows(), strict=True
        ):
            row[f"range_{tag}_m"], row[f"psill_{tag}"] = p["range"], p["psill"]
        sigma_typ = median(sigma[valid])
        for d in DIAMETERS_M:
            n_eff = xdem.spatialstats.number_effective_samples(np.pi * (d / 2) ** 2, params)
            row[f"lod95_mean_{d}m_m"] = 1.96 * sigma_typ / np.sqrt(n_eff)
        rows.append(row)
        one_d = binning[(binning.nd == 1) & (binning["count"] >= 100)].dropna(subset=["slope"])
        curves[pair] = (one_d, emp)
        print(f"  {pair}: NMAD {row['nmad_dh_m']:.3f} m, LoD95 flat {row['lod95_flat_m']:.2f}"
              f" / steep {row['lod95_steep_m']:.2f} m, 30 m area "
              f"{row['lod95_mean_30m_m']:.3f} m", flush=True)  # fmt: skip

    report = pd.DataFrame(rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    report.to_csv(FIGURES / "uncertainty_lod.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.round(3).to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for pair, (one_d, emp) in curves.items():
        mid = one_d.slope.apply(lambda iv: iv.mid)
        ax1.plot(mid, 1.96 * one_d.nmad, marker="o", ms=3, label=pair)
        e = emp[emp["count"] > 0]
        ax2.plot(e.lags, e.exp, marker="o", ms=3, label=pair)
    ax1.set(xlabel="slope [deg]", ylabel="LoD95 of a 1 m pixel [m]",
            title="Error grows with slope")  # fmt: skip
    ax2.set(xscale="log", xlabel="lag [m]", ylabel="variance of dh / sigma",
            title="Spatial correlation of the errors")  # fmt: skip
    ax2.axhline(1, c="k", lw=0.5)
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "uncertainty_lod.png", dpi=120)
    print(f"wrote {OUT_DIR}, {FIGURES / 'uncertainty_lod.csv'}")


if __name__ == "__main__":
    main()
