"""Morphometry of the mapped dolines (Aug 2025 reference outlines, Aug 2025 LiDAR DSM).

Per polygon (EPSG:32632, DSM on the 20 cm common grid):
- planform: area, perimeter, equivalent diameter D = 2 sqrt(A / pi),
  circularity 4 pi A / P^2, elongation L/W and long-axis azimuth (0-180 deg)
  from the minimum rotated rectangle
- depth and volume below a "lid": a plane fitted (least squares, rim outliers
  > 3 MAD dropped) to DSM heights sampled every 0.2 m along the outline. A
  tilted lid follows rims on sloping ground, where a horizontal one would
  overfill the downhill side. depth = lid - DSM inside the polygon, >= 0;
  volume = sum(depth) * pixel area; d/D = max depth / D
- wall steepness: median and 90th percentile slope inside the polygon
- setting: regional slope and aspect at the doline (DSM smoothed with a 25 m
  Gaussian, so the doline itself doesn't dominate), and the angle between the
  doline's long axis and the contour direction (0 = along the contour,
  90 = downslope; uniform on 0-90 if unrelated)
Lithology and distance to the nearest dGK25 fault are joined from
geology_vs_dolines.csv by id.

Outputs: data/processed/morphometry_20250820.gpkg (polygons + metrics),
outputs/figures/morphometry_20250820.csv and .png
"""

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from scipy import ndimage, stats

from master_thesis.paths import FIGURES, INTERIM, MANUAL, PROCESSED

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CRS = "EPSG:32632"
EPOCH = "2025-08-20"
DSM = INTERIM / "coregistered_20cm" / "20250820_terra_ppk_lidar_dsm.tif"
SMOOTH_M = 25.0
OUT = PROCESSED / "morphometry_20250820.gpkg"


def axis(geom) -> tuple[float, float]:
    """Elongation L/W and azimuth (0-180) of the long side of the min rotated rectangle."""
    rect = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)
    sides = np.diff(rect[:3], axis=0)
    lengths = np.hypot(*sides.T)
    long = sides[np.argmax(lengths)]
    return float(lengths.max() / max(lengths.min(), 1e-6)), float(
        np.degrees(np.arctan2(long[0], long[1])) % 180
    )


def lid_depth(geom, src) -> dict:
    x0, y0, x1, y1 = geom.buffer(1.0).bounds
    win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
    z = src.read(1, window=win, masked=True).astype("f8").filled(np.nan)
    tf = src.window_transform(win)
    inside = geometry_mask([geom], z.shape, tf, invert=True) & np.isfinite(z)
    ring = geom.exterior
    s = np.arange(0, ring.length, 0.2)
    pts = np.array([ring.interpolate(d).coords[0] for d in s])
    rows, cols = rasterio.transform.rowcol(tf, pts[:, 0], pts[:, 1])
    rows, cols = np.clip(rows, 0, z.shape[0] - 1), np.clip(cols, 0, z.shape[1] - 1)
    zr = z[rows, cols]
    ok = np.isfinite(zr)
    if ok.sum() < 6 or not inside.any():
        return {}
    pts, zr = pts[ok], zr[ok]
    med = np.median(zr)
    mad = 1.4826 * np.median(np.abs(zr - med)) or 0.01
    keep = np.abs(zr - med) <= 3 * mad
    a = np.c_[np.ones(keep.sum()), pts[keep]]
    coef, *_ = np.linalg.lstsq(a, zr[keep], rcond=None)
    rr, cc = np.nonzero(inside)
    xs, ys = rasterio.transform.xy(tf, rr, cc)
    lid = coef[0] + coef[1] * np.asarray(xs) + coef[2] * np.asarray(ys)
    depth = np.clip(lid - z[rr, cc], 0, None)
    gy, gx = np.gradient(np.where(np.isfinite(z), z, np.nanmedian(z)), tf.a)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))[inside]
    return {
        "max_depth_m": float(depth.max()),
        "mean_depth_m": float(depth.mean()),
        "volume_m3": float(depth.sum() * tf.a**2),
        "rim_relief_m": float(np.ptp(zr[keep])),
        "rim_tilt_deg": float(np.degrees(np.arctan(np.hypot(coef[1], coef[2])))),
        "slope_med_deg": float(np.median(slope)),
        "slope_p90_deg": float(np.percentile(slope, 90)),
    }


def regional_aspect(src, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slope and aspect (azimuth of steepest descent) of the smoothed terrain at points."""
    f = 5  # 1 m is plenty for a 25 m smoothing
    z = src.read(1, out_shape=(src.height // f, src.width // f), masked=True)
    z = z.astype("f8").filled(np.nan)
    res = src.res[0] * f
    tf = src.transform * src.transform.scale(f, f)
    valid = np.isfinite(z)
    filled = np.where(valid, z, 0)
    w = ndimage.gaussian_filter(valid.astype(float), SMOOTH_M / res)
    zs = ndimage.gaussian_filter(filled, SMOOTH_M / res) / np.where(w > 0.2, w, np.nan)
    dzdy, dzdx = np.gradient(zs, -res, res)  # rows run south
    rows, cols = rasterio.transform.rowcol(tf, xy[:, 0], xy[:, 1])
    gx, gy = dzdx[rows, cols], dzdy[rows, cols]
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(-gx, -gy)) % 360
    return slope, aspect


def main() -> None:
    pg = gpd.read_file(MANUAL / "sinkhole_polygons.gpkg", layer="sinkholes").to_crs(CRS)
    pg = pg[(pg.epoch == EPOCH) & pg.id.notna()].copy()
    pg["id"] = pg.id.astype(int)
    pg["area_m2"] = pg.area
    pg["perimeter_m"] = pg.length
    pg["diam_m"] = 2 * np.sqrt(pg.area_m2 / np.pi)
    pg["circularity"] = 4 * np.pi * pg.area_m2 / pg.perimeter_m**2
    ax = np.array([axis(g) for g in pg.geometry])
    pg["elongation"], pg["axis_deg"] = ax[:, 0], ax[:, 1]
    with rasterio.open(DSM) as src:
        metrics = pd.DataFrame([lid_depth(g, src) for g in pg.geometry], index=pg.index)
        cen = np.c_[pg.geometry.centroid.x, pg.geometry.centroid.y]
        pg["reg_slope_deg"], pg["aspect_deg"] = regional_aspect(src, cen)
    pg = pg.join(metrics)
    pg["d_over_D"] = pg.max_depth_m / pg.diam_m
    contour = (pg.aspect_deg + 90) % 180
    d = np.abs(pg.axis_deg - contour) % 180
    pg["axis_vs_contour_deg"] = np.minimum(d, 180 - d)
    geo = pd.read_csv(FIGURES / "geology_vs_dolines.csv", encoding="utf-8")
    pg = pg.merge(geo[["id", "unit", "fault_dist_m"]], on="id", how="left")

    num = ["area_m2", "perimeter_m", "diam_m", "circularity", "elongation", "axis_deg",
           "max_depth_m", "mean_depth_m", "volume_m3", "d_over_D", "rim_relief_m",
           "rim_tilt_deg", "slope_med_deg", "slope_p90_deg", "reg_slope_deg", "aspect_deg",
           "axis_vs_contour_deg", "fault_dist_m"]  # fmt: skip
    pg[num] = pg[num].round(3)
    pg.to_file(OUT, layer="morphometry_20250820", driver="GPKG")
    pg.drop(columns="geometry").to_csv(FIGURES / "morphometry_20250820.csv", index=False)

    # orientation vs slope: only elongated dolines on sloping ground
    sel = (pg.elongation >= 1.3) & (pg.reg_slope_deg >= 5)
    ang = pg.loc[sel, "axis_vs_contour_deg"]
    p_c = stats.wilcoxon(ang - 45).pvalue if len(ang) > 5 else np.nan
    summary = pg.groupby("type")[["area_m2", "max_depth_m", "volume_m3", "d_over_D",
                                  "elongation", "slope_p90_deg"]].median().round(2)  # fmt: skip
    summary["n"] = pg.groupby("type").size()
    print(summary.to_string())
    tot_v, tot_a = pg.volume_m3.sum(), pg.area_m2.sum()
    print(f"\ntotal volume {tot_v:.0f} m3, area {tot_a:.0f} m2, n = {len(pg)}")
    print(f"axis vs contour (elongation >= 1.3, slope >= 5 deg): n = {len(ang)}, "
          f"median {ang.median():.1f} deg (0 = along contour, 90 = downslope), "
          f"share < 45 deg {np.mean(ang < 45):.2f}, Wilcoxon vs 45 p = {p_c:.3f}")  # fmt: skip
    summary.to_csv(FIGURES / "morphometry_20250820_by_type.csv")

    types = pg["type"].value_counts().index.tolist()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    panels = [(axs[0, 0], "area_m2", True), (axs[0, 1], "max_depth_m", True),
              (axs[0, 2], "d_over_D", False), (axs[1, 0], "slope_p90_deg", False)]  # fmt: skip
    for a, col, log in panels:
        a.boxplot([pg.loc[pg["type"] == t, col].dropna() for t in types], showfliers=True)
        a.set_xticks(range(1, len(types) + 1), [f"{t}\n(n={sum(pg['type'] == t)})" for t in types],
                     fontsize=7)  # fmt: skip
        a.set_title(col)
        if log:
            a.set_yscale("log")
    a = axs[1, 1]
    for t in types:
        s = pg[pg["type"] == t]
        a.scatter(s.diam_m, s.max_depth_m, s=12, label=t)
    xx = np.logspace(-0.3, 1.5, 10)
    for r, ls in [(0.1, ":"), (0.33, "--"), (1, "-")]:
        a.plot(xx, r * xx, "k", ls=ls, lw=0.7, label=f"d/D = {r}")
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xlabel("equivalent diameter D [m]"), a.set_ylabel("max depth d [m]")
    a.legend(fontsize=6)
    a = axs[1, 2]
    a.hist(ang, bins=np.arange(0, 91, 10), color="tab:blue")
    a.axhline(len(ang) / 9, color="k", lw=0.7, ls="--")
    a.set_xlabel("long axis vs contour [deg] (0 = along contour, 90 = downslope)")
    a.set_title(f"elongated dolines on slopes, n = {len(ang)}, p = {p_c:.3f}")
    fig.suptitle(f"Doline morphometry, Aug 2025 (n = {len(pg)})")
    fig.tight_layout()
    fig.savefig(FIGURES / "morphometry_20250820.png", dpi=110)
    print(f"wrote {OUT} and {FIGURES / 'morphometry_20250820.png'}")


if __name__ == "__main__":
    main()
