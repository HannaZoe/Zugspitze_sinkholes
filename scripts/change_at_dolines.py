"""How much did the dolines change, and can the warped 2026 DSM see it?

Decides whether reprocessing the Aug 2026 Mavic 3M images (pseudo-GCPs from
the LiDAR) is worth it: if the changes are of the order of the 2026 DSM's
error, it is; if they are metre-scale collapses or nothing, it isn't.

Doline zones: no mapped polygons yet, so 15 m buffers around the doline points
(Sinkholes.shp + field points of type Doline / Cave Doline), overlapping
buffers merged into one zone. 15 m = up to ~8 m half-width of the larger
dolines + ~5 m phone-GPS error of the field points + a little room for rim
retreat. Zones are generous: they also hold stable ground around the doline.

Pairs (all on the 1 m grid = 5x5 average of the co-registered 20 cm grid,
EPSG:32632, ellipsoidal heights):
- LiDAR: Oct 2024 -> Jun 2025, Jun 2025 -> Aug 2025, Oct 2024 -> Aug 2025.
- Aug 2025 LiDAR -> Aug 2026 3M (photogrammetric). The delivered 3M DSM is
  warped by up to +-1 m (three base set-ups, see experiment log 2026-11-17/18).
  Corrected here by a smooth surface: normalised Gaussian (sigma 60 m) of dh,
  fitted on stable ground only (outside 20 m feature buffers, no snow,
  >5 m from edges, 4-NMAD inliers, two passes). Checked on a 100 m
  checkerboard holdout.

Error model per pair as in uncertainty_lod.py (Hugonnet et al. 2022): sigma
from slope + max curvature, variogram of dh / sigma. Per zone:
- max |dh| and the area with |dh| > LoD95 (per pixel, 1.96 sigma);
- gain / loss / net volume of those significant pixels;
- net volume over the whole zone with its 1-sigma error
  A * sigma_mean(A), sigma_mean from the effective sample count of the zone
  area (the area-mean error of Hugonnet et al., with the zone's mean sigma).
Pixels with snow in either epoch's RGB ortho are excluded.

Outputs:
- data/processed/change_20cm/dh_<late>_<early>.tif (20 cm, LiDAR pairs and the
  corrected 2026 pair), 20260827_3m_dsm_smoothcorr.tif (corrected 2026 DSM);
- data/processed/uncertainty_1m/lod95_20260827smooth_20250820.tif;
- figures/change_at_dolines.csv / .png, figures/change_chips/sheet_XX.jpg.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter, label

from master_thesis.dh_uncertainty import error_model, load_avg, sigma_of_mean
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
OUT_DIR = PROCESSED / "change_20cm"
FACTOR, RES = 5, 1.0
ZONE_BUFFER_M = 15
SMOOTH_SIGMA_M = 60
HOLDOUT_BLOCK_M = 100
CHIP_M = 40
# A zone counts as changed if a connected same-sign patch of pixels beyond LoD95 is larger
# than what stable ground produces FALSE_PER_HA times per ha (and at least MIN_CLUSTER_M2).
MIN_CLUSTER_M2 = 10
FALSE_PER_HA = 0.5
# Snow in the dolines is real in Oct 2024 (early snow) and Jun 2025 (snow-filled), so those
# pixels are dropped from the zone stats. In the August orthos the (deliberately aggressive)
# snow mask mostly flags bright limestone, checked visually, so August is not masked there.
SNOW_IN_DOLINES = {"20241025", "20250604"}
RELIEF_BINS = [-10, -2, -1, -0.5, -0.2, 0.2, 0.5, 1, 10]
# label: (DSM, RGB ortho)
EPOCHS = {
    "20241025": ("20241025_terra_ppk_lidar_dsm", "20241025_terra_ppk_rgb_om"),
    "20250604": ("20250604_terra_ppk_lidar_dsm", "20250604_terra_ppk_rgb_om"),
    "20250820": ("20250820_terra_ppk_lidar_dsm", "20250820_terra_ppk_rgb_om"),
    "20260827": ("20260827_terra_rtk_dsm_20cm_local_coregistration",
                 "20260827_terra_rtk_rgb_om_20cm_local_coregistration"),
}  # fmt: skip
PAIRS = [("20241025", "20250604"), ("20250604", "20250820"), ("20241025", "20250820"),
         ("20250820", "20260827")]  # fmt: skip


def doline_zones(tf, shape) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Merged 15 m buffers around doline points, and a label raster (0 = none)."""
    pts = [gpd.read_file(MANUAL / "Sinkholes.shp").to_crs(CRS).assign(src="Sinkholes.shp", kind="")]
    field = gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").to_crs(CRS)
    field = field[field["type"].isin(["Doline", "Cave Doline"])]
    pts.append(field.assign(src="field", kind=field["type"]))
    pts = gpd.GeoDataFrame(pd.concat(pts, ignore_index=True)[["src", "kind", "geometry"]], crs=CRS)
    pts = pts[np.isfinite(pts.geometry.x) & np.isfinite(pts.geometry.y)]  # one corrupted point
    zones = gpd.GeoDataFrame(geometry=list(pts.buffer(ZONE_BUFFER_M).union_all().geoms), crs=CRS)
    zones = zones.iloc[np.argsort(zones.centroid.x.to_numpy())].reset_index(drop=True)
    zones["zone"] = np.arange(1, len(zones) + 1)
    joined = gpd.sjoin(pts, zones, predicate="within")
    agg = joined.groupby("zone").agg(
        n_points=("src", "size"), field_points=("src", lambda s: int((s == "field").sum())),
        cave=("kind", lambda s: bool((s == "Cave Doline").any())),
    )  # fmt: skip
    zones = zones.join(agg, on="zone")
    labels = rasterize(zip(zones.geometry, zones.zone, strict=True), out_shape=shape, transform=tf,
                       fill=0, dtype="int32")  # fmt: skip
    return zones, labels


def smooth(dh: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian mean of dh over the masked pixels only (normalised convolution)."""
    w = gaussian_filter(mask.astype("float32"), sigma_px)
    s = gaussian_filter(np.where(mask, dh, 0).astype("float32"), sigma_px)
    return np.where(w > 1e-3, s / np.maximum(w, 1e-3), np.nan)


def correct_2026(dh: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Smooth error surface of the 2026 DSM, fitted on stable ground; holdout check."""
    sig = SMOOTH_SIGMA_M / RES
    stable = inliers_of(dh, base, k=4.0)
    for _ in range(2):  # refit after dropping what the first surface can't explain
        surf = smooth(dh, stable, sig)
        stable = inliers_of(dh - surf, base, k=4.0)
    yy, xx = np.indices(dh.shape)
    blk = round(HOLDOUT_BLOCK_M / RES)
    test = (yy // blk + xx // blk) % 2 == 1
    resid_test = (dh - smooth(dh, stable & ~test, sig))[stable & test]
    surf = smooth(dh, stable, sig)
    check = {"nmad_before": nmad(dh[stable]), "nmad_holdout": nmad(resid_test),
             "median_holdout": median(resid_test)}  # fmt: skip
    return surf, stable, check


def relief_bias(dh: np.ndarray, dem: np.ndarray) -> pd.Series:
    """Median dh binned by local relief (DEM minus its 5 m Gaussian mean).

    Pits come out negative. If dh rises in pits, the later DSM fills
    depressions (snow, or photogrammetry not reaching the bottom).
    """
    w = gaussian_filter(np.isfinite(dem).astype("float32"), 5 / RES)
    relief = dem - gaussian_filter(np.nan_to_num(dem), 5 / RES) / np.maximum(w, 1e-3)
    ok = np.isfinite(dh) & np.isfinite(relief)
    bins = pd.cut(relief[ok], RELIEF_BINS)
    return pd.Series(dh[ok]).groupby(bins, observed=True).median()


def largest_cluster(sig: np.ndarray, d: np.ndarray) -> float:
    """Area [m2] of the largest connected same-sign patch of significant pixels."""
    best = 0
    for patch in (sig & (d > 0), sig & (d < 0)):
        lab, n = label(patch, structure=np.ones((3, 3)))
        if n:
            best = max(best, int(np.bincount(lab.ravel())[1:].max()))
    return best * RES**2


def cluster_threshold(sig: np.ndarray, d: np.ndarray, mask: np.ndarray) -> float:
    """Patch area [m2] that stable ground exceeds at most FALSE_PER_HA times per ha.

    Errors are spatially correlated, so patches of significant pixels also form
    on stable ground; a doline change has to be bigger than those.
    """
    areas = []
    for patch in (sig & (d > 0) & mask, sig & (d < 0) & mask):
        lab, n = label(patch, structure=np.ones((3, 3)))
        if n:
            areas.append(np.bincount(lab.ravel())[1:] * RES**2)
    areas = np.sort(np.concatenate(areas))[::-1]
    k = int(FALSE_PER_HA * mask.sum() * RES**2 / 1e4)  # allowed number of false patches
    return float(max(areas[k] if k < areas.size else 0.0, MIN_CLUSTER_M2))


def zone_stats(dh, lod, sigma, labels, zones, params, pair) -> list[dict]:
    rows = []
    sig_all = np.abs(dh) > lod
    for z in zones.itertuples():
        inz = labels == z.zone
        ok = inz & np.isfinite(dh)
        d, lz = dh[ok], lod[ok]
        sig = np.abs(d) > lz
        rr, cc = np.nonzero(inz)
        box = np.s_[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1]
        cluster = largest_cluster(sig_all[box] & ok[box], dh[box])
        area = float(ok.sum()) * RES**2
        sv = sigma_of_mean(area, float(np.nanmean(sigma[ok])) if ok.any() else np.nan, params)
        rows.append({
            "pair": pair, "zone": z.zone, "zone_area_m2": float(inz.sum()) * RES**2,
            "coverage": ok.sum() / max(inz.sum(), 1),
            "max_abs_dh_m": float(np.max(np.abs(d))) if d.size else np.nan,
            "p95_abs_dh_m": float(np.percentile(np.abs(d), 95)) if d.size else np.nan,
            "lod95_median_m": median(lz),
            "sig_area_m2": float(sig.sum()) * RES**2,
            "max_cluster_m2": cluster,
            "sig_gain_m3": float(d[sig & (d > 0)].sum()) * RES**2,
            "sig_loss_m3": float(d[sig & (d < 0)].sum()) * RES**2,
            "net_m3": float(d.sum()) * RES**2,
            "net_sigma_m3": area * sv,
        })  # fmt: skip
    return rows


def write(path, arr, profile) -> None:
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


def to_20cm(arr_1m, tf_1m, profile_20) -> np.ndarray:
    out = np.full((profile_20["height"], profile_20["width"]), np.nan, "float32")
    reproject(arr_1m.astype("float32"), out, src_transform=tf_1m, src_crs=CRS, src_nodata=np.nan,
              dst_transform=profile_20["transform"], dst_crs=CRS, dst_nodata=np.nan,
              resampling=Resampling.bilinear)  # fmt: skip
    return out


def read_20cm(name: str) -> np.ndarray:
    with rasterio.open(IN_DIR / f"{name}.tif") as src:
        a = src.read(1).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            a[a == src.nodata] = np.nan
    return a


def chip(name: str, bounds, bands=(1, 2, 3)) -> np.ndarray:
    with rasterio.open(IN_DIR / f"{name}.tif") as src:
        return src.read(list(bands), window=from_bounds(*bounds, src.transform), boundless=True)


def to_rgb8(a: np.ndarray) -> Image.Image:
    a = np.nan_to_num(a.astype("float32"))
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8").transpose(1, 2, 0))


def dh_rgb(d: np.ndarray, lim: float = 1.0) -> Image.Image:
    rgba = plt.get_cmap("RdBu")(np.clip((np.nan_to_num(d) / lim + 1) / 2, 0, 1))
    rgba[~np.isfinite(d)] = (0.6, 0.6, 0.6, 1)
    return Image.fromarray((rgba[..., :3] * 255).astype("uint8"))


def contact_sheets(zones, dh_paths: dict) -> None:
    """Per zone: Aug 2025 RGB | Aug 2026 RGB | dh 2025->26 | dh Oct24->Aug25 (+-1 m)."""
    out = FIGURES / "change_chips"
    out.mkdir(parents=True, exist_ok=True)
    tile, head, per = 130, 14, 8
    for k in range(0, len(zones), per):
        sheet = Image.new("RGB", (2 * 4 * tile + 10, (per // 2) * (tile + head)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, z in enumerate(zones.iloc[k : k + per].itertuples()):
            c = z.geometry.centroid
            bnd = (c.x - CHIP_M / 2, c.y - CHIP_M / 2, c.x + CHIP_M / 2, c.y + CHIP_M / 2)
            ims = [to_rgb8(chip(EPOCHS["20250820"][1], bnd)),
                   to_rgb8(chip(EPOCHS["20260827"][1], bnd))]  # fmt: skip
            for path in dh_paths.values():
                with rasterio.open(path) as src:
                    ims.append(dh_rgb(src.read(1, window=from_bounds(*bnd, src.transform),
                                               boundless=True, fill_value=np.nan)))  # fmt: skip
            x0, y0 = (i % 2) * (4 * tile + 10), (i // 2) * (tile + head)
            draw.text((x0 + 2, y0 + 1), f"zone {z.zone}{' cave' if z.cave else ''}", fill="black")
            for j, im in enumerate(ims):
                sheet.paste(im.resize((tile, tile)), (x0 + j * tile, y0 + head))
        sheet.save(out / f"sheet_{k // per + 1:02d}.jpg", quality=75)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dems, snow = {}, {}
    for ep, (dsm, ortho) in EPOCHS.items():
        arr, tf = load_avg(IN_DIR / f"{dsm}.tif", FACTOR)
        dems[ep] = arr[0]
        snow[ep] = snow_mask(np.nan_to_num(load_avg(IN_DIR / f"{ortho}.tif", FACTOR)[0]), RES)
        print(f"  {ep}: snow {snow[ep].mean():.1%} of the grid", flush=True)
    shape = dems["20250820"].shape
    zones, labels = doline_zones(tf, shape)
    features = feature_mask(
        polygons=[gpd.read_file(MANUAL / "Suspicious_Areas.shp").geometry],
        points=[gpd.read_file(MANUAL / "Sinkholes.shp").geometry,
                gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").geometry],
        transform=tf, shape=shape, crs=CRS,
    )  # fmt: skip
    print(f"  {len(zones)} doline zones from {int(zones.n_points.sum())} points", flush=True)

    with rasterio.open(IN_DIR / f"{EPOCHS['20250820'][0]}.tif") as src:
        p20 = src.profile | dict(dtype="float32", nodata=np.nan, count=1, compress="deflate",
                                 predictor=3, tiled=True)  # fmt: skip
    p1 = p20 | dict(width=shape[1], height=shape[0], transform=tf)

    rows, summary, dh_paths, relief = [], [], {}, {}
    for early, late in PAIRS:
        pair = f"{late}_{early}"
        dh = dems[late] - dems[early]
        valid = erode_valid(np.isfinite(dh), RES)
        nosnow = ~(snow[early] | snow[late])
        base = ~features & nosnow & valid
        s20 = None
        if late == "20260827":
            surf, stable, check = correct_2026(dh, base)
            dh = dh - surf
            print(f"  2026 smooth correction: stable NMAD {check['nmad_before']:.3f} -> holdout "
                  f"{check['nmad_holdout']:.3f} m (median {check['median_holdout']:+.3f})",
                  flush=True)  # fmt: skip
            s20 = to_20cm(surf, tf, p20)
            pair = f"{late}smooth_{early}"
        else:
            stable, check = inliers_of(dh, base, k=4.0), {}
        sigma, *_, params = error_model(dh, np.nanmean([dems[late], dems[early]], axis=0),
                                        stable, tf, CRS, RES)  # fmt: skip
        lod = 1.96 * sigma
        if s20 is not None:
            write(PROCESSED / "uncertainty_1m" / f"lod95_{pair}.tif",
                  np.where(valid, lod, np.nan), p1)  # fmt: skip
        snowy = np.logical_or.reduce([snow[e] for e in (early, late) if e in SNOW_IN_DOLINES]
                                     or [np.zeros(shape, bool)])  # fmt: skip
        dh_zone = np.where(valid & ~snowy, dh, np.nan)
        relief[pair] = relief_bias(dh_zone, dems["20250820"])
        rows += zone_stats(dh_zone, lod, sigma, labels, zones, params, pair)
        thr = cluster_threshold(np.abs(dh) > lod, np.nan_to_num(dh), stable)
        summary.append({"pair": pair, "stable_px": int(stable.sum()),
                        "nmad_stable_m": nmad(dh[stable]),
                        "lod95_median_m": median(lod[valid]),
                        "cluster_threshold_m2": thr} | check)  # fmt: skip

        # 20 cm dh for mapping / inspection
        d20 = read_20cm(EPOCHS[late][0]) - read_20cm(EPOCHS[early][0])
        if s20 is not None:
            corr = read_20cm(EPOCHS[late][0]) - s20
            write(OUT_DIR / "20260827_3m_dsm_smoothcorr.tif", corr, p20)
            d20 = corr - read_20cm(EPOCHS[early][0])
            del corr
        dh_paths[pair] = OUT_DIR / f"dh_{pair}.tif"
        write(dh_paths[pair], d20, p20)
        del d20
        print(f"  {pair}: NMAD {summary[-1]['nmad_stable_m']:.3f} m, "
              f"LoD95 median {summary[-1]['lod95_median_m']:.2f} m", flush=True)  # fmt: skip

    res = pd.DataFrame(rows).merge(zones.drop(columns="geometry"), on="zone")
    thr = pd.DataFrame(summary).set_index("pair").cluster_threshold_m2
    res["cluster_threshold_m2"] = res.pair.map(thr)
    FIGURES.mkdir(parents=True, exist_ok=True)
    res.to_csv(FIGURES / "change_at_dolines.csv", index=False, float_format="%.3f")
    pd.DataFrame(summary).to_csv(FIGURES / "change_at_dolines_pairs.csv", index=False,
                                 float_format="%.3f")  # fmt: skip
    zones.to_file(OUT_DIR / "doline_zones.gpkg", layer="zones", driver="GPKG")

    # per pair: how many zones changed beyond noise, and by how much
    res["significant"] = (res.max_cluster_m2 > res.cluster_threshold_m2) & (res.coverage >= 0.5)
    res["net_sig"] = np.abs(res.net_m3) > 1.96 * res.net_sigma_m3
    res["mean_dh_m"] = res.net_m3 / (res.zone_area_m2 * res.coverage)
    tab = res.groupby("pair").agg(
        zones=("zone", "size"), mean_coverage=("coverage", "mean"),
        zones_sig_px=("significant", "sum"), zones_sig_net=("net_sig", "sum"),
        median_p95_abs_dh=("p95_abs_dh_m", "median"), max_abs_dh=("max_abs_dh_m", "max"),
        median_lod95=("lod95_median_m", "median"),
        threshold_m2=("cluster_threshold_m2", "first"),
    )  # fmt: skip
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(tab.round(3).to_string())
        print(pd.DataFrame(summary).round(3).to_string(index=False))
        print("median dh by local relief [m] (negative relief = pit):")
        print(pd.DataFrame(relief).round(3).to_string())
    pd.DataFrame(relief).to_csv(FIGURES / "change_at_dolines_relief.csv", float_format="%.3f")

    order = [f"{late}_{early}" if late != "20260827" else f"{late}smooth_{early}"
             for early, late in PAIRS]  # fmt: skip
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ticks = (range(1, len(order) + 1), [p.replace("_", "\n") for p in order])
    ax = axs[0, 0]
    ax.boxplot([res.loc[res.pair == p, "p95_abs_dh_m"].dropna() for p in order])
    ax.plot(ticks[0], [res.loc[res.pair == p, "lod95_median_m"].median() for p in order],
            "r_", ms=25, label="median LoD95 (1 px)")  # fmt: skip
    ax.set(title="(a) 95th percentile |dh| per doline zone [m]", yscale="log")
    ax.set_xticks(*ticks, fontsize=7)
    ax.legend(fontsize=7)
    ax = axs[0, 1]
    ax.bar(ticks[0], [res.loc[res.pair == p, "coverage"].mean() for p in order])
    ax.set(title="(b) share of doline-zone pixels usable (snow-free)", ylim=(0, 1))
    ax.set_xticks(*ticks, fontsize=7)
    ax = axs[1, 0]
    for p in order:
        r = res[res.pair == p]
        ax.errorbar(r.zone, r.net_m3, yerr=1.96 * r.net_sigma_m3, fmt="o", ms=3, label=p)
    ax.axhline(0, c="k", lw=0.5)
    ax.set(xlabel="zone (west -> east)", ylabel="m3 (95 % CI, snow-free pixels only)",
           title="(c) net volume change per zone", yscale="symlog")  # fmt: skip
    ax.legend(fontsize=6)
    ax = axs[1, 1]
    rel = pd.DataFrame(relief)
    mids = [iv.mid for iv in rel.index]
    for p in order:
        ax.plot(mids, rel[p], marker="o", ms=3, label=p)
    ax.axhline(0, c="k", lw=0.5)
    ax.set(xlabel="local relief [m] (DSM - 5 m mean; < 0 = pit)", ylabel="median dh [m]",
           title="(d) dh vs local relief, whole area", xscale="symlog")  # fmt: skip
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "change_at_dolines.png", dpi=120)

    contact_sheets(zones, {k: dh_paths[k] for k in (order[3], order[2])})
    print(f"wrote {FIGURES / 'change_at_dolines.csv'}, {OUT_DIR}")


if __name__ == "__main__":
    main()
