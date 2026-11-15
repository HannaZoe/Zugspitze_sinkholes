"""Do the three PPK LiDAR DSMs agree on stable ground, and with the DGM1?

Before any DSM differencing we need to know how far apart the "best" epochs
already are on ground that should not have changed. Everything is compared
on a common 0.5 m grid (EPSG:32632, the project CRS) over the area all three
LiDAR epochs cover:

- the three LiDAR pairs (each epoch minus the next/reference)
- each LiDAR epoch minus the Bavarian DGM1 (1 m airborne LiDAR DTM,
  EPSG:25832 + DHHN2016 heights, reprojected here)

Stable ground = valid in every layer, slope below MAX_SLOPE, outside a buffer
around every mapped doline / suspicious area / field point, and not bright
in any epoch's RGB ortho (a cheap snow safety net; it also drops some pale
limestone, which costs nothing but sample size).

Per pair: median and NMAD of dh (bias and robust spread), a best-fit plane
(tilt) and the Nuth & Kaab (2011) horizontal/vertical shift from xdem. Writes
a CSV table, a figure, and the stable-ground mask as GeoTIFF for reuse in
the actual co-registration step.

Note: PROJ treats ETRS89 and WGS84 as identical (~1 m accuracy), so a
real datum shift between the DGM1 and the UAV data would show up here as a
horizontal Nuth & Kaab shift, not be silently corrected.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xdem
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from master_thesis.paths import FIELD_DATA, FIGURES, INTERIM, MANUAL, RAW

CRS = rasterio.crs.CRS.from_epsg(32632)
RES = 0.5  # m, plenty for bias/tilt/shift; 5 cm would just be 100x slower
EPOCHS = ["20241025", "20250604", "20250820"]
REF = "20250820"  # snow-free, the intended reference epoch
PAIRS = [("20241025", "20250820"), ("20250604", "20250820"), ("20241025", "20250604")]
MAX_SLOPE = 40  # deg
POLY_BUFFER = 10  # m around mapped polygons
POINT_BUFFER = 20  # m around point features (points mark centres, dolines reach ~40 m)
SNOW_BRIGHTNESS = 200  # mean RGB (0-255) above which a pixel might be snow

OUT_TABLE = FIGURES / "lidar_stability.csv"
OUT_FIG = FIGURES / "lidar_stability.png"
OUT_MASK = INTERIM / "stable_ground_mask_0p5m.tif"


def common_grid() -> tuple[rasterio.Affine, tuple[int, int]]:
    """Intersection of the three LiDAR extents, snapped to RES."""
    bounds = []
    for d in EPOCHS:
        with rasterio.open(RAW / d / f"{d}_terra_ppk_lidar_dsm.tif") as src:
            assert src.crs == CRS, f"{d} is {src.crs}"
            bounds.append(src.bounds)
    left = np.ceil(max(b.left for b in bounds) / RES) * RES
    right = np.floor(min(b.right for b in bounds) / RES) * RES
    bottom = np.ceil(max(b.bottom for b in bounds) / RES) * RES
    top = np.floor(min(b.top for b in bounds) / RES) * RES
    shape = (round((top - bottom) / RES), round((right - left) / RES))
    return from_origin(left, top, RES, RES), shape


def warp(src, band, transform, shape, resampling, src_crs=None, src_transform=None):
    """Resample one band (array or dataset band) onto the common grid as float32/NaN."""
    out = np.full(shape, np.nan, dtype="float32")
    reproject(
        band,
        out,
        src_transform=src_transform or src.transform,
        src_crs=src_crs or src.crs,
        src_nodata=src.nodata if src_transform is None else -9999.0,
        dst_transform=transform,
        dst_crs=CRS,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return out


def load_dsm(path, transform, shape):
    with rasterio.open(path) as src:
        return warp(src, rasterio.band(src, 1), transform, shape, Resampling.average)


def load_dgm1(transform, shape):
    tiles = [rasterio.open(p) for p in sorted((RAW / "bavaria_dgm1").glob("*.tif"))]
    mosaic, mosaic_tf = merge(tiles, nodata=-9999.0)
    src = tiles[0]
    out = warp(src, mosaic[0], transform, shape, Resampling.bilinear, src.crs, mosaic_tf)
    for t in tiles:
        t.close()
    return out


def bright_mask(date, transform, shape):
    """True where the RGB ortho is bright enough to possibly be snow."""
    with rasterio.open(RAW / date / f"{date}_terra_ppk_rgb_om.tif") as src:
        rgb = [
            warp(src, rasterio.band(src, b), transform, shape, Resampling.average)
            for b in (1, 2, 3)
        ]
    return np.nanmean(rgb, axis=0) > SNOW_BRIGHTNESS


def feature_mask(transform, shape):
    """True inside the buffer around any known doline-ish feature."""
    polys = gpd.read_file(MANUAL / "Suspicious_Areas.shp").to_crs(CRS).buffer(POLY_BUFFER)
    points = pd.concat(
        [
            gpd.read_file(MANUAL / "Sinkholes.shp").to_crs(CRS).geometry,
            gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").to_crs(CRS).geometry,
        ]
    )
    points = points[points.notna() & ~points.is_empty].buffer(POINT_BUFFER)
    geoms = [g for g in pd.concat([polys, points]) if g is not None and not g.is_empty]
    return geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


def slope_deg(dem):
    gy, gx = np.gradient(dem, RES)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def nmad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def compare(name, a, b, stable, transform):
    """Stats of dh = a - b on stable ground, plus plane tilt and Nuth & Kaab shift."""
    m = stable & np.isfinite(a) & np.isfinite(b)
    dh = (a - b)[m]
    rows, cols = np.nonzero(m)
    x = (cols - cols.mean()) * RES / 1000  # km
    y = -(rows - rows.mean()) * RES / 1000
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x, y]), dh, rcond=None)
    resid = dh - np.column_stack([np.ones_like(x), x, y]) @ coef

    nk = xdem.coreg.NuthKaab().fit(b, a, inlier_mask=m, transform=transform, crs=CRS)
    shift = nk.to_matrix()[:3, 3]  # what a needs to move by to sit on b

    return {
        "pair": name,
        "n_px": int(m.sum()),
        "median_m": np.median(dh),
        "nmad_m": nmad(dh),
        "p5_m": np.percentile(dh, 5),
        "p95_m": np.percentile(dh, 95),
        "frac_abs_gt_0p5m": np.mean(np.abs(dh) > 0.5),
        "tilt_east_m_per_km": coef[1],
        "tilt_north_m_per_km": coef[2],
        "nmad_after_plane_m": nmad(resid),
        "nk_dx_m": shift[0],
        "nk_dy_m": shift[1],
        "nk_dz_m": shift[2],
    }


def main() -> None:
    transform, shape = common_grid()
    print(f"common grid {shape[1]} x {shape[0]} px at {RES} m")

    dsm = {d: load_dsm(RAW / d / f"{d}_terra_ppk_lidar_dsm.tif", transform, shape) for d in EPOCHS}
    dsm["dgm1"] = load_dgm1(transform, shape)

    valid = np.logical_and.reduce([np.isfinite(v) for v in dsm.values()])
    steep = slope_deg(dsm[REF]) >= MAX_SLOPE
    features = feature_mask(transform, shape)
    bright = np.logical_or.reduce([bright_mask(d, transform, shape) for d in EPOCHS])
    stable = valid & ~steep & ~features & ~bright
    for label, excl in [("steep", steep), ("near features", features), ("bright", bright)]:
        print(f"  excluded {label}: {np.mean(excl & valid):.1%} of valid area")
    print(f"  stable ground: {stable.sum() / valid.sum():.1%} of valid area")

    pairs = PAIRS + [(d, "dgm1") for d in EPOCHS]
    table = pd.DataFrame(
        [compare(f"{a} - {b}", dsm[a], dsm[b], stable, transform) for a, b in pairs]
    )
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))

    INTERIM.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="uint8",
                   crs=CRS, transform=transform, compress="deflate")  # fmt: skip
    with rasterio.open(OUT_MASK, "w", **profile) as dst:
        dst.write(stable.astype("uint8"), 1)

    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3 * len(pairs)),
                             gridspec_kw={"width_ratios": [3, 1]})  # fmt: skip
    for (a, b), (ax_map, ax_hist) in zip(pairs, axes, strict=True):
        dh = np.where(stable, dsm[a] - dsm[b], np.nan)
        im = ax_map.imshow(dh, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax_map.set_title(f"{a} - {b} (stable ground only)")
        ax_map.set_axis_off()
        fig.colorbar(im, ax=ax_map, label="dh [m]", shrink=0.8)
        vals = dh[np.isfinite(dh)]
        ax_hist.hist(vals, bins=200, range=(-2, 2), color="0.3")
        ax_hist.axvline(np.median(vals), color="C3")
        ax_hist.set_xlabel("dh [m]")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=120)
    print(f"wrote {OUT_TABLE}, {OUT_FIG}, {OUT_MASK}")


if __name__ == "__main__":
    main()
