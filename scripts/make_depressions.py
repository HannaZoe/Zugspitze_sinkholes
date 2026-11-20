"""Closed depressions ("Senkenkonturen") from the LiDAR DSMs, as a mapping aid.

Depth of closed depressions = filled DSM - DSM, with the sinks filled by
morphological reconstruction (the standard sink-fill approach for karst
depressions from LiDAR, e.g. Doctor & Young 2013). Frame edges and nodata act
as outlets, so nothing is "filled" against a data gap.

Outlines are drawn where the depth reaches 0.1, 0.3 and 1.0 m. Patches
smaller than MIN_AREA_M2 are dropped (gaps between boulders). The 0.1 m
outline is the spill rim of the closed depression. On a slope that is
smaller than the morphological rim (the downhill lip is lower), so it is a
guide for mapping, not the outline itself.

Inputs: data/interim/coregistered_20cm/<epoch>_terra_ppk_lidar_dsm.tif (EPSG:32632)
Outputs: data/processed/depressions/<epoch>_depth.tif (float32, m) and
depressions.gpkg with one layer per epoch (level_m, area_m2, max_depth_m).
"""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from skimage.morphology import reconstruction

from master_thesis.paths import INTERIM, PROCESSED

CRS = "EPSG:32632"
IN_DIR = INTERIM / "coregistered_20cm"
OUT = PROCESSED / "depressions"
EPOCHS = ["20250820", "20241025", "20250604"]
LEVELS = [0.1, 0.3, 1.0]
MIN_AREA_M2 = 0.5


def sink_depth(z: np.ndarray) -> np.ndarray:
    """Depth below the spill level of each closed depression (NaN where no data)."""
    valid = np.isfinite(z)
    outlet = ~valid
    outlet[[0, -1], :] = outlet[:, [0, -1]] = True  # frame edges drain too
    low = np.nanmin(z) - 1
    surface = np.where(valid, z, low)
    seed = np.where(outlet, surface, surface.max())
    filled = reconstruction(seed, surface, method="erosion")
    return np.where(valid, filled - surface, np.nan).astype("float32")


def outlines(depth: np.ndarray, transform, px_area: float) -> gpd.GeoDataFrame:
    rows = []
    for level in LEVELS:
        lab, n = ndimage.label(depth >= level)
        if not n:
            continue
        idx = np.arange(1, n + 1)
        area = ndimage.sum_labels(np.ones_like(depth), lab, idx) * px_area
        dmax = ndimage.maximum(np.nan_to_num(depth), lab, idx)
        keep = np.zeros(n + 1, bool)
        keep[1:] = area >= MIN_AREA_M2
        lab = np.where(keep[lab], lab, 0).astype("int32")
        for geom, v in shapes(lab, mask=lab > 0, transform=transform):
            i = int(v) - 1
            rows.append({
                "level_m": level, "area_m2": round(float(area[i]), 2),
                "max_depth_m": round(float(dmax[i]), 2), "geometry": shape(geom),
            })  # fmt: skip
    return gpd.GeoDataFrame(rows, crs=CRS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpkg = OUT / "depressions.gpkg"
    for epoch in EPOCHS:
        with rasterio.open(IN_DIR / f"{epoch}_terra_ppk_lidar_dsm.tif") as src:
            z = src.read(1, masked=True).astype("float64").filled(np.nan)
            profile = src.profile | {"dtype": "float32", "nodata": -9999.0, "compress": "deflate"}
            transform = src.transform
            px_area = abs(src.res[0] * src.res[1])
        depth = sink_depth(z)
        with rasterio.open(OUT / f"{epoch}_depth.tif", "w", **profile) as dst:
            dst.write(np.where(np.isnan(depth), -9999.0, depth), 1)
        gdf = outlines(depth, transform, px_area)
        gdf.to_file(gpkg, layer=epoch, driver="GPKG")
        n = gdf.level_m.value_counts().sort_index().to_dict()
        print(f"{epoch}: outlines per level {n}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
