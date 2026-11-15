"""Put every raster we have on one shared 20 cm grid, for side-by-side viewing.

Inputs differ in resolution (2 cm orthos ... 1 m DGM1) and CRS (EPSG:32632
UAV products, EPSG:4326 Agisoft exports, EPSG:25832 DOP/DGM1). This writes
each one onto the *same* grid: EPSG:32632, RES m pixels, the union of all UAV
footprints snapped to RES, so layers can be toggled in QGIS pixel-on-pixel
and stacked as arrays later.

- Downsampling uses `average` (DSMs and imagery alike), the only upsampling
  (DGM1, 1 m -> 20 cm) uses `bilinear` -- it still only carries 1 m detail.
- EPSG:25832 -> 32632 is a null datum shift in PROJ. That is fine here: the
  20250820 LiDAR DSM sits within ~1 cm of the DGM1 horizontally (see
  check_lidar_stability.py), i.e. the UAV data is effectively ETRS89.
- Heights are passed through untouched: UAV DSMs are ellipsoidal, DGM1 is
  DHHN2016 (~49.7 m lower here). Vertical alignment is the co-registration
  step's job, not this one's.
- Nothing is co-registered here. Offsets you see between layers are real
  georeferencing differences -- which is the point of looking.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from master_thesis.paths import INTERIM, RAW

CRS = rasterio.crs.CRS.from_epsg(32632)
RES = 0.2  # m, coarsest of the imagery (20260827 epoch, DOP20)
OUT_DIR = INTERIM / "common_grid_20cm"
UAV_DATES = ["20241025", "20250604", "20250606", "20250717", "20250820", "20260827"]


def common_grid():
    """Union of all UAV footprints in EPSG:32632, snapped outward to RES."""
    bounds = []
    for d in UAV_DATES:
        for f in (RAW / d).glob("*.tif"):
            with rasterio.open(f) as src:
                bounds.append(transform_bounds(src.crs, CRS, *src.bounds))
    left, bottom = (np.floor(min(b[i] for b in bounds) / RES) * RES for i in (0, 1))
    right, top = (np.ceil(max(b[i] for b in bounds) / RES) * RES for i in (2, 3))
    shape = (round((top - bottom) / RES), round((right - left) / RES))
    return from_origin(left, top, RES, RES), shape


def nodata_for(src):
    if src.nodata is not None:
        return src.nodata
    return np.nan if np.dtype(src.dtypes[0]).kind == "f" else 0


def write_on_grid(src, out_path, transform, shape, resampling):
    """Warp an open dataset onto the common grid, keeping dtype and band count."""
    nodata = nodata_for(src)
    vrt_opts = dict(crs=CRS, transform=transform, width=shape[1], height=shape[0],
                    resampling=resampling, nodata=nodata, warp_mem_limit=2048,
                    num_threads="ALL_CPUS")  # fmt: skip
    with WarpedVRT(src, **vrt_opts) as vrt:
        profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=src.count,
                       dtype=src.dtypes[0], crs=CRS, transform=transform, nodata=nodata,
                       compress="deflate", predictor=2, tiled=True, BIGTIFF="IF_SAFER")  # fmt: skip
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                dst.write(vrt.read(window=window), window=window)
            dst.colorinterp = src.colorinterp
            dst.build_overviews([2, 4, 8, 16], Resampling.average)


def main() -> None:
    transform, shape = common_grid()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"common grid: {shape[1]} x {shape[0]} px at {RES} m, UL ({transform.c}, {transform.f})")

    sources = [f for d in UAV_DATES for f in sorted((RAW / d).glob("*.tif"))]
    sources += sorted((RAW / "bavaria_dop_hist").rglob("*.tif"))
    for f in sources:
        out = OUT_DIR / f.name
        if out.exists():
            print(f"  skip {f.name} (exists)")
            continue
        with rasterio.open(f) as src:
            write_on_grid(src, out, transform, shape, Resampling.average)
        print(f"  wrote {f.name}")

    # DGM1 comes as 1 km tiles: mosaic in memory, then warp like the rest
    out = OUT_DIR / "bavaria_dgm1.tif"
    if not out.exists():
        tiles = [rasterio.open(p) for p in sorted((RAW / "bavaria_dgm1").glob("*.tif"))]
        mosaic, mosaic_tf = merge(tiles, nodata=-9999.0)
        profile = tiles[0].profile | dict(height=mosaic.shape[1], width=mosaic.shape[2],
                                          transform=mosaic_tf, driver="GTiff")  # fmt: skip
        for t in tiles:
            t.close()
        with rasterio.MemoryFile() as mem, mem.open(**profile) as src:
            src.write(mosaic)
            write_on_grid(src, out, transform, shape, Resampling.bilinear)
        print("  wrote bavaria_dgm1.tif")


if __name__ == "__main__":
    main()
