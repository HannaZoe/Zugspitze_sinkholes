"""Hillshades of the co-registered elevation models, as a mapping backdrop.

Doline rims and walls read much better in a hillshade than in the ortho,
especially under snow or shadow. One hillshade per co-registered LiDAR DSM
plus the DGM1, on the common 20 cm grid (EPSG:32632), uint8, 0 = nodata.
Sun from the north-west (azimuth 315 deg, altitude 45 deg), the cartographic
default. Output: data/interim/coregistered_20cm/hillshade/<name>_hillshade.tif.

For the UAV LiDAR epochs two light-independent extras, because a single sun
direction shifts where a rim appears (walls facing the light look flatter):
- <name>_hillshade_multi.tif: mean of 6 azimuths (0, 60, ..., 300), alt 45
- <name>_slope.tif: slope in degrees (uint8, 0-90, nodata 255). The rim
  (upper break of slope) is where flat turns steep, no light involved.
"""

import numpy as np
import rasterio
import xdem
from rasterio.enums import Resampling

from master_thesis.paths import INTERIM

IN_DIR = INTERIM / "coregistered_20cm"
OUT_DIR = IN_DIR / "hillshade"
MULTI_AZIMUTHS = range(0, 360, 60)
LIDAR = {"20241025_terra_ppk_lidar_dsm", "20250604_terra_ppk_lidar_dsm",
         "20250820_terra_ppk_lidar_dsm"}  # fmt: skip
DEMS = [
    "20241025_terra_ppk_lidar_dsm",
    "20250604_terra_ppk_lidar_dsm",
    "20250820_terra_ppk_lidar_dsm",
    "bavaria_dgm1",
]


def write(path, arr, profile, nodata) -> None:
    profile = profile | {"dtype": "uint8", "nodata": nodata, "count": 1,
                         "compress": "deflate", "predictor": 2}  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
    print(f"  wrote {path.name}", flush=True)


def to_uint8_shade(hs: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(hs), np.clip(hs, 1, 255), 0).astype("uint8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in DEMS:
        with rasterio.open(IN_DIR / f"{name}.tif") as src:
            dem, profile, res = src.read(1, masked=True).filled(np.nan), src.profile, src.res[0]
        hs = xdem.terrain.hillshade(dem, resolution=res, azimuth=315, altitude=45)
        write(OUT_DIR / f"{name}_hillshade.tif", to_uint8_shade(hs), profile, 0)
        if name not in LIDAR:
            continue
        multi = np.mean([xdem.terrain.hillshade(dem, resolution=res, azimuth=a, altitude=45)
                         for a in MULTI_AZIMUTHS], axis=0)  # fmt: skip
        write(OUT_DIR / f"{name}_hillshade_multi.tif", to_uint8_shade(multi), profile, 0)
        slope = xdem.terrain.slope(dem, resolution=res)
        slope = np.where(np.isfinite(slope), np.clip(np.round(slope), 0, 90), 255).astype("uint8")
        write(OUT_DIR / f"{name}_slope.tif", slope, profile, 255)


if __name__ == "__main__":
    main()
