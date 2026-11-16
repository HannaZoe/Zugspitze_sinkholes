"""Hillshades of the co-registered elevation models, as a mapping backdrop.

Doline rims and walls read much better in a hillshade than in the ortho,
especially under snow or shadow. One hillshade per co-registered LiDAR DSM
plus the DGM1, on the common 20 cm grid (EPSG:32632), uint8, 0 = nodata.
Sun from the north-west (azimuth 315 deg, altitude 45 deg), the cartographic
default. Output: data/interim/coregistered_20cm/hillshade/<name>_hillshade.tif.
"""

import numpy as np
import rasterio
import xdem
from rasterio.enums import Resampling

from master_thesis.paths import INTERIM

IN_DIR = INTERIM / "coregistered_20cm"
OUT_DIR = IN_DIR / "hillshade"
DEMS = [
    "20241025_terra_ppk_lidar_dsm",
    "20250604_terra_ppk_lidar_dsm",
    "20250820_terra_ppk_lidar_dsm",
    "bavaria_dgm1",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in DEMS:
        with rasterio.open(IN_DIR / f"{name}.tif") as src:
            dem, profile = src.read(1, masked=True).filled(np.nan), src.profile
        hs = xdem.terrain.hillshade(dem, resolution=src.res[0], azimuth=315, altitude=45)
        hs = np.where(np.isfinite(hs), np.clip(hs, 1, 255), 0).astype("uint8")
        profile.update(dtype="uint8", nodata=0, count=1, compress="deflate", predictor=2)
        with rasterio.open(OUT_DIR / f"{name}_hillshade.tif", "w", **profile) as dst:
            dst.write(hs, 1)
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
        print(f"  wrote {name}_hillshade.tif", flush=True)


if __name__ == "__main__":
    main()
