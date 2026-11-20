"""Relief layers to trace moraines and glacier limits on the Zugspitzplatt.

Uses the Bavarian DGM1 (bare-earth LiDAR DTM, 1 m, EPSG:25832, CC BY 4.0), all
tiles in data/raw/bavaria_dgm1/ (study area plus the Schneeferner and summit
area to the north and west), mosaicked. Outputs in data/processed/glacial/
(EPSG:25832; QGIS reprojects to 32632 on the fly, < 1 m difference here):
- dgm1_mosaic.tif
- lrm.tif: local relief model (Hesse 2010 idea): DTM minus a Gaussian low-pass
  (sigma SIGMA_M), so ridges like moraine walls come out positive and hollows
  negative, independent of the overall slope. Moraine ridges here are a few
  metres high and tens of metres wide, which a 15 m sigma keeps.
- hillshade_multi.tif: multidirectional hillshade (6 azimuths, alt 45), so no
  ridge orientation is hidden by a single light direction.
"""

import numpy as np
import rasterio
from rasterio.merge import merge
from scipy import ndimage

from master_thesis.paths import PROCESSED, RAW

SIGMA_M = 15.0
OUT = PROCESSED / "glacial"


def hillshade(z: np.ndarray, res: float, azimuth: float, altitude: float = 45.0) -> np.ndarray:
    dzdy, dzdx = np.gradient(z, -res, res)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    return np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = sorted((RAW / "bavaria_dgm1").glob("*.tif"))
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, tf = merge(srcs, nodata=-9999.0)
    profile = srcs[0].profile | {"height": mosaic.shape[1], "width": mosaic.shape[2],
                                 "transform": tf, "nodata": -9999.0, "dtype": "float32",
                                 "compress": "deflate", "count": 1}  # fmt: skip
    for s in srcs:
        s.close()
    z = mosaic[0].astype("f8")
    valid = z > -9000
    z[~valid] = np.nan
    res = tf.a
    filled = np.where(valid, z, 0)
    w = ndimage.gaussian_filter(valid.astype(float), SIGMA_M / res)
    low = ndimage.gaussian_filter(filled, SIGMA_M / res) / np.where(w > 0.2, w, np.nan)
    lrm = z - low
    zf = np.where(valid, z, np.nanmedian(z))
    hs = np.mean([hillshade(zf, res, a) for a in range(0, 360, 60)], axis=0)
    for name, arr in [("dgm1_mosaic", z), ("lrm", lrm), ("hillshade_multi", hs)]:
        out = np.where(valid & np.isfinite(arr), arr, -9999.0).astype("float32")
        with rasterio.open(OUT / f"{name}.tif", "w", **profile) as dst:
            dst.write(out, 1)
    bounds = rasterio.transform.array_bounds(*z.shape, tf)
    print(f"{len(tiles)} tiles, mosaic {z.shape}, bounds {bounds}")
    print(f"LRM p1/p99: {np.nanpercentile(lrm, [1, 99]).round(2)} m; wrote {OUT}")


if __name__ == "__main__":
    main()
