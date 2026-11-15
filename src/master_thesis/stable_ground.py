"""Masks for "ground that should not have changed", used for co-registration and
uncertainty. All functions work on arrays already on a common grid; `res` is
the pixel size in metres.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from scipy.ndimage import binary_dilation, binary_erosion


def nmad(x: np.ndarray) -> float:
    """Normalized median absolute deviation, the robust std of dh."""
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def median(x: np.ndarray) -> float:
    """Median of the finite values, NaN if there are none."""
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def inliers_of(dh: np.ndarray, mask: np.ndarray, k: float = 3.0, floor: float = 0.05) -> np.ndarray:
    """Masked px whose dh is within k NMAD (at least k * floor m) of the median.

    Drops snow, real change and blunders before fitting anything to dh.
    """
    return mask & (np.abs(dh - median(dh[mask])) < k * max(nmad(dh[mask]), floor))


def slope_deg(dem: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(dem, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def feature_mask(
    polygons: list[gpd.GeoSeries],
    points: list[gpd.GeoSeries],
    transform: Affine,
    shape: tuple[int, int],
    crs: CRS,
    poly_buffer: float = 10.0,
    point_buffer: float = 20.0,
) -> np.ndarray:
    """True within a buffer of every mapped feature (dolines, suspicious areas, ...).

    Points mark feature centres and dolines reach ~40 m across, hence the
    larger point buffer. Empty/missing geometries are skipped.
    """
    buffered = [s.to_crs(crs).buffer(poly_buffer) for s in polygons]
    buffered += [s.to_crs(crs).buffer(point_buffer) for s in points]
    geoms = [g for g in pd.concat(buffered) if g is not None and not g.is_empty]
    return geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


def snow_mask(rgb: np.ndarray, res: float, dilate_m: float = 2.0) -> np.ndarray:
    """True where an 8-bit RGB ortho (3, H, W) looks like snow.

    Snow here is mostly blue-grey (shade, overcast), not white, so plain
    brightness misses it. Blueness (B - R) / brightness separates it from the
    grey/brown limestone much better (snow ~+0.07..+0.17, rock < ~+0.02 on the
    Oct 2024 / Jun 2025 epochs). Deliberately aggressive: it also flags some
    rock, which only costs sample size.
    """
    r, g, b = (band.astype("float32") for band in rgb[:3])
    brightness = (r + g + b) / 3
    blueness = (b - r) / np.maximum(brightness, 1)
    snow = (brightness > 60) & (blueness > 0.04)
    return binary_dilation(snow, iterations=max(1, round(dilate_m / res)))


def erode_valid(valid: np.ndarray, res: float, edge_m: float = 5.0) -> np.ndarray:
    """Drop a strip along data edges, where resampling and LiDAR edges misbehave."""
    return binary_erosion(valid, iterations=max(1, round(edge_m / res)))
