"""Error model for elevation differences (dh), after Hugonnet et al. (2022).

Shared by scripts/uncertainty_lod.py (LoD maps per epoch pair) and
scripts/change_at_dolines.py (change and volume per doline). Works on 1 m
grids: 20 cm rasters averaged 5x5.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xdem
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling


def load_avg(path: Path, factor: int = 5) -> tuple[np.ndarray, Affine]:
    """Read all bands of a raster block-averaged by `factor`, nodata -> NaN."""
    with rasterio.open(path) as src:
        shape = (src.count, src.height // factor, src.width // factor)
        arr = src.read(out_shape=shape, resampling=Resampling.average).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            arr[arr == src.nodata] = np.nan
        tf = src.transform * Affine.scale(src.width / shape[2], src.height / shape[1])
    return arr, tf


def error_model(dh: np.ndarray, dem_mean: np.ndarray, stable: np.ndarray, tf: Affine,
                crs: CRS, res: float = 1.0):  # fmt: skip
    """Per-pixel sigma(slope, max curvature) and the variogram of dh / sigma.

    Returns sigma, slope, the 1D/2D binning table and the empirical variogram
    plus the fitted Gaussian + spherical model parameters.
    """
    slope, maxc = xdem.terrain.get_terrain_attribute(
        dem_mean, attribute=["slope", "max_curvature"], resolution=res
    )
    # passed as a raster: xdem 0.2.3 takes the pixel size only from a raster here
    dh_r = xdem.DEM.from_array(dh, tf, crs, nodata=-9999.0)
    sigma, binning, _ = xdem.spatialstats.infer_heteroscedasticity_from_stable(
        dh_r, [slope, maxc], stable_mask=stable, list_var_names=["slope", "maxc"]
    )
    sigma = sigma.get_nanarray() if hasattr(sigma, "get_nanarray") else sigma
    emp, params, _ = xdem.spatialstats.infer_spatial_correlation_from_stable(
        dh, ["Gaussian", "Spherical"], stable_mask=stable, errors=sigma, gsd=res,
        subsample=2000, n_variograms=5, random_state=42,
    )  # fmt: skip
    return sigma, slope, binning, emp, params


def sigma_of_mean(area_m2: float, sigma_typ: float, params: pd.DataFrame) -> float:
    """1-sigma error of the mean dh over an area, via the effective sample count."""
    if not area_m2 > 0:
        return np.nan
    n_eff = xdem.spatialstats.number_effective_samples(area_m2, params)
    return float(sigma_typ / np.sqrt(max(n_eff, 1.0)))
