"""Rigid x/y alignment of rasters that already share a grid.

`tile_shifts` measures local offsets between two images with phase
correlation (skimage) tile by tile; the median is a robust rigid shift, the
spread between tiles shows whether one shift is enough (small) or the image
is warped (large). `shift_raster` applies a shift and resamples back onto the
same grid. Shifts are always "how far the target must move (east, north) in
metres to sit on the reference".
"""

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from skimage.registration import phase_cross_correlation


def grey(src: rasterio.DatasetReader, window: Window) -> np.ndarray:
    """Mean of the first three bands (or the only one), scaled to 0..1 for ints, NaN = nodata."""
    arr = src.read(indexes=[1, 2, 3] if src.count >= 3 else [1], window=window).astype("float32")
    valid = src.read_masks(1, window=window) > 0
    if np.dtype(src.dtypes[0]).kind != "f":
        arr = arr / np.iinfo(src.dtypes[0]).max
    g = arr.mean(axis=0)
    g[~valid | ~np.isfinite(g)] = np.nan
    return g


def tile_shifts(
    ref_src: rasterio.DatasetReader,
    tgt_src: rasterio.DatasetReader,
    window: int = 512,
    min_valid: float = 0.9,
    min_cv: float = 0.05,
    max_shift_m: float = 10.0,
) -> pd.DataFrame:
    """Per-tile shift of `tgt_src` onto `ref_src` (same grid) in metres.

    Tiles need `min_valid` overlap and some texture (grey std/mean >= `min_cv`,
    which skips snow fields and deep shadow); shifts above `max_shift_m` are
    treated as mismatches and dropped.
    """
    res = ref_src.res[0]
    rows = []
    for r0 in range(0, ref_src.height - window, window):
        for c0 in range(0, ref_src.width - window, window):
            win = Window(c0, r0, window, window)
            if not ref_src.read_masks(1, window=win).any():
                continue
            a, b = grey(ref_src, win), grey(tgt_src, win)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.mean() < min_valid:
                continue
            if min(x[ok].std() / max(abs(x[ok].mean()), 1e-6) for x in (a, b)) < min_cv:
                continue
            a, b = (np.where(ok, x - np.nanmean(x), 0) for x in (a, b))
            (sr, sc), error, _ = phase_cross_correlation(a, b, upsample_factor=10)
            dx, dy = sc * res, -sr * res  # rows grow southwards
            if np.hypot(dx, dy) > max_shift_m:
                continue
            x, y = ref_src.xy(r0 + window // 2, c0 + window // 2)
            rows.append({"x": x, "y": y, "dx_m": dx, "dy_m": dy, "error": error})
    return pd.DataFrame(rows)


def shift_raster(
    arr: np.ndarray, transform: Affine, crs: CRS, nodata: float, dx: float, dy: float
) -> np.ndarray:
    """Move a (bands, rows, cols) array by (dx, dy) metres, resampled back onto `transform`."""
    out = np.full_like(arr, nodata)
    src_tf = Affine(transform.a, transform.b, transform.c + dx, transform.d, transform.e,
                    transform.f + dy)  # fmt: skip
    reproject(arr, out, src_transform=src_tf, src_crs=crs, src_nodata=nodata,
              dst_transform=transform, dst_crs=crs, dst_nodata=nodata,
              resampling=Resampling.bilinear)  # fmt: skip
    return out
