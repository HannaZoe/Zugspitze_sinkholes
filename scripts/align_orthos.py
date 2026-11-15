"""Rigid x/y alignment of every ortho to the 20250820 ortho by image matching.

Runs after coregister_to_reference.py. That script moves orthos with the
x/y shift of their DSM, but the ortho check showed this is unreliable where
the DSM is weak (Altum multispectral DSMs) or still distorted in z
(20260827, where it undid Elio's AROSICS alignment). So orthos are aligned
on their own terms instead: median phase-correlation shift over ~100 m tiles
(master_thesis.image_alignment) against the reference ortho, applied to the
raw common-grid ortho. DSMs keep their DSM-based registration.

- Orthos with < MIN_TILES usable tiles (no overlap: 20250606, 20250717) keep
  the DSM-based shift already written by coregister_to_reference.py.
- Thermal IR doesn't correlate with RGB; it gets the shift of the 3T RGB ortho
  from the same flight.
- The historical DOP20s (no DSM) are handled the same way, written as new
  files next to the UAV ones.

One rigid shift per image: the tile IQR in the report says whether that is
enough (cm-dm) or the image is warped (dm-m -> would need local AROSICS).
EPSG:32632, common 20 cm grid. Output: data/interim/coregistered_20cm/,
report ortho_shifts.csv in the same folder.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from master_thesis.image_alignment import shift_raster, tile_shifts
from master_thesis.paths import INTERIM

IN_DIR, OUT_DIR = INTERIM / "common_grid_20cm", INTERIM / "coregistered_20cm"
REF = "20250820_terra_ppk_rgb_om"
MIN_TILES = 8
PAD = 250  # px
SIBLINGS = {  # follower: leader (same flight, can't be matched itself)
    "20260827_terra_rtk_thermal_ir_om_20cm_local_coregistration":
        "20260827_terra_rtk_thermal_rgb_om_20cm_local_coregistration",
}  # fmt: skip


def data_window(src: rasterio.DatasetReader) -> Window:
    """Bounding window of valid data, padded (read from a 1/20 overview)."""
    f = 20
    r, c = np.nonzero(src.dataset_mask(out_shape=(src.height // f, src.width // f)) > 0)
    r0, c0 = max(r.min() * f - PAD, 0), max(c.min() * f - PAD, 0)
    r1, c1 = min((r.max() + 1) * f + PAD, src.height), min((c.max() + 1) * f + PAD, src.width)
    return Window(c0, r0, c1 - c0, r1 - r0)


def write_shifted(name: str, dx: float, dy: float) -> None:
    with rasterio.open(IN_DIR / f"{name}.tif") as src:
        win = data_window(src)
        arr, tf, profile = src.read(window=win), src.window_transform(win), src.profile
        nodata = src.nodata if src.nodata is not None else 0
        colorinterp = src.colorinterp
    out = shift_raster(arr, tf, src.crs, nodata, dx, dy)
    with rasterio.open(OUT_DIR / f"{name}.tif", "w", **profile) as dst:
        dst.write(out, window=win)
        dst.colorinterp = colorinterp
        dst.build_overviews([2, 4, 8, 16], Resampling.average)


def main() -> None:
    names = sorted(p.stem for p in IN_DIR.glob("*_om*.tif")) + sorted(
        p.stem for p in IN_DIR.glob("bavaria_dop20_*.tif")
    )
    names = [n for n in names if n != REF and n not in SIBLINGS]

    rows = []
    with rasterio.open(OUT_DIR / f"{REF}.tif") as ref:
        for name in names:
            with rasterio.open(IN_DIR / f"{name}.tif") as tgt:
                t = tile_shifts(ref, tgt)
            if len(t) < MIN_TILES:
                rows.append({"image": name, "method": "dsm (no overlap)", "n_tiles": len(t)})
                print(f"  {name}: {len(t)} tiles, keeping the DSM-based shift", flush=True)
                continue
            q = t[["dx_m", "dy_m"]].quantile([0.25, 0.5, 0.75])
            dx, dy = q.loc[0.5, "dx_m"], q.loc[0.5, "dy_m"]
            followers = [f for f, leader in SIBLINGS.items() if leader == name]
            for n in [name, *followers]:
                write_shifted(n, dx, dy)
                rows.append({
                    "image": n, "method": "image" if n == name else f"image (via {name})",
                    "n_tiles": len(t), "dx_m": dx, "dy_m": dy,
                    "iqr_dx_m": q.loc[0.75, "dx_m"] - q.loc[0.25, "dx_m"],
                    "iqr_dy_m": q.loc[0.75, "dy_m"] - q.loc[0.25, "dy_m"],
                })  # fmt: skip
            print(f"  {name[:50]:50s} n={len(t):3d} shift ({dx:+.2f}, {dy:+.2f}) m", flush=True)

    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "ortho_shifts.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
