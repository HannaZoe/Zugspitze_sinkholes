"""Sensitivity test: which bias model does the 20260827 DSMs need?

After VerticalShift + Deramp(1) + NuthKaab (coregister_to_reference.py) both
20260827 DSMs still show a large-scale, non-planar residual (~+-1 m, dome/
bowl-like), and their N&K x/y shift (~0.3 m) disagrees with the ortho-based
one (~0). Candidates, all against the 20250820 LiDAR DSM on the common 20 cm
grid (EPSG:32632):

  A  VS + Deramp(1) + NK                (current)
  B  VS + Deramp(2) + NK                (literature: order 2 at most)
  C  VS + Deramp(1) + NK + elevation bias (Nuth & Kaab 2011 step ii)
  D  VS + Deramp(2) + NK + elevation bias

Fair comparison: stable ground split into a 100 m checkerboard, models fit on
the black squares and scored (median, NMAD) on the white ones. Exploration
only -- does not write any rasters.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xdem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coregister_to_reference import (  # noqa: E402
    CRS,
    IN_DIR,
    REF,
    RES,
    footprint,
    horizontal_shift,
    read,
    stable_base,
    window_transform,
)  # fmt: skip

from master_thesis.paths import FIGURES  # noqa: E402
from master_thesis.stable_ground import erode_valid, inliers_of, median, nmad  # noqa: E402

DSMS = {
    "3m": "20260827_terra_rtk_dsm_20cm_local_coregistration",
    "3t": "20260827_terra_rtk_thermal_dsm_20cm_local_coregistration",
}
BLOCK = 500  # px = 100 m checkerboard


def models():
    vs, nk = xdem.coreg.VerticalShift, xdem.coreg.NuthKaab
    ramp = xdem.coreg.Deramp

    def elev():
        return xdem.coreg.TerrainBias(terrain_attribute="elevation")

    return {
        "A ramp1": lambda: vs() + ramp(poly_order=1) + nk(),
        "B ramp2": lambda: vs() + ramp(poly_order=2) + nk(),
        "C ramp1+elev": lambda: vs() + ramp(poly_order=1) + nk() + elev(),
        "D ramp2+elev": lambda: vs() + ramp(poly_order=2) + nk() + elev(),
    }


def main() -> None:
    import rasterio

    with rasterio.open(IN_DIR / f"{REF}.tif") as src:
        grid_profile, grid_shape = src.profile, src.shape

    rows = []
    for label, name in DSMS.items():
        win = footprint([name], grid_shape)
        tf = window_transform(win, grid_profile["transform"])
        ref, dsm = read(REF, win)[0][0], read(name, win)[0][0]
        mask = stable_base(ref, tf, ref.shape) & erode_valid(
            np.isfinite(dsm) & np.isfinite(ref), RES
        )
        mask = inliers_of(dsm - ref, mask)
        rr, cc = np.indices(ref.shape)
        black = (rr // BLOCK + cc // BLOCK) % 2 == 0
        fit_mask, test_mask = mask & black, mask & ~black

        for model_name, make in models().items():
            try:
                coreg = make().fit(ref, dsm, inlier_mask=fit_mask, transform=tf, crs=CRS,
                                   subsample=1_000_000, random_state=42)  # fmt: skip
                dh = coreg.apply(dsm, transform=tf, crs=CRS)[0] - ref
            except Exception as err:  # some combos may not be supported by xdem
                print(f"  {label} {model_name}: failed ({err})")
                continue
            dx, dy = horizontal_shift(coreg)
            rows.append({
                "dsm": label, "model": model_name, "dx_m": dx, "dy_m": dy,
                "test_median_m": median(dh[test_mask]), "test_nmad_m": nmad(dh[test_mask]),
                "train_nmad_m": nmad(dh[fit_mask]),
            })  # fmt: skip
            print(f"  {label} {model_name:14s} dx {dx:+.2f} dy {dy:+.2f}  "
                  f"test NMAD {rows[-1]['test_nmad_m']:.3f} m", flush=True)  # fmt: skip

    out = pd.DataFrame(rows)
    out.to_csv(FIGURES / "bias_models_20260827.csv", index=False, float_format="%.3f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
