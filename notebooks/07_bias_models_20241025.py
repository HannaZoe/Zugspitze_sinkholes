"""Sensitivity test: can the east-end bend of the 20241025 DSMs be corrected?

After Nuth & Kaab (coregister_to_reference.py) both 20241025 DSMs (LiDAR and
RGB photogrammetry, same flight) agree with the 20250820 LiDAR to a few cm,
except over the last ~400 m in the east, where they rise to ~+0.2 m (see the
experiment log, 2026-11-16). Both DSMs show it, so it is most likely a GNSS
trajectory error of that flight. Candidate corrections, all against the
20250820 LiDAR DSM on the common 20 cm grid (EPSG:32632):

  A  NK                          (current)
  B  Deramp(1) + NK
  C  Deramp(2) + NK              (literature: order 2 at most)
  D  NK + along-strip bias       (DirectionalBias, binned median along the strip)
  E  NK + across-strip bias      (control: should not help)

The strip runs ~10 deg south of east. Fair comparison as in notebook 06:
stable ground (incl. the 20241025 snow mask) split into a 100 m checkerboard,
fit on black, scored on white. Also reported: the median residual of the test
pixels in 100 m strips at the east end, which is where the bend sits.
Exploration only -- writes no rasters.
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
from master_thesis.stable_ground import (  # noqa: E402
    erode_valid,
    inliers_of,
    median,
    nmad,
    snow_mask,
)  # fmt: skip

DSMS = {"lidar": "20241025_terra_ppk_lidar_dsm", "rgb": "20241025_terra_ppk_rgb_dsm"}
SNOW_FROM = "20241025_terra_ppk_rgb_om"
BLOCK = 500  # px = 100 m checkerboard
STRIP_ANGLE = -10  # deg, strip direction (from east, counter-clockwise)
EAST_STRIPS = range(651000, 651500, 100)  # easting of the 100 m strips reported


def models():
    nk, ramp = xdem.coreg.NuthKaab, xdem.coreg.Deramp

    def along(angle):
        return xdem.coreg.DirectionalBias(angle=angle, fit_or_bin="bin", bin_sizes=30)

    return {
        "A nk": lambda: nk(),
        "B ramp1+nk": lambda: ramp(poly_order=1) + nk(),
        "C ramp2+nk": lambda: ramp(poly_order=2) + nk(),
        "D nk+along": lambda: nk() + along(STRIP_ANGLE),
        "E nk+across": lambda: nk() + along(STRIP_ANGLE + 90),
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
        mask &= ~snow_mask(read(SNOW_FROM, win)[0], RES)
        mask = inliers_of(dsm - ref, mask)
        rr, cc = np.indices(ref.shape)
        black = (rr // BLOCK + cc // BLOCK) % 2 == 0
        fit_mask, test_mask = mask & black, mask & ~black
        easting = tf.c + (cc + 0.5) * tf.a

        for model_name, make in models().items():
            try:
                coreg = make().fit(ref, dsm, inlier_mask=fit_mask, transform=tf, crs=CRS,
                                   subsample=1_000_000, random_state=42)  # fmt: skip
                dh = coreg.apply(dsm, transform=tf, crs=CRS)[0] - ref
            except Exception as err:  # some combos may not be supported by xdem
                print(f"  {label} {model_name}: failed ({err})")
                continue
            dx, dy = horizontal_shift(coreg)
            row = {
                "dsm": label, "model": model_name, "dx_m": dx, "dy_m": dy,
                "test_median_m": median(dh[test_mask]), "test_nmad_m": nmad(dh[test_mask]),
                "train_nmad_m": nmad(dh[fit_mask]),
            }  # fmt: skip
            for e0 in EAST_STRIPS:
                strip = test_mask & (easting >= e0) & (easting < e0 + 100)
                row[f"E{e0 // 100 % 100}"] = median(dh[strip])
            rows.append(row)
            strips = " ".join(f"{row[f'E{e0 // 100 % 100}']:+.2f}" for e0 in EAST_STRIPS)
            print(f"  {label} {model_name:12s} test NMAD {row['test_nmad_m']:.3f} m, "
                  f"east strips {strips}", flush=True)  # fmt: skip

    out = pd.DataFrame(rows)
    out.to_csv(FIGURES / "bias_models_20241025.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
