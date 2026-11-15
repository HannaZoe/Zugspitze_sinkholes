"""Quick-look of coregister_to_reference.py: dh to the reference, before vs after.

One row per DSM, left = raw (common 20 cm grid), right = co-registered, both
minus that chain's reference (20250820 LiDAR DSM, or the registered DGM1 for
the lower-route flights), median removed for display (DGM1 is ~50 m off
before). Each row is cropped to the DSM's own data and plotted at 1 m.
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

from master_thesis.paths import FIGURES, INTERIM

BEFORE, AFTER = INTERIM / "common_grid_20cm", INTERIM / "coregistered_20cm"
REF = "20250820_terra_ppk_lidar_dsm"
VIA_DGM1 = ("20250606", "20250717")  # see coregister_to_reference.VIA_DGM1
FACTOR = 5  # 20 cm -> 1 m for display
LIMIT = 1.0  # colour range, m
OUT = FIGURES / "coregistration_before_after.png"


def load(path):
    with rasterio.open(path) as src:
        shape = (src.height // FACTOR, src.width // FACTOR)
        arr = src.read(1, out_shape=shape, resampling=Resampling.average).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            arr[arr == src.nodata] = np.nan
    return arr


def main() -> None:
    refs = {"lidar": load(AFTER / f"{REF}.tif"), "dgm1": load(AFTER / "bavaria_dgm1.tif")}
    dsms = sorted(p.stem for p in AFTER.glob("*.tif") if "dsm" in p.stem or "dgm1" in p.stem)
    dsms = [d for d in dsms if d != REF]

    fig, axes = plt.subplots(len(dsms), 2, figsize=(14, 3 * len(dsms)), squeeze=False)
    for name, (ax_b, ax_a) in zip(dsms, axes, strict=True):
        ref_key = "dgm1" if name.startswith(VIA_DGM1) else "lidar"
        ref = refs[ref_key]
        after = load(AFTER / f"{name}.tif")
        r, c = np.nonzero(np.isfinite(after) & np.isfinite(ref))
        crop = np.s_[r.min() : r.max() + 1, c.min() : c.max() + 1]
        for ax, folder, label in [(ax_b, BEFORE, "before"), (ax_a, AFTER, "after")]:
            dh = (load(folder / f"{name}.tif") - ref)[crop]
            med = np.nanmedian(dh)
            im = ax.imshow(dh - med, cmap="RdBu_r", vmin=-LIMIT, vmax=LIMIT)
            ax.set_title(f"{name[:40]} - {ref_key}, {label} (median {med:+.2f} m removed)",
                         fontsize=9)  # fmt: skip
            ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.3, label="dh - median [m]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
