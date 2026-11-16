"""Independent x/y check: residual offset of every ortho vs the 20250820 ortho.

The DSM-based co-registration moves each ortho by one rigid shift. This checks
the result from the image side: phase correlation (skimage) in WINDOW-px
tiles spread over the overlap, each giving a local shift. Median ~0 means the
rigid shift worked; a large spread between tiles means the dataset is warped
(a single shift can't fix that -- local methods like AROSICS would be needed).

Runs on both grids, so before/after can be compared: common_grid_20cm (raw)
and coregistered_20cm. The historical DOP20s are only on the raw grid; their
numbers show whether they need aligning at all. Shifts are reported the same
way as in coregister_to_reference.py: how far the image would still have to
move (east, north) to sit on the reference. EPSG:32632, 20 cm grid.
"""

import matplotlib.pyplot as plt
import pandas as pd
import rasterio

from master_thesis.image_alignment import tile_shifts
from master_thesis.paths import FIGURES, INTERIM

WINDOW = 512  # px, ~100 m
MIN_VALID = 0.9  # fraction of valid px a tile needs in both images
MIN_CV = 0.05  # std/mean of grey values, skips featureless tiles (snow, shadow)
MAX_SHIFT_M = 10.0  # anything bigger is a mismatch, not a shift
LIM_M = 3.0  # axis range of the plot
REF = "20250820_terra_ppk_rgb_om"
# thermal IR does not correlate with RGB; its 3T RGB sibling stands in for it
SKIP = ("thermal_ir_om",)
OUT_TABLE = FIGURES / "ortho_alignment.csv"
OUT_TILES = FIGURES / "ortho_alignment_tiles.csv"
OUT_FIG = FIGURES / "ortho_alignment.png"


def main() -> None:
    before, after = INTERIM / "common_grid_20cm", INTERIM / "coregistered_20cm"
    targets = [("before", p) for p in sorted(before.glob("*_om*.tif"))]
    targets += [("before", p) for p in sorted(before.glob("bavaria_dop20_*.tif"))]
    targets += [("after", p) for p in sorted(after.glob("*_om*.tif"))]
    targets += [("after", p) for p in sorted(after.glob("bavaria_dop20_*.tif"))]
    targets = [(s, p) for s, p in targets if p.stem != REF and not any(k in p.stem for k in SKIP)]

    summary, tiles = [], []
    with rasterio.open(after / f"{REF}.tif") as ref_src:
        for stage, path in targets:
            with rasterio.open(path) as tgt_src:
                t = tile_shifts(ref_src, tgt_src, WINDOW, MIN_VALID, MIN_CV, MAX_SHIFT_M)
            if t.empty:
                print(f"  {stage:6s} {path.stem}: no usable tiles")
                continue
            t["image"], t["stage"] = path.stem, stage
            tiles.append(t)
            q = t[["dx_m", "dy_m"]].quantile([0.25, 0.5, 0.75])
            summary.append({
                "image": path.stem, "stage": stage, "n_tiles": len(t),
                "median_dx_m": q.loc[0.5, "dx_m"], "median_dy_m": q.loc[0.5, "dy_m"],
                "iqr_dx_m": q.loc[0.75, "dx_m"] - q.loc[0.25, "dx_m"],
                "iqr_dy_m": q.loc[0.75, "dy_m"] - q.loc[0.25, "dy_m"],
            })  # fmt: skip
            s = summary[-1]
            print(f"  {stage:6s} {path.stem[:48]:48s} n={len(t):3d} median "
                  f"({s['median_dx_m']:+.2f}, {s['median_dy_m']:+.2f}) m, "
                  f"IQR ({s['iqr_dx_m']:.2f}, {s['iqr_dy_m']:.2f}) m", flush=True)  # fmt: skip

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT_TABLE, index=False, float_format="%.3f")
    tiles = pd.concat(tiles, ignore_index=True)
    tiles.to_csv(OUT_TILES, index=False, float_format="%.3f")

    # one panel per image: every tile's residual shift as a dot, before (grey)
    # and after (blue). A tight blue cloud at 0 = aligned; a wide one = warped.
    images = sorted(tiles.image.unique())
    ncol = 4
    nrow = -(-len(images) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
    for ax, name in zip(axes.flat, images, strict=False):
        for stage, colour in [("before", "0.6"), ("after", "tab:blue")]:
            t = tiles[(tiles.image == name) & (tiles.stage == stage)]
            ax.scatter(t.dx_m, t.dy_m, s=8, c=colour, label=f"{stage} (n={len(t)})")
        ax.axhline(0, c="k", lw=0.5)
        ax.axvline(0, c="k", lw=0.5)
        ax.set(xlim=(-LIM_M, LIM_M), ylim=(-LIM_M, LIM_M), aspect="equal")
        ax.set_title(name.replace("_20cm_local_coregistration", "")[:34], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower left")
    for ax in axes.flat[len(images) :]:
        ax.set_axis_off()
    fig.supxlabel("residual shift east [m]")
    fig.supylabel("residual shift north [m]")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=110)
    print(f"wrote {OUT_TABLE}, {OUT_TILES}, {OUT_FIG}")


if __name__ == "__main__":
    main()
