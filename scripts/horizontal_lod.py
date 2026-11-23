"""Smallest detectable horizontal displacement between the co-registered RGB orthos.

The horizontal counterpart of the height LoD: after co-registration, how far do
two orthos still disagree in x/y on ground that has not moved? Phase
correlation (master_thesis.image_alignment.tile_shifts) in WINDOW-px tiles
(40 m) over the overlap of each pair, at 1/100 pixel precision; tiles
without texture (snow, shadow) or with implausible shifts are dropped. Per pair:
- median dx / dy: remaining bias (should be ~0)
- sigma: NMAD of dx and dy, averaged
- LoD95 = 2.45 sigma: radius of the 95 % circle of a 2D normal error
- p95 |d|: empirical 95th percentile of the displacement length
This is registration only, measured at the scale of a 40 m tile. Drawing a rim
by hand adds its own (larger) uncertainty, to be measured by repeat digitising.
EPSG:32632, 20 cm grid. Outputs: outputs/figures/horizontal_lod.csv and .png.
"""

import matplotlib
import numpy as np
import pandas as pd
import rasterio

from master_thesis.image_alignment import tile_shifts
from master_thesis.paths import FIGURES, INTERIM

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CO = INTERIM / "coregistered_20cm"
WINDOW = 200  # px = 40 m
UAV = {"Oct 2024": "20241025_terra_ppk_rgb_om", "Jun 2025": "20250604_terra_ppk_rgb_om",
       "Aug 2025": "20250820_terra_ppk_rgb_om",
       "Aug 2026": "20260827_terra_rtk_rgb_om_20cm_local_coregistration"}  # fmt: skip
DOPS = ["2006-07-02", "2009-09-01", "2012-07-27", "2015-07-01", "2018-09-11", "2020-07-08",
        "2022-06-12", "2024-07-30"]  # fmt: skip
PAIRS = [("Oct 2024", "Aug 2025"), ("Jun 2025", "Aug 2025"), ("Aug 2025", "Aug 2026"),
         ("Oct 2024", "Jun 2025"), ("Oct 2024", "Aug 2026")]  # fmt: skip


def nmad(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def pair_stats(ref: str, tgt: str) -> tuple[dict, pd.DataFrame]:
    with rasterio.open(CO / f"{ref}.tif") as a, rasterio.open(CO / f"{tgt}.tif") as b:
        t = tile_shifts(a, b, window=WINDOW, max_shift_m=2.0, upsample=100)
    d = np.hypot(t.dx_m, t.dy_m)
    sigma = (nmad(t.dx_m.to_numpy()) + nmad(t.dy_m.to_numpy())) / 2
    stats = {
        "n_tiles": len(t),
        "median_dx_m": t.dx_m.median(),
        "median_dy_m": t.dy_m.median(),
        "sigma_m": sigma,
        "lod95_m": 2.45 * sigma,
        "p95_m": float(np.percentile(d, 95)),
    }
    return stats, t


def main() -> None:
    rows, tiles = [], {}
    pairs = [(UAV[a], UAV[b], f"{a} - {b}", "UAV") for a, b in PAIRS]
    pairs += [
        (UAV["Aug 2025"], f"bavaria_dop20_{d}", f"DOP {d[:4]} - Aug 2025", "DOP") for d in DOPS
    ]
    for ref, tgt, label, kind in pairs:
        stats, t = pair_stats(ref, tgt)
        rows.append({"pair": label, "kind": kind} | stats)
        tiles[label] = t
        print(
            f"  {label}: n={stats['n_tiles']}, sigma {stats['sigma_m']:.3f} m, "
            f"LoD95 {stats['lod95_m']:.2f} m, p95 {stats['p95_m']:.2f} m",
            flush=True,
        )
    out = pd.DataFrame(rows).round(3)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out.to_csv(FIGURES / "horizontal_lod.csv", index=False)

    fig, axs = plt.subplots(2, 7, figsize=(21, 6.5), sharex=True, sharey=True)
    for ax, (label, t) in zip(axs.ravel(), tiles.items(), strict=False):
        ax.scatter(t.dx_m, t.dy_m, s=6, alpha=0.6)
        lod = out.loc[out.pair == label, "lod95_m"].iloc[0]
        ax.add_patch(plt.Circle((0, 0), lod, fill=False, color="tab:red"))
        ax.axhline(0, c="grey", lw=0.5), ax.axvline(0, c="grey", lw=0.5)
        ax.set_title(f"{label}\nLoD95 {lod:.2f} m", fontsize=9)
        ax.set_aspect("equal")
    for ax in axs.ravel()[len(tiles) :]:
        ax.axis("off")
    lim = 1.0
    axs[0, 0].set_xlim(-lim, lim), axs[0, 0].set_ylim(-lim, lim)
    fig.supxlabel("residual shift east [m]"), fig.supylabel("residual shift north [m]")
    fig.suptitle(
        "Horizontal residuals of the co-registered RGB orthos (40 m tiles); red: LoD95 circle"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "horizontal_lod.png", dpi=110)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
