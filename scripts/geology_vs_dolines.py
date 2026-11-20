"""Do the mapped dolines follow the geology? Faults, lithology, orientation, clustering.

Data (EPSG:32632):
- dolines: Hanna's Aug 2025 point inventory (data/manual/sinkhole_points.gpkg)
- faults: LfU dGK25 structural lines, vectorised by fetch_dgk25.py
- lithology: dGK25 unit at each doline (WMS feature info, cached) and, for the
  area shares, at a regular grid of points (same query)
- domain: the Aug 2025 UAV footprint (where the inventory could find dolines)
- shapes: closed depressions of the Aug 2025 LiDAR (make_depressions.py)

Tests (all against a null of random points inside the domain, Monte Carlo):
1. distance to the nearest mapped fault (are dolines closer than random?)
2. lithology: share of dolines per unit vs area share of the unit
3. orientation: doline long axes and nearest-neighbour directions vs fault
   strikes (axial data); per doline the angle to the nearest fault's strike,
   uniform on 0-90 deg if unrelated
4. clustering: mean nearest-neighbour distance vs random (Clark-Evans idea,
   edge effects handled by simulating inside the same domain)

Caveats that matter for 1: the dGK25 lines come from the analogue 1:25,000
map, accurate to tens of metres, so differences at < ~25 m are below that.
And faults are only drawn where bedrock crops out, not under Quaternary
cover. So test 1 is also run for bedrock dolines against random points on
bedrock only (bedrock = red-hatched Wettersteinkalk in the rendered unit map,
checked against the queried units).

Outputs: outputs/figures/geology_vs_dolines.png, geology_vs_dolines.csv (per
doline), geology_vs_dolines_tests.csv
"""

import json
import sys
import time
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from PIL import Image
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).parent))
from fetch_dgk25 import BOUNDS, unit_at, wms  # noqa: E402
from prefill_doline_types import depression_at, elongation  # noqa: E402

from master_thesis.paths import FIGURES, MANUAL, PROCESSED, RAW  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CRS = "EPSG:32632"
N_SIM = 999
GRID_M = 40  # spacing of the lithology sample grid
UNIT_CACHE = RAW / "dgk25" / "units_cache.json"
RNG = np.random.default_rng(42)


def axial_mean(deg: np.ndarray, w=None) -> tuple[float, float]:
    """Mean direction (0-180) and resultant length of axial data."""
    a = np.radians(2 * np.asarray(deg))
    w = np.ones_like(a) if w is None else np.asarray(w)
    c, s = np.sum(w * np.cos(a)) / w.sum(), np.sum(w * np.sin(a)) / w.sum()
    return float(np.degrees(np.arctan2(s, c)) / 2 % 180), float(np.hypot(c, s))


def axial_diff(a, b) -> np.ndarray:
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


def random_points(domain, n: int, accept=None) -> np.ndarray:
    x0, y0, x1, y1 = domain.bounds
    out = []
    while len(out) < n:
        xy = RNG.uniform([x0, y0], [x1, y1], size=(n * 5, 2))
        inside = gpd.GeoSeries(gpd.points_from_xy(*xy.T)).within(domain).to_numpy()
        if accept is not None:
            inside &= accept(xy)
        out.extend(xy[inside])
    return np.asarray(out[:n])


def limestone_mask() -> tuple[np.ndarray, Affine]:
    """Wettersteinkalk (red hatching in the dGK25 unit rendering) at 1 m."""
    x0, y0, x1, y1 = BOUNDS
    w, h = int(x1 - x0), int(y1 - y0)
    r = wms({"REQUEST": "GetMap", "LAYERS": "geoleinheit_dgk25", "BBOX": f"{x0},{y0},{x1},{y1}",
             "WIDTH": w, "HEIGHT": h, "FORMAT": "image/png"})  # fmt: skip
    rgb = np.asarray(Image.open(BytesIO(r.content)).convert("RGB")).astype(int)
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 140) & (rgb[..., 2] < 140)
    frac = ndimage.uniform_filter(red.astype(float), size=7)
    return frac > 0.08, Affine(1, 0, x0, 0, -1, y1)


def cached_units(xy: np.ndarray) -> list[str | None]:
    cache = json.loads(UNIT_CACHE.read_text()) if UNIT_CACHE.exists() else {}
    units = []
    for x, y in xy:
        key = f"{x:.0f},{y:.0f}"
        if key not in cache:
            cache[key] = unit_at(float(x), float(y))
            time.sleep(1.0)  # be gentle with the LfU server
            UNIT_CACHE.write_text(json.dumps(cache))
        units.append(cache[key])
    return units


def nn_mean(xy: np.ndarray) -> float:
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def main() -> None:
    pts = gpd.read_file(MANUAL / "sinkhole_points.gpkg").to_crs(CRS)
    fp = gpd.read_file(PROCESSED / "uav_footprints.gpkg", layer="footprints")
    domain = fp[fp.flight.str.startswith("Aug 2025")].geometry.iloc[0]
    pts = pts[pts.within(domain)].reset_index(drop=True)
    xy = np.c_[pts.geometry.x, pts.geometry.y]
    seg = gpd.read_file(RAW / "dgk25" / "faults_segments.gpkg").to_crs(CRS)

    # distance to the nearest fault from the skeleton raster (all mapped lines)
    with rasterio.open(RAW / "dgk25" / "faults_skeleton.tif") as src:
        skel, tf = src.read(1), src.transform
    dist = ndimage.distance_transform_edt(skel == 0) * tf.a

    def fault_dist(p: np.ndarray) -> np.ndarray:
        rows, cols = rasterio.transform.rowcol(tf, p[:, 0], p[:, 1])
        return dist[np.asarray(rows), np.asarray(cols)]

    pts["fault_dist_m"] = fault_dist(xy).round(1)
    sims = np.array([np.median(fault_dist(random_points(domain, len(pts)))) for _ in range(N_SIM)])
    med = float(pts.fault_dist_m.median())
    p_dist = (np.sum(sims <= med) + 1) / (N_SIM + 1)

    # lithology
    pts["unit"] = cached_units(xy)
    lime, ltf = limestone_mask()

    def on_lime(p: np.ndarray) -> np.ndarray:
        rows, cols = rasterio.transform.rowcol(ltf, p[:, 0], p[:, 1])
        return lime[np.asarray(rows), np.asarray(cols)]

    pts["bedrock_map"] = on_lime(xy)
    is_lime = pts.unit.fillna("").str.contains("Wettersteinkalk")
    agree = float((pts.bedrock_map == is_lime).mean())
    bed = pts[is_lime]
    sims_b = np.array([np.median(fault_dist(random_points(domain, len(bed), on_lime)))
                       for _ in range(N_SIM)])  # fmt: skip
    med_b = float(bed.fault_dist_m.median())
    p_dist_b = (np.sum(sims_b <= med_b) + 1) / (N_SIM + 1)
    lime_share = float(on_lime(random_points(domain, 5000)).mean())
    gx, gy = np.meshgrid(np.arange(domain.bounds[0], domain.bounds[2], GRID_M),
                         np.arange(domain.bounds[1], domain.bounds[3], GRID_M))  # fmt: skip
    grid = np.c_[gx.ravel(), gy.ravel()]
    grid = grid[gpd.GeoSeries(gpd.points_from_xy(*grid.T)).within(domain).to_numpy()]
    grid_units = pd.Series(cached_units(grid)).fillna("none")
    area_share = grid_units.value_counts(normalize=True)
    dol_counts = pts.unit.fillna("none").value_counts()
    lith = pd.DataFrame({"dolines": dol_counts, "area_share": area_share}).fillna(0)
    lith["expected"] = lith.area_share * len(pts)
    lith_p = {u: stats.binomtest(int(r.dolines), len(pts), r.area_share).pvalue
              for u, r in lith.iterrows() if 0 < r.area_share < 1}  # fmt: skip

    # orientation: doline long axes from the LiDAR depression, nearest fault strike
    axes = []
    dep = gpd.read_file(PROCESSED / "depressions" / "depressions.gpkg", layer="20250820")
    for p in pts.geometry:
        found = depression_at(p, dep)
        if found is None:
            axes.append(np.nan)
            continue
        outline, _, diam, _ = found
        if diam < 2 or elongation(outline) < 1.3:
            axes.append(np.nan)
            continue
        rect = np.asarray(outline.minimum_rotated_rectangle.exterior.coords)
        sides = np.diff(rect[:3], axis=0)
        long = sides[np.argmax(np.hypot(*sides.T))]
        axes.append(float(np.degrees(np.arctan2(long[0], long[1])) % 180))
    pts["axis_deg"] = np.round(axes, 1)
    near = gpd.sjoin_nearest(pts[["geometry"]], seg[["strike_deg", "geometry"]])
    near = near[~near.index.duplicated()]
    pts["nearest_fault_strike"] = near.strike_deg
    has_axis = pts.axis_deg.notna()
    ang = axial_diff(pts.loc[has_axis, "axis_deg"], pts.loc[has_axis, "nearest_fault_strike"])
    pts.loc[has_axis, "axis_vs_fault_deg"] = ang.round(1)
    p_ang = stats.wilcoxon(ang - 45, alternative="less").pvalue if len(ang) > 5 else np.nan

    # nearest-neighbour directions (alignment of dolines)
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    j = d.argmin(axis=1)
    nn_dir = np.degrees(np.arctan2(xy[j, 0] - xy[:, 0], xy[j, 1] - xy[:, 1])) % 180

    # clustering
    nn_obs = nn_mean(xy)
    nn_sim = np.array([nn_mean(random_points(domain, len(pts))) for _ in range(199)])
    p_nn = (np.sum(nn_sim <= nn_obs) + 1) / 200

    f_mean, f_r = axial_mean(seg.strike_deg, seg.length_m)
    a_mean, a_r = axial_mean(pts.axis_deg.dropna()) if has_axis.any() else (np.nan, np.nan)
    n_mean, n_r = axial_mean(nn_dir)
    tests = pd.DataFrame([
        ["dolines", len(pts), ""],
        ["median distance to fault [m]", round(med, 1), f"random: {np.median(sims):.1f} "
         f"(5-95 %: {np.percentile(sims, 5):.1f}-{np.percentile(sims, 95):.1f}), p = {p_dist:.3f}"],
        ["bedrock dolines only: median distance [m]", round(med_b, 1),
         f"n = {len(bed)}; random on bedrock: {np.median(sims_b):.1f} (5-95 %: "
         f"{np.percentile(sims_b, 5):.1f}-{np.percentile(sims_b, 95):.1f}), p = {p_dist_b:.3f}"],
        ["unit map vs queried units, agreement", round(agree, 2),
         f"limestone share of domain (map) {lime_share:.0%}"],
        ["share within 25 m of a fault", round(float((pts.fault_dist_m <= 25).mean()), 2), ""],
        ["fault strike mean / R (length-weighted)", round(f_mean), f"R = {f_r:.2f}"],
        ["doline long axis mean / R", round(a_mean) if has_axis.any() else "",
         f"R = {a_r:.2f}, n = {int(has_axis.sum())}"],
        ["nearest-neighbour direction mean / R", round(n_mean), f"R = {n_r:.2f}"],
        ["axis vs nearest fault strike, median [deg]",
         round(float(np.median(ang)), 1) if len(ang) else "",
         f"45 if unrelated; Wilcoxon p = {p_ang:.3f}"],
        ["mean NN distance [m]", round(nn_obs, 1), f"random: {nn_sim.mean():.1f}, p = {p_nn:.3f}"],
    ], columns=["test", "value", "detail"])  # fmt: skip
    for u, pv in lith_p.items():
        r = lith.loc[u]
        tests.loc[len(tests)] = [f"lithology: {u}", int(r.dolines),
                                 f"expected {r.expected:.1f} ({r.area_share:.0%} of area), "
                                 f"binomial p = {pv:.3f}"]  # fmt: skip

    FIGURES.mkdir(parents=True, exist_ok=True)
    pts.drop(columns="geometry").assign(x=xy[:, 0].round(1), y=xy[:, 1].round(1)).to_csv(
        FIGURES / "geology_vs_dolines.csv", index=False)  # fmt: skip
    tests.to_csv(FIGURES / "geology_vs_dolines_tests.csv", index=False)
    print(tests.to_string(index=False))
    print(lith.round(3).to_string())
    print(pts.groupby(pts["type"].fillna("(empty)")).fault_dist_m.agg(["count", "median"]))

    # figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 3, (1, 3))
    x0, y0, x1, y1 = domain.bounds
    ext = (tf.c, tf.c + tf.a * skel.shape[1], tf.f + tf.e * skel.shape[0], tf.f)
    lext = (ltf.c, ltf.c + lime.shape[1], ltf.f - lime.shape[0], ltf.f)
    ax.imshow(np.ma.masked_equal(lime, 0), cmap="Reds", vmin=0, vmax=4, extent=lext, alpha=0.5)
    ax.imshow(np.ma.masked_equal(ndimage.binary_dilation(skel, iterations=2), 0),
              cmap="Greys_r", extent=ext, interpolation="nearest")  # fmt: skip
    gpd.GeoSeries([domain]).boundary.plot(ax=ax, color="grey", lw=0.8)
    for t, g in pts.groupby(pts["type"].fillna("(no type)")):
        ax.scatter(g.geometry.x, g.geometry.y, s=14, label=f"{t} ({len(g)})")
    ax.set_xlim(x0 - 50, x1 + 50)
    ax.set_ylim(y0 - 50, y1 + 50)  # noqa: E702
    ax.legend(fontsize=7, loc="lower left")
    ax.set_aspect("equal")  # noqa: E702
    ax.set_title("Dolines (Aug 2025 inventory), dGK25 faults (black), Wettersteinkalk (red)")
    ax = fig.add_subplot(2, 3, 4)
    ax.hist(sims, bins=30, color="lightgrey", label="random, whole area")
    ax.axvline(med, color="red", label="all dolines")
    ax.hist(sims_b, bins=30, color="tab:green", alpha=0.4, label="random, bedrock only")
    ax.axvline(med_b, color="darkgreen", ls="--", label="bedrock dolines")
    ax.set_xlabel("median distance to nearest fault [m]")
    ax.legend()  # noqa: E702
    for k, (title, data, w) in enumerate([
        ("fault strike (length-weighted)", seg.strike_deg, seg.length_m),
        ("doline long axes / NN directions", pts.axis_deg.dropna(), None),
    ]):  # fmt: skip
        ax = fig.add_subplot(2, 3, 5 + k, projection="polar")
        bins = np.radians(np.arange(0, 361, 10))
        vals = np.r_[data, (np.asarray(data) + 180) % 360]
        ww = None if w is None else np.r_[w, w]
        ax.hist(np.radians(vals), bins=bins, weights=ww, color="tab:blue", alpha=0.7)
        if k == 1:
            ax.hist(np.radians(np.r_[nn_dir, (nn_dir + 180) % 360]), bins=bins,
                    histtype="step", color="tab:orange", lw=1.5)  # fmt: skip
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # noqa: E702
        ax.set_title(title, fontsize=9)
        ax.set_yticklabels([])  # noqa: E702
    fig.tight_layout()
    fig.savefig(FIGURES / "geology_vs_dolines.png", dpi=110)
    print(f"wrote {FIGURES / 'geology_vs_dolines.png'}")


if __name__ == "__main__":
    main()
