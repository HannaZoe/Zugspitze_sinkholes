"""Vector layers of the LfU dGK25 for mapping: geological units and faults.

The LfU serves the dGK25 only as a WMS (no WFS, feature info without
geometry; the exact vectors exist only as a Bavaria-wide UmweltAtlas
download). So the layers are derived from the rendered map:

- Units: the unit layer is rendered at 1 m over the study area plus
  ~300 m; the output is clipped to the UAV flights + 150 m. Hatching and dot
  patterns are turned into texture features by smoothing the colours at two
  scales (7 and 15 m), clustered (k-means), and each cluster is assigned the
  unit most of its sample points have. Sample
  points = units queried by feature info (cached, units_cache.json) on a
  regular grid plus earlier queries. The unit map is cleaned with a 15 m
  majority filter and patches < MIN_AREA_M2, then polygonised and dissolved.
  The doline points (queried separately) are held out as a check.
  (Naming every polygon by a query is not feasible: ~14,000 raw patches.)
- Faults: the skeleton from fetch_dgk25.py is turned into continuous lines
  (8-connected pixel chains merged, simplified by 1 m), each tagged with the
  line type of the nearest Hough segment (proven / inferred fault).

Both are good to the map's own accuracy (tens of metres), not better.
Output: data/raw/dgk25/dgk25_vector.gpkg, layers `units` and `faults`
(EPSG:32632).
"""

import json
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import shapes, sieve
from scipy import ndimage
from shapely.geometry import LineString, shape
from shapely.ops import linemerge, unary_union
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from fetch_dgk25 import BOUNDS, feature_info, wms  # noqa: E402
from geology_vs_dolines import cached_units  # noqa: E402

from master_thesis.paths import MANUAL, PROCESSED, RAW  # noqa: E402

CRS = "EPSG:32632"
OUT = RAW / "dgk25" / "dgk25_vector.gpkg"
N_CLUSTERS = 24
SMOOTH_PX = (7, 15)
GRID_M = 60
MARGIN_M = 150  # output: study area plus this margin
MAJORITY_PX = 15
MIN_AREA_M2 = 400


def render_units() -> tuple[np.ndarray, Affine]:
    x0, y0, x1, y1 = BOUNDS
    w, h = int(x1 - x0), int(y1 - y0)
    r = wms({"REQUEST": "GetMap", "LAYERS": "geoleinheit_dgk25", "BBOX": f"{x0},{y0},{x1},{y1}",
             "WIDTH": w, "HEIGHT": h, "FORMAT": "image/png"})  # fmt: skip
    rgb = np.asarray(Image.open(BytesIO(r.content)).convert("RGB")).astype("f4")
    return rgb, Affine(1, 0, x0, 0, -1, y1)


def area_of_interest():
    """All UAV flights plus MARGIN_M (output extent) and the Aug 2025 flight."""
    fp = gpd.read_file(PROCESSED / "uav_footprints.gpkg", layer="footprints").to_crs(CRS)
    ref = fp[fp.flight.str.startswith("Aug 2025")].geometry.iloc[0]
    return unary_union(fp.geometry).buffer(MARGIN_M), ref


def unit_polygons() -> gpd.GeoDataFrame:
    rgb, tf = render_units()
    feats = np.concatenate([np.stack([ndimage.uniform_filter(rgb[..., b], k) for b in range(3)], -1)
                            for k in SMOOTH_PX], axis=-1)  # fmt: skip
    flat = feats.reshape(-1, feats.shape[-1])
    km = KMeans(N_CLUSTERS, n_init=4, random_state=0).fit(flat[::37])
    clusters = km.predict(flat).reshape(rgb.shape[:2])

    # training: a regular grid in the margin around the flights (inside them the
    # earlier 40 m grid and the doline queries are already cached)
    aoi, ref = area_of_interest()
    x0, y0, x1, y1 = aoi.bounds
    gx, gy = np.meshgrid(
        np.arange(x0 + GRID_M / 2, x1, GRID_M), np.arange(y0 + GRID_M / 2, y1, GRID_M)
    )
    grid = np.c_[gx.ravel(), gy.ravel()]
    pts = gpd.GeoSeries(gpd.points_from_xy(*grid.T), crs=CRS)
    cached_units(grid[(pts.within(aoi) & ~pts.within(ref)).to_numpy()])
    cache = json.loads((RAW / "dgk25" / "units_cache.json").read_text())
    dol = gpd.read_file(MANUAL / "sinkhole_points.gpkg").to_crs(CRS)
    dol_xy = list(zip(dol.geometry.x, dol.geometry.y, strict=True))
    dol_keys = {f"{x:.0f},{y:.0f}" for x, y in dol_xy}
    train = [(float(k.split(",")[0]), float(k.split(",")[1]), u) for k, u in cache.items()
             if u and k not in dol_keys]  # fmt: skip

    def pix(x, y):
        r, c = rasterio.transform.rowcol(tf, x, y)
        return (r, c) if 0 <= r < clusters.shape[0] and 0 <= c < clusters.shape[1] else None

    votes: dict[int, Counter] = {}
    for x, y, u in train:
        rc = pix(x, y)
        if rc:
            votes.setdefault(int(clusters[rc]), Counter())[u] += 1
    names = sorted({u for c in votes.values() for u in c})
    lut = np.full(N_CLUSTERS, -1)
    for k, c in votes.items():
        lut[k] = names.index(c.most_common(1)[0][0])
    unit_px = lut[clusters]
    # 15 m majority filter (unknown clusters take their neighbourhood's majority)
    share = np.stack([ndimage.uniform_filter((unit_px == i).astype("f4"), MAJORITY_PX)
                      for i in range(len(names))])  # fmt: skip
    unit_px = (share.argmax(0) + 1).astype("int32")
    unit_px = sieve(unit_px, size=MIN_AREA_M2, connectivity=8)

    # held-out check at the doline points
    hits = [(unit_px[pix(x, y)] - 1, cache.get(f"{x:.0f},{y:.0f}")) for x, y in dol_xy
            if pix(x, y)]  # fmt: skip
    ok = [names[a] == b for a, b in hits if b]
    print(f"  {len(train)} training points, {len(names)} units; held-out doline points: "
          f"{np.mean(ok):.0%} correct (n = {len(ok)})")  # fmt: skip

    polys = [(shape(g), names[int(v) - 1]) for g, v in shapes(unit_px, transform=tf)]
    gdf = gpd.GeoDataFrame({"unit": [u for _, u in polys]}, geometry=[g for g, _ in polys], crs=CRS)
    out = gdf.dissolve(by="unit", as_index=False)
    out["geometry"] = out.geometry.buffer(0).intersection(aoi).simplify(1.0)
    codes = {}
    for u in names:  # one query per unit for its short code
        x, y, _ = next(t for t in train if t[2] == u)
        props = feature_info("geoleinheit_dgk25", x, y, tol_px=0)
        codes[u] = props[0].get("Kurzname der Geologischen Einheit") if props else None
        time.sleep(0.5)
    out["code"] = out.unit.map(codes)
    out["area_ha"] = (out.area / 1e4).round(2)
    return out[["unit", "code", "area_ha", "geometry"]]


def fault_lines() -> gpd.GeoDataFrame:
    with rasterio.open(RAW / "dgk25" / "faults_skeleton.tif") as src:
        skel, tf = src.read(1).astype(bool), src.transform
    rows, cols = np.nonzero(skel)
    segs = []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # each neighbour pair once
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (r2 < skel.shape[0]) & (c2 >= 0) & (c2 < skel.shape[1])
        ok[ok] = skel[r2[ok], c2[ok]]
        for a, b, c, d in zip(rows[ok], cols[ok], r2[ok], c2[ok], strict=True):
            segs.append(LineString([tf * (b + 0.5, a + 0.5), tf * (d + 0.5, c + 0.5)]))
    merged = linemerge(unary_union(segs))
    lines = [g.simplify(1.0) for g in getattr(merged, "geoms", [merged]) if g.length > 15]
    gdf = gpd.GeoDataFrame(geometry=lines, crs=CRS)
    hough = gpd.read_file(RAW / "dgk25" / "faults_segments.gpkg")
    near = gpd.sjoin_nearest(gdf, hough[["line_type", "geometry"]], max_distance=5)
    near = near[~near.index.duplicated()]
    gdf["line_type"] = near.line_type.reindex(gdf.index)
    gdf["type"] = np.where(gdf.line_type.fillna("").str.contains("vermutet")
                           & ~gdf.line_type.fillna("").str.contains("nachgewiesen"),
                           "fault, inferred", "fault, proven")  # fmt: skip
    gdf["length_m"] = gdf.length.round(0)
    print(f"  {len(gdf)} fault lines, {gdf.length.sum() / 1000:.1f} km")
    return gdf[["type", "line_type", "length_m", "geometry"]]


def main() -> None:
    units = unit_polygons()
    units.to_file(OUT, layer="units", driver="GPKG")
    print(units.drop(columns="geometry").to_string(index=False))
    fault_lines().to_file(OUT, layer="faults", driver="GPKG")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
