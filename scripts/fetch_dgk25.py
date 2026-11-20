"""Faults and geological units of the LfU dGK25 around the study area, as data.

The LfU serves the digital geological map 1:25,000 (CC BY 4.0, "Datenquelle:
Bayerisches Landesamt für Umwelt") only as a WMS: no WFS, and feature info
comes without geometry. So:
- Structural lines (`strukturln_dgk25`) are rendered at RES_M per pixel,
  the line pixels are thinned to a one-pixel skeleton and stored as a raster.
  Line segments come from a probabilistic Hough transform on the skeleton,
  each tagged with its line type by a GetFeatureInfo query at its midpoint.
- Units (`geoleinheit_dgk25`) are queried per point (`unit_at`).

The map is digitised from the analogue GK25, so its own positional accuracy is
tens of metres. Rendering at 0.5 m adds nothing to that.

Outputs in data/raw/dgk25/ (EPSG:25832 grid, which equals EPSG:32632 to < 1 m
here): faults_skeleton.tif (1 = line), faults_segments.gpkg (segments with
`line_type`, `strike_deg` 0-180).
"""

import json
import time
from io import BytesIO

import geopandas as gpd
import numpy as np
import rasterio
import requests
from affine import Affine
from PIL import Image
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

from master_thesis.paths import RAW

URL = "https://www.lfu.bayern.de/gdi/wms/geologie/dgk25"
CRS = "EPSG:32632"  # requested as EPSG:25832; same numbers to < 1 m here
BOUNDS = (649200.0, 5252160.0, 652250.0, 5254100.0)  # study area + ~300 m
RES_M = 0.5
TILE = 4000  # px, WMS max is 4096
OUT = RAW / "dgk25"


def wms(params: dict, tries: int = 5) -> requests.Response:
    """WMS request with retries: the LfU server drops connections under many queries."""
    base = {"SERVICE": "WMS", "VERSION": "1.3.0", "CRS": "EPSG:25832", "STYLES": ""}
    for k in range(tries):
        try:
            r = requests.get(URL, params=base | params, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(5 * 2**k)


def render_lines() -> tuple[np.ndarray, Affine]:
    x0, y0, x1, y1 = BOUNDS
    w, h = round((x1 - x0) / RES_M), round((y1 - y0) / RES_M)
    mask = np.zeros((h, w), bool)
    for ty in range(0, h, TILE):
        for tx in range(0, w, TILE):
            tw, th = min(TILE, w - tx), min(TILE, h - ty)
            bx0, by1 = x0 + tx * RES_M, y1 - ty * RES_M
            bbox = f"{bx0},{by1 - th * RES_M},{bx0 + tw * RES_M},{by1}"
            r = wms({"REQUEST": "GetMap", "LAYERS": "strukturln_dgk25", "BBOX": bbox,
                     "WIDTH": tw, "HEIGHT": th, "FORMAT": "image/png",
                     "TRANSPARENT": "TRUE"})  # fmt: skip
            rgba = np.asarray(Image.open(BytesIO(r.content)).convert("RGBA"))
            dark = (rgba[..., 3] > 100) & (rgba[..., :3].max(axis=2) < 110)
            mask[ty : ty + th, tx : tx + tw] = dark
    return mask, Affine(RES_M, 0, x0, 0, -RES_M, y1)


def feature_info(layer: str, x: float, y: float, tol_px: int = 4) -> list[dict]:
    """Attributes of `layer` at (x, y); tolerance in pixels of a 1 m/px query image."""
    half = 50
    bbox = f"{x - half},{y - half},{x + half},{y + half}"
    r = wms({"REQUEST": "GetFeatureInfo", "LAYERS": layer, "QUERY_LAYERS": layer,
             "BBOX": bbox, "WIDTH": 101, "HEIGHT": 101, "I": 50, "J": 50,
             "INFO_FORMAT": "application/geo+json", "FEATURE_COUNT": 5,
             "TOLERANCE": tol_px})  # fmt: skip
    try:
        return [f["properties"] for f in json.loads(r.text).get("features", [])]
    except json.JSONDecodeError:
        return []


def unit_at(x: float, y: float) -> str | None:
    props = feature_info("geoleinheit_dgk25", x, y, tol_px=0)
    return props[0].get("Geologische Einheit") if props else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mask, tf = render_lines()
    closed = ndimage.binary_closing(mask, iterations=3)  # bridge dashed (inferred) lines
    skel = skeletonize(closed)
    profile = {"driver": "GTiff", "height": skel.shape[0], "width": skel.shape[1],
               "count": 1, "dtype": "uint8", "crs": CRS, "transform": tf,
               "compress": "deflate", "nodata": 0}  # fmt: skip
    with rasterio.open(OUT / "faults_skeleton.tif", "w", **profile) as dst:
        dst.write(skel.astype("uint8"), 1)

    segs = probabilistic_hough_line(skel, threshold=10, line_length=40, line_gap=8, rng=0)
    rows = []
    for (c0, r0), (c1, r1) in segs:
        (xa, ya), (xb, yb) = tf * (c0, r0), tf * (c1, r1)
        strike = np.degrees(np.arctan2(xb - xa, yb - ya)) % 180  # azimuth, axial
        props = feature_info("strukturln_dgk25", (xa + xb) / 2, (ya + yb) / 2)
        time.sleep(1.0)  # be gentle with the LfU server
        line_type = "; ".join(sorted({str(v) for p in props for k, v in p.items()
                                      if k.lower() not in ("shape", "objectid")}))  # fmt: skip
        rows.append({"strike_deg": round(float(strike), 1), "line_type": line_type or None,
                     "geometry": LineString([(xa, ya), (xb, yb)])})  # fmt: skip
    gdf = gpd.GeoDataFrame(rows, crs=CRS)
    gdf["length_m"] = gdf.length.round(1)
    gdf.to_file(OUT / "faults_segments.gpkg", layer="segments", driver="GPKG")
    print(f"line pixels {mask.sum()}, skeleton {skel.sum()}, segments {len(gdf)}")
    print(gdf.line_type.value_counts(dropna=False).head(10).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
