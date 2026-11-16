"""Outlines of the UAV flights we use, and which one is the largest.

The valid-data mask of each co-registered UAV raster is polygonised at 1 m
(5x5 blocks of the 20 cm grid), small holes (< 100 m2, nodata specks) are
filled and the outline simplified by 1 m. EPSG:32632.
Output: data/processed/uav_footprints.gpkg with two layers:
- footprints: one outline per flight, with area_ha
- largest: the largest of them, as the study-area outline for mapping
"""

import geopandas as gpd
import rasterio
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from master_thesis.paths import INTERIM, PROCESSED

CRS = "EPSG:32632"
IN_DIR = INTERIM / "coregistered_20cm"
OUT = PROCESSED / "uav_footprints.gpkg"
FACTOR = 5
MIN_HOLE_M2 = 100
FLIGHTS = {
    "Oct 2024 (Terra LiDAR/RGB)": "20241025_terra_ppk_lidar_dsm",
    "Jun 2025 (Terra LiDAR/RGB)": "20250604_terra_ppk_lidar_dsm",
    "Aug 2025 (Terra LiDAR/RGB, reference)": "20250820_terra_ppk_lidar_dsm",
    "Aug 2026 (Mavic 3M)": "20260827_terra_rtk_rgb_om_20cm_local_coregistration",
    "Aug 2026 (Mavic 3T)": "20260827_terra_rtk_thermal_rgb_om_20cm_local_coregistration",
}


def fill_holes(poly: Polygon) -> Polygon:
    return Polygon(poly.exterior, [r for r in poly.interiors if Polygon(r).area >= MIN_HOLE_M2])


def outline(name: str):
    with rasterio.open(IN_DIR / f"{name}.tif") as src:
        shape_1m = (src.height // FACTOR, src.width // FACTOR)
        valid = (src.dataset_mask(out_shape=shape_1m) > 0).astype("uint8")
        tf = src.transform * Affine.scale(src.width / shape_1m[1], src.height / shape_1m[0])
    polys = [shape(g) for g, v in shapes(valid, mask=valid.astype(bool), transform=tf) if v]
    merged = unary_union(polys)
    parts = getattr(merged, "geoms", [merged])
    largest_part = max(parts, key=lambda p: p.area)  # drops stray specks off the flight
    return fill_holes(largest_part).simplify(1.0)


def main() -> None:
    rows = [
        {"flight": label, "source": name, "geometry": outline(name)}
        for label, name in FLIGHTS.items()
    ]
    gdf = gpd.GeoDataFrame(rows, crs=CRS)
    gdf["area_ha"] = (gdf.area / 1e4).round(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT, layer="footprints", driver="GPKG")
    gdf.loc[[gdf.area_ha.idxmax()]].to_file(OUT, layer="largest", driver="GPKG")
    print(
        gdf.drop(columns="geometry").sort_values("area_ha", ascending=False).to_string(index=False)
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
