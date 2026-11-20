"""Review list for the doline point inventory: where the mapped points and the
other sources disagree.

Layers in data/processed/mapping_review.gpkg (EPSG:32632):
- deep_unmarked: closed depressions (0.1 m outline) more than 1 m deep in the
  Aug 2025 LiDAR with no mapped point within 3 m (missed doline, or just a
  gap between boulders?)
- points_no_depression: mapped points more than 5 m from any closed depression
  (shaft the LiDAR didn't see into, flat feature, or misplaced point?)
- older_unmatched: Sinkholes.shp and field doline points with no mapped point
  within 10 m
"""

import geopandas as gpd
import pandas as pd

from master_thesis.paths import FIELD_DATA, MANUAL, PROCESSED

CRS = "EPSG:32632"
OUT = PROCESSED / "mapping_review.gpkg"


def main() -> None:
    pts = gpd.read_file(MANUAL / "sinkhole_points.gpkg").to_crs(CRS)
    dep = gpd.read_file(PROCESSED / "depressions" / "depressions.gpkg", layer="20250820")
    rim = dep[dep.level_m == 0.1]
    mapped = pts.union_all()

    deep = rim[(rim.max_depth_m > 1) & ~rim.buffer(3).intersects(mapped)]
    far = pts[pts.distance(rim.union_all()) > 5]
    shp = gpd.read_file(MANUAL / "Sinkholes.shp").to_crs(CRS).assign(origin="Sinkholes.shp")
    field = gpd.read_file(FIELD_DATA / "fieldwork_annotations.gpkg").to_crs(CRS)
    field = field[field["type"].astype(str).str.contains("olin|ave", case=False)]
    field = field.assign(origin="field point " + field["type"].astype(str))
    older = pd.concat([shp[["origin", "geometry"]], field[["origin", "geometry"]]])
    older = gpd.GeoDataFrame(older, crs=CRS)
    older = older[~older.buffer(10).intersects(mapped)]

    for name, gdf in [("deep_unmarked", deep), ("points_no_depression", far),
                      ("older_unmatched", older)]:  # fmt: skip
        gdf.to_file(OUT, layer=name, driver="GPKG")
        print(f"{name}: {len(gdf)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
