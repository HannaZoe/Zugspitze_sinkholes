"""Keep the doline inventory consistent: point ids 1..n, polygons linked to points.

Runs in QGIS's Python (GeoPackage edits need GDAL's SQL functions):

    "C:\\Program Files\\QGIS 3.44.8\\bin\\python-qgis-ltr.bat" scripts\\link_doline_ids.py

Close QGIS first. Steps:
1. Points (data/manual/sinkhole_points.gpkg): ids renumbered 1..n without gaps,
   in the existing order; empty epoch/source filled with the Aug 2025 defaults;
   `year` (int) from the epoch.
2. Polygons (data/manual/sinkhole_polygons.gpkg): each gets the id of the one
   point inside it (no point inside: nearest point within NEAREST_M, if that
   point has no polygon of its own in the same epoch);
   polygons holding several points keep an empty id and are listed.
   Epoch spellings normalised, `year` set, `type` taken from the point,
   `source` from the point (or DOP20 / Mavic 3M by epoch).
Commit the gpkgs before running, so the previous state stays recoverable.
"""

import os
from pathlib import Path

from osgeo import ogr

ogr.UseExceptions()

ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2] / "Master_Thesis"))
POINTS = ROOT / "data" / "manual" / "sinkhole_points.gpkg"
POLYGONS = ROOT / "data" / "manual" / "sinkhole_polygons.gpkg"
NEAREST_M = 3.0
DEFAULT_EPOCH, DEFAULT_SOURCE = "2025-08-20", "UAV RGB + LiDAR hillshade"
TYPES = {"sinkhole - funnel", "sinkhole - bowl", "sinkhole - elongated", "cave",
         "multi-sinkhole", "unclear"}  # fmt: skip
EPOCH = {None: DEFAULT_EPOCH, "": DEFAULT_EPOCH, "08.2025": "2025-08-20",
         "08.2026": "2026-08-27", "2003": "2003-09-04"}  # fmt: skip
SOURCE_BY_EPOCH = {"2003-09-04": "DOP20", "2026-08-27": "UAV RGB (Mavic 3M)"}


def ensure_int_field(layer, name: str) -> None:
    if layer.GetLayerDefn().GetFieldIndex(name) < 0:
        layer.CreateField(ogr.FieldDefn(name, ogr.OFTInteger))


def renumber_points(layer) -> dict[int, tuple]:
    ensure_int_field(layer, "year")
    feats = sorted(layer, key=lambda f: (f.GetField("id") is None, f.GetField("id") or 0))
    points = {}
    for new_id, f in enumerate(feats, start=1):
        f.SetField("id", new_id)
        f.SetField("epoch", f.GetField("epoch") or DEFAULT_EPOCH)
        f.SetField("source", f.GetField("source") or DEFAULT_SOURCE)
        f.SetField("year", int(f.GetField("epoch")[:4]))
        layer.SetFeature(f)
        points[new_id] = (f.GetGeometryRef().Clone(), f.GetField("type"), f.GetField("source"))
    return points


def link_polygons(layer, points: dict[int, tuple]) -> None:
    ensure_int_field(layer, "year")
    feats = list(layer)
    epoch_of = {f.GetFID(): EPOCH.get(f.GetField("epoch"), f.GetField("epoch")) for f in feats}
    inside = {f.GetFID(): [i for i, (p, _, _) in points.items()
                           if f.GetGeometryRef().Contains(p)] for f in feats}  # fmt: skip
    taken = {(epoch_of[k], v[0]) for k, v in inside.items() if len(v) == 1}
    several, nearest, orphans = [], [], []
    for f in feats:
        g, fid = f.GetGeometryRef(), f.GetFID()
        hits = inside[fid]
        pid = hits[0] if len(hits) == 1 else None
        if not hits:
            dist, i = min((g.Distance(p), i) for i, (p, _, _) in points.items())
            if dist <= NEAREST_M and (epoch_of[fid], i) not in taken:
                pid = i
                nearest.append(i)
                taken.add((epoch_of[fid], i))
            else:
                orphans.append(fid)
        elif len(hits) > 1:
            several.append((fid, hits))
        epoch = epoch_of[fid]
        f.SetField("epoch", epoch)
        f.SetField("year", int(epoch[:4]))
        if pid is None:
            f.SetFieldNull("id")
        else:
            f.SetField("id", pid)
            _, ptype, psource = points[pid]
            if ptype:
                f.SetField("type", ptype)
            elif f.GetField("type") not in TYPES:
                f.SetFieldNull("type")
            f.SetField("source", SOURCE_BY_EPOCH.get(epoch, psource or DEFAULT_SOURCE))
        layer.SetFeature(f)
    print(f"polygons linked via nearest point (<= {NEAREST_M} m): {nearest}")
    print(f"polygons with several points (no id, check by hand): {several}")
    print(f"polygons without a point (no id, check by hand), fid: {orphans}")


def main() -> None:
    pds = ogr.Open(str(POINTS), 1)
    points = renumber_points(pds.GetLayer("sinkhole_points"))
    gds = ogr.Open(str(POLYGONS), 1)
    link_polygons(gds.GetLayer("sinkholes"), points)
    r = gds.ExecuteSQL("SELECT year, count(*), sum(id IS NULL) FROM sinkholes GROUP BY year")
    for x in r:
        print(f"polygons {x.GetField(0)}: {x.GetField(1)}, without id {x.GetField(2)}")
    gds.ReleaseResultSet(r)
    r = gds.ExecuteSQL("SELECT id, year, count(*) c FROM sinkholes WHERE id IS NOT NULL "
                       "GROUP BY id, year HAVING c > 1")  # fmt: skip
    print("same id twice in one year:", [(x.GetField(0), x.GetField(1)) for x in r])
    gds.ReleaseResultSet(r)
    print(f"points: {len(points)}, ids 1-{len(points)}")


if __name__ == "__main__":
    main()
