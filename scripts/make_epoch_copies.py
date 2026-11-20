"""Copy the Aug 2025 reference outlines into the other UAV epochs for copy-and-adjust mapping.

Runs in QGIS's Python (GeoPackage edits need GDAL's SQL functions); close QGIS:

    "C:\\Program Files\\QGIS 3.44.8\\bin\\python-qgis-ltr.bat" scripts\\make_epoch_copies.py

Every Aug 2025 polygon in data/manual/sinkhole_polygons.gpkg (layer
`sinkholes`) is copied once per epoch in EPOCHS, with id, type, notes and
genesis kept and epoch/year/source set. Two fields are added for the
multi-epoch mapping (docs/mapping_protocol.md):
- `snow`: none / partial / filled (empty on the copies, Hanna fills it)
- `changed`: no on the copies until Hanna moves vertices (then yes);
  n.a. on the reference epoch
Ids that already have a polygon in an epoch are skipped, so running it twice
or after hand-drawn test polygons does not duplicate anything.
"""

import os
from pathlib import Path

from osgeo import ogr

ogr.UseExceptions()

ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2] / "Master_Thesis"))
POLYGONS = ROOT / "data" / "manual" / "sinkhole_polygons.gpkg"
REFERENCE = "2025-08-20"
EPOCHS = {  # epoch -> source
    "2024-10-25": "UAV RGB + LiDAR hillshade",
    "2025-06-04": "UAV RGB + LiDAR hillshade",
    "2026-08-27": "UAV RGB (Mavic 3M)",
}


def main() -> None:
    ds = ogr.Open(str(POLYGONS), 1)
    layer = ds.GetLayer("sinkholes")
    for name in ("snow", "changed"):
        if layer.GetLayerDefn().GetFieldIndex(name) < 0:
            layer.CreateField(ogr.FieldDefn(name, ogr.OFTString))

    feats = list(layer)
    ref = [f for f in feats if f.GetField("epoch") == REFERENCE]
    have = {(f.GetField("epoch"), f.GetField("id")) for f in feats if f.GetField("id") is not None}
    layer.StartTransaction()
    for f in ref:
        f.SetField("snow", f.GetField("snow") or "none")
        f.SetField("changed", "n.a.")
        layer.SetFeature(f)
    for epoch, source in EPOCHS.items():
        n = 0
        for f in ref:
            if (epoch, f.GetField("id")) in have:
                continue
            new = ogr.Feature(layer.GetLayerDefn())
            new.SetFrom(f)
            new.SetFID(ogr.NullFID)
            new.SetField("epoch", epoch)
            new.SetField("year", int(epoch[:4]))
            new.SetField("source", source)
            new.SetFieldNull("snow")
            new.SetField("changed", "no")
            layer.CreateFeature(new)
            n += 1
        print(f"{epoch}: {n} copies")
    layer.CommitTransaction()
    r = ds.ExecuteSQL("SELECT epoch, count(*), sum(id IS NULL) FROM sinkholes GROUP BY epoch")
    for x in r:
        print(f"  {x.GetField(0)}: {x.GetField(1)} polygons, {x.GetField(2)} without id")
    ds.ReleaseResultSet(r)


if __name__ == "__main__":
    main()
