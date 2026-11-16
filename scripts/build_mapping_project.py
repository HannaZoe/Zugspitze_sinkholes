"""Build the QGIS mapping project with every co-registered layer, ready to digitise.

Runs in QGIS's own Python, not the uv environment:

    "C:\\Program Files\\QGIS 3.44.8\\bin\\python-qgis-ltr.bat" scripts\\build_mapping_project.py

Paths follow master_thesis.paths (sibling Master_Thesis repo, or THESIS_ROOT),
but without importing the package, which isn't installed in QGIS's Python.
Everything is EPSG:32632 on the common 20 cm grid, i.e. already aligned to the
20250820 reference, so layers can be toggled against each other directly.
Writes Master_Thesis/qgis/projects/kartierung.qgz (relative paths) and
overwrites it if it exists; the older Zugspitze.qgz is left alone.
"""

import os
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)

ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2] / "Master_Thesis"))
CO = ROOT / "data" / "interim" / "coregistered_20cm"
OUT = ROOT / "qgis" / "projects" / "kartierung.qgz"

# (group, [(layer name, path, visible)]), top of the legend first
GROUPS = [
    ("Kartierung", [
        ("Dolinen-Polygone (zu digitalisieren)",
         f"{ROOT / 'data/manual/sinkhole_polygons.gpkg'}|layername=sinkholes", True),
        ("Feldpunkte Aug 2026",
         str(ROOT / "fieldwork/field_data/fieldwork_annotations.gpkg"), True),
        ("Dolinenpunkte (Sinkholes.shp)", str(ROOT / "data/manual/Sinkholes.shp"), True),
        ("Verdachtsflaechen", str(ROOT / "data/manual/Suspicious_Areas.shp"), False),
    ]),
    ("Schummerung (LiDAR)", [
        ("Aug 2025 LiDAR", CO / "hillshade/20250820_terra_ppk_lidar_dsm_hillshade.tif", True),
        ("Jun 2025 LiDAR", CO / "hillshade/20250604_terra_ppk_lidar_dsm_hillshade.tif", False),
        ("Okt 2024 LiDAR", CO / "hillshade/20241025_terra_ppk_lidar_dsm_hillshade.tif", False),
        ("DGM1", CO / "hillshade/bavaria_dgm1_hillshade.tif", False),
    ]),
    ("Drohnen-Orthos", [
        ("Aug 2025 RGB (Referenz)", CO / "20250820_terra_ppk_rgb_om.tif", True),
        ("Aug 2026 RGB (3M)",
         CO / "20260827_terra_rtk_rgb_om_20cm_local_coregistration.tif", False),
        ("Aug 2026 Thermal (3T)",
         CO / "20260827_terra_rtk_thermal_ir_om_20cm_local_coregistration.tif", False),
        ("Jun 2025 RGB (Schnee!)", CO / "20250604_terra_ppk_rgb_om.tif", False),
        ("Okt 2024 RGB", CO / "20241025_terra_ppk_rgb_om.tif", False),
    ]),
    ("Historische DOP20", [
        (f"DOP {d}", CO / f"bavaria_dop20_{d}.tif", False)
        for d in ["2024-07-30", "2022-06-12", "2020-07-08", "2018-09-11", "2015-07-01",
                  "2012-07-27", "2009-09-01", "2006-07-02", "2003-09-04"]
    ]),
    ("Hilfslayer", [
        ("LoD95 Jun -> Aug 2025 [m]",
         ROOT / "data/processed/uncertainty_1m/lod95_20250820_20250604.tif", False),
        ("LoD95 Okt 2024 -> Aug 2025 [m]",
         ROOT / "data/processed/uncertainty_1m/lod95_20250820_20241025.tif", False),
    ]),
]  # fmt: skip


def main() -> None:
    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:32632"))
    root = project.layerTreeRoot()
    for group_name, layers in GROUPS:
        group = root.addGroup(group_name)
        for name, path, visible in layers:
            path = str(path)
            vector = path.endswith((".gpkg", ".shp")) or "|layername=" in path
            layer = QgsVectorLayer(path, name, "ogr") if vector else QgsRasterLayer(path, name)
            if not layer.isValid():
                print(f"  skipped (not found / invalid): {name} -> {path}")
                continue
            project.addMapLayer(layer, False)
            node = group.addLayer(layer)
            node.setItemVisibilityChecked(visible)
        if group_name not in ("Kartierung", "Schummerung (LiDAR)"):
            group.setExpanded(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    project.write(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    qgs = QgsApplication([], False)
    qgs.initQgis()
    main()
    qgs.exitQgis()
