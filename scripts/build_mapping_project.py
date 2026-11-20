"""Build the QGIS mapping project with every co-registered layer, ready to digitise.

Runs in QGIS's own Python, not the uv environment:

    "C:\\Program Files\\QGIS 3.44.8\\bin\\python-qgis-ltr.bat" scripts\\build_mapping_project.py

Paths follow master_thesis.paths (sibling Master_Thesis repo, or THESIS_ROOT),
but without importing the package, which isn't installed in QGIS's Python.
Everything is EPSG:32632 on the common 20 cm grid, i.e. already aligned to the
20250820 reference, so layers can be toggled against each other directly.
Writes Master_Thesis/qgis/projects/kartierung.qgz (relative paths). If the
project already exists it is only extended: layers whose name is not in it yet
are added, everything else (styles, extent, own layers) stays as saved.
The older Zugspitze.qgz is left alone.
The doline point layer (same fields as the polygon layer, auto-numbered id)
is created on first run if it doesn't exist yet. New groups and layers are
inserted at their place in GROUPS, so vectors stay above the rasters.
"""

import os
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext,
    QgsDefaultValue,
    QgsEditorWidgetSetup,
    QgsFillSymbol,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
    QgsWkbTypes,
)
from qgis.PyQt.QtGui import QColor

ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2] / "Master_Thesis"))
CO = ROOT / "data" / "interim" / "coregistered_20cm"
OUT = ROOT / "qgis" / "projects" / "kartierung.qgz"
FOOTPRINTS = ROOT / "data" / "processed" / "uav_footprints.gpkg"
PHOTOS = ROOT / "fieldwork" / "field_data" / "field_photos.gpkg"
POLYGONS = ROOT / "data" / "manual" / "sinkhole_polygons.gpkg"
POINTS = ROOT / "data" / "manual" / "sinkhole_points.gpkg"
POINT_LAYER = "Dolinen-Punkte (zu setzen)"
DEPR = ROOT / "data" / "processed" / "depressions"
REVIEW = ROOT / "data" / "processed" / "mapping_review.gpkg"
# LfU Bayern digital geological map 1:25,000 (WMS, CC BY 4.0): "wms:<layer>" paths
DGK25 = "https://www.lfu.bayern.de/gdi/wms/geologie/dgk25?"
WMS_OPACITY = {"geoleinheit_dgk25": 0.45, "dgk25_raster": 0.6}
# closed-depression outlines per depth level: level -> (colour, width)
DEPR_STYLE = {0.1: ("0,220,255", "0.3"), 0.3: ("0,120,255", "0.5"), 1.0: ("0,0,160", "0.7")}
PHOTO_LAYER = "Feldfotos (Maus drauf = Foto)"
# hover tip: the photo itself; paths in the layer are relative to the project folder
PHOTO_TIP = (
    '<img src="file:///[% @project_folder || \'/\' || "photo" %]" width="420"/>'
    '<br/>[% "label" %] ([% "source" %])'
)
# polygon layers drawn as outline only: layer name -> outline colour
OUTLINES = {
    "Umriss groesste Befliegung (Aug 2026 3M)": "0,0,0",
    "Umrisse aller Fluege": "120,120,120",
    "Dolinen-Polygone (zu digitalisieren)": "230,30,30",
    "Tiefe Senken ohne Punkt (> 1 m)": "255,0,255",
}
# review points: name -> marker colour
REVIEW_POINTS = {
    "Punkte ohne Senke (> 5 m)": "255,140,0",
    "Alte Punkte ohne Treffer (> 10 m)": "0,200,0",
}

# (group, [(layer name, path, visible)]), top of the legend first
GROUPS = [
    ("Kartierung", [
        (POINT_LAYER, f"{POINTS}|layername=sinkhole_points", True),
        ("Dolinen-Polygone (zu digitalisieren)", f"{POLYGONS}|layername=sinkholes", True),
        ("Feldpunkte Aug 2026",
         str(ROOT / "fieldwork/field_data/fieldwork_annotations.gpkg"), True),
        (PHOTO_LAYER, f"{PHOTOS}|layername=field_photos", True),
        ("Dolinenpunkte (Sinkholes.shp)", str(ROOT / "data/manual/Sinkholes.shp"), True),
        ("Verdachtsflaechen", str(ROOT / "data/manual/Suspicious_Areas.shp"), False),
        ("Umriss groesste Befliegung (Aug 2026 3M)", f"{FOOTPRINTS}|layername=largest", True),
        ("Umrisse aller Fluege", f"{FOOTPRINTS}|layername=footprints", False),
    ]),
    ("Zu pruefen (Inventar-Abgleich)", [
        ("Tiefe Senken ohne Punkt (> 1 m)", f"{REVIEW}|layername=deep_unmarked", True),
        ("Punkte ohne Senke (> 5 m)", f"{REVIEW}|layername=points_no_depression", True),
        ("Alte Punkte ohne Treffer (> 10 m)", f"{REVIEW}|layername=older_unmatched", True),
    ]),
    ("Senken aus LiDAR-DSM (Hilfe)", [
        ("Senkenkonturen Aug 2025", f"{DEPR / 'depressions.gpkg'}|layername=20250820", True),
        ("Senkenkonturen Okt 2024", f"{DEPR / 'depressions.gpkg'}|layername=20241025", False),
        ("Senkenkonturen Jun 2025 (Schnee!)",
         f"{DEPR / 'depressions.gpkg'}|layername=20250604", False),
        ("Senkentiefe Aug 2025 [m]", DEPR / "20250820_depth.tif", False),
    ]),
    ("Geologie dGK25 (LfU Bayern, CC BY 4.0)", [
        ("Stoerungen / Strukturlinien (dGK25)", "wms:strukturln_dgk25", True),
        ("Geologische Einheiten (dGK25)", "wms:geoleinheit_dgk25", False),
        ("GK25 gescannt mit Kuerzeln", "wms:dgk25_raster", False),
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


def setup_photos(layer: QgsVectorLayer) -> None:
    """Show the linked image on hover (map tip) and in the identify/attribute form."""
    layer.renderer().setSymbol(QgsMarkerSymbol.createSimple({
        "name": "square", "color": "255,160,0", "outline_color": "0,0,0", "size": "2.6",
    }))  # fmt: skip
    layer.setEditorWidgetSetup(layer.fields().indexOf("photo"), QgsEditorWidgetSetup(
        "ExternalResource", {"DocumentViewer": 1, "RelativeStorage": 1, "FileWidget": True},
    ))  # fmt: skip
    layer.setMapTipTemplate(PHOTO_TIP)
    layer.setMapTipsEnabled(True)


def ensure_point_layer() -> None:
    """Create the doline point layer with the polygon layer's fields, if missing."""
    if POINTS.exists():
        return
    polys = QgsVectorLayer(f"{POLYGONS}|layername=sinkholes", "p", "ogr")
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName, opts.layerName = "GPKG", "sinkhole_points"
    writer = QgsVectorFileWriter.create(
        str(POINTS), polys.fields(), QgsWkbTypes.Point,
        QgsCoordinateReferenceSystem("EPSG:32632"), QgsCoordinateTransformContext(), opts,
    )  # fmt: skip
    del writer  # flushes and closes the file
    print(f"  created {POINTS}")


def setup_points(layer: QgsVectorLayer) -> None:
    """Red dots labelled with their id; id numbers itself (max + 1)."""
    layer.renderer().setSymbol(QgsMarkerSymbol.createSimple({
        "name": "circle", "color": "230,30,30", "outline_color": "255,255,255", "size": "2.4",
    }))  # fmt: skip
    idx = layer.fields().indexOf("id")
    layer.setDefaultValueDefinition(idx, QgsDefaultValue('coalesce(maximum("id"), 0) + 1'))
    label = QgsPalLayerSettings()
    label.fieldName = "id"
    fmt = label.format()
    fmt.setColor(QColor("yellow"))
    fmt.buffer().setEnabled(True)
    label.setFormat(fmt)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(label))
    layer.setLabelsEnabled(True)


def setup_depressions(layer) -> None:
    if isinstance(layer, QgsRasterLayer):  # depth: transparent at 0, dark blue at 2 m
        ramp = QgsColorRampShader(0, 2, None, QgsColorRampShader.Interpolated)
        ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(0.05, QColor(255, 255, 255, 0), "0"),
            QgsColorRampShader.ColorRampItem(0.3, QColor(120, 190, 255, 160), "0.3"),
            QgsColorRampShader.ColorRampItem(2.0, QColor(0, 0, 140, 220), "2 m"),
        ])  # fmt: skip
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(ramp)
        layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))
        return
    cats = [
        QgsRendererCategory(
            lvl,
            QgsFillSymbol.createSimple(
                {
                    "color": "0,0,0,0",
                    "outline_color": col,
                    "outline_width": w,
                }
            ),
            f"{lvl} m tief",
        )  # fmt: skip
        for lvl, (col, w) in DEPR_STYLE.items()
    ]
    layer.setRenderer(QgsCategorizedSymbolRenderer("level_m", cats))


def main() -> None:
    project = QgsProject.instance()
    if OUT.exists():
        project.read(str(OUT))
    else:
        project.setCrs(QgsCoordinateReferenceSystem("EPSG:32632"))
    ensure_point_layer()

    # skip layers already in the project by name *or* source (Hanna renames layers)
    def norm(src: str) -> str:
        return src.replace("\\", "/").lower()

    layers_now = project.mapLayers().values()
    existing = {layer.name() for layer in layers_now} | {norm(lyr.source()) for lyr in layers_now}
    root = project.layerTreeRoot()
    for g_idx, (group_name, layers) in enumerate(GROUPS):
        group = root.findGroup(group_name) or root.insertGroup(g_idx, group_name)
        for l_idx, (name, path, visible) in enumerate(layers):
            path = str(path)
            if name in existing or norm(path) in existing:
                continue
            vector = path.endswith((".gpkg", ".shp")) or "|layername=" in path
            if path.startswith("wms:"):
                wms = path[4:]
                uri = (f"crs=EPSG:25832&dpiMode=7&format=image/png&layers={wms}"
                       f"&styles=&url={DGK25}")  # fmt: skip
                layer = QgsRasterLayer(uri, name, "wms")
                if layer.isValid():
                    layer.renderer().setOpacity(WMS_OPACITY.get(wms, 1.0))
            elif vector:
                layer = QgsVectorLayer(path, name, "ogr")
            else:
                layer = QgsRasterLayer(path, name)
            if not layer.isValid():
                print(f"  skipped (not found / invalid): {name} -> {path}")
                continue
            if name in OUTLINES:
                layer.renderer().setSymbol(QgsFillSymbol.createSimple({
                    "color": "0,0,0,0", "outline_color": OUTLINES[name], "outline_width": "0.6",
                }))  # fmt: skip
            if name in REVIEW_POINTS:
                layer.renderer().setSymbol(QgsMarkerSymbol.createSimple({
                    "name": "diamond", "color": "0,0,0,0", "outline_color": REVIEW_POINTS[name],
                    "outline_width": "0.6", "size": "4",
                }))  # fmt: skip
            if name == PHOTO_LAYER:
                setup_photos(layer)
            if name == POINT_LAYER:
                setup_points(layer)
            if name.startswith("Senken"):
                setup_depressions(layer)
            project.addMapLayer(layer, False)
            pos = min(l_idx, len(group.children()))
            group.insertLayer(pos, layer).setItemVisibilityChecked(visible)
            print(f"  added {name}")
        if group_name not in (
            "Kartierung",
            "Schummerung (LiDAR)",
            "Senken aus LiDAR-DSM (Hilfe)",
            "Zu pruefen (Inventar-Abgleich)",
        ):
            group.setExpanded(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    project.write(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    qgs = QgsApplication([], False)
    qgs.initQgis()
    main()
    qgs.exitQgis()
