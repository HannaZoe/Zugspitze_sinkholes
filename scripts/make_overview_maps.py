"""Two overview maps for presentations, rendered with a QGIS print layout.

Runs in QGIS's own Python:

    "C:\\Program Files\\QGIS 3.44.8\\bin\\python-qgis-ltr.bat" scripts\\make_overview_maps.py

1. map_fieldpoints_2026.png: field points of the 27 Aug 2026 campaign by type,
   on the Aug 2026 Mavic 3M RGB ortho. Points with invalid coordinates (one
   QField point stored outside lat/lon range) are left out and reported.
2. map_inventory_2025.png: the Aug 2025 doline inventory (points by type) on
   the Aug 2025 RGB ortho, plus a zoom panel where the outlines are visible.
Everything in EPSG:32632; outputs go to Master_Thesis/outputs/figures/.
"""

import os
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsContrastEnhancement,
    QgsCoordinateReferenceSystem,
    QgsFeatureRequest,
    QgsFillSymbol,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemMap,
    QgsLayoutItemPicture,
    QgsLayoutItemScaleBar,
    QgsLayoutItemShape,
    QgsLayoutMeasurement,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsLegendRenderer,
    QgsLegendStyle,
    QgsMarkerSymbol,
    QgsMultiBandColorRenderer,
    QgsPrintLayout,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsRendererCategory,
    QgsUnitTypes,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor, QFont

ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2] / "Master_Thesis"))
CO = ROOT / "data" / "interim" / "coregistered_20cm"
FIG = ROOT / "outputs" / "figures"
CRS = QgsCoordinateReferenceSystem("EPSG:32632")
PAGE_W, PAGE_H = 420, 160  # mm, wide for a 16:9 slide with a title above
DPI = 200

FIELD_TYPES = [  # value, label, colour, marker
    ("Doline", "Doline", "230,60,40", "circle"),
    ("Cave Doline", "Cave doline", "120,40,160", "triangle"),
    ("Rillenkarst", "Rillenkarst", "40,120,220", "square"),
    ("Gully", "Gully", "40,160,120", "square"),
]
DOLINE_TYPES = [
    ("sinkhole - bowl", "Bowl", "70,150,230"),
    ("sinkhole - funnel", "Funnel", "235,120,30"),
    ("multi-sinkhole", "Multi-sinkhole", "40,170,90"),
    ("sinkhole - elongated", "Elongated", "200,60,170"),
    ("cave", "Cave", "220,30,30"),
    ("unclear", "Unclear", "150,150,150"),
]


def ortho(path: Path, name: str) -> QgsRasterLayer:
    """RGB with ONE 2-98 % stretch shared by all bands, so colours stay natural."""
    layer = QgsRasterLayer(str(path), name)
    dp = layer.dataProvider()
    cuts = [dp.cumulativeCut(b, 0.02, 0.98, layer.extent(), 250000) for b in (1, 2, 3)]
    lo, hi = min(c[0] for c in cuts), max(c[1] for c in cuts)
    renderer = QgsMultiBandColorRenderer(dp, 1, 2, 3)
    for b, setter in [(1, renderer.setRedContrastEnhancement),
                      (2, renderer.setGreenContrastEnhancement),
                      (3, renderer.setBlueContrastEnhancement)]:  # fmt: skip
        ce = QgsContrastEnhancement(dp.dataType(b))
        ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum)
        ce.setMinimumValue(lo)
        ce.setMaximumValue(hi)
        setter(ce)
    layer.setRenderer(renderer)
    return layer


def marker(colour: str, shape: str = "circle", size: str = "3.2") -> QgsMarkerSymbol:
    return QgsMarkerSymbol.createSimple({
        "name": shape, "color": colour, "outline_color": "255,255,255",
        "outline_width": "0.35", "size": size,
    })  # fmt: skip


def categorized(layer: QgsVectorLayer, field: str, classes, polygon=False, other=None):
    cats = []
    for value, label, colour, *shape in classes:
        sym = (
            QgsFillSymbol.createSimple(
                {"color": colour + ",60", "outline_color": colour, "outline_width": "1.0"}
            )  # fmt: skip
            if polygon
            else marker(colour, shape[0] if shape else "circle")
        )
        cats.append(QgsRendererCategory(value, sym, label))
    if other:
        cats.append(QgsRendererCategory(None, marker("170,170,170"), other))
    layer.setRenderer(QgsCategorizedSymbolRenderer(field, cats))


def outline(layer: QgsVectorLayer, colour="255,255,255", width="0.5", dash=True):
    layer.setRenderer(layer.renderer().clone())
    sym = QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": colour,
                                      "outline_width": width,
                                      "outline_style": "dash" if dash else "solid"})  # fmt: skip
    layer.renderer().setSymbol(sym)


def add_map(layout, layers, extent, x, y, w, h, frame="0,0,0"):
    m = QgsLayoutItemMap(layout)
    m.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    m.attemptResize(QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters))
    m.setCrs(CRS)
    m.setLayers(layers)
    m.zoomToExtent(extent)
    m.setFrameEnabled(True)
    m.setFrameStrokeColor(QColor(*map(int, frame.split(","))))
    layout.addLayoutItem(m)
    return m


def label(layout, text, x, y, w, h, size=14, bold=True, bg=True):
    lab = QgsLayoutItemLabel(layout)
    lab.setText(text)
    font = QFont("Arial", size)
    font.setBold(bold)
    lab.setFont(font)
    lab.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    lab.attemptResize(QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters))
    lab.setMarginX(2)
    if bg:
        lab.setBackgroundEnabled(True)
        lab.setBackgroundColor(QColor(255, 255, 255, 215))
    layout.addLayoutItem(lab)
    return lab


def scalebar(layout, m, x, y, step, segments=4):
    bar = QgsLayoutItemScaleBar(layout)
    bar.setStyle("Single Box")
    bar.setLinkedMap(m)
    bar.setUnits(QgsUnitTypes.DistanceMeters)
    bar.setUnitLabel("m")
    bar.setNumberOfSegments(segments)
    bar.setNumberOfSegmentsLeft(0)
    bar.setUnitsPerSegment(step)
    bar.setBackgroundEnabled(True)
    bar.setBackgroundColor(QColor(255, 255, 255, 215))
    layout.addLayoutItem(bar)
    bar.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
    return bar


def decorations(layout, m, legend_layers, legend_title, credit, map_w=PAGE_W):
    legend = QgsLayoutItemLegend(layout)
    legend.setTitle(legend_title)
    legend.setAutoUpdateModel(False)
    root = legend.model().rootGroup()
    root.removeAllChildren()
    for lyr in legend_layers:
        node = root.addLayer(lyr)
        QgsLegendRenderer.setNodeLegendStyle(node, QgsLegendStyle.Hidden)  # no layer-name line
    legend.setLinkedMap(m)
    legend.setBackgroundEnabled(True)
    legend.setBackgroundColor(QColor(255, 255, 255, 225))
    for style, size in [(QgsLegendStyle.Title, 11), (QgsLegendStyle.SymbolLabel, 9),
                        (QgsLegendStyle.Subgroup, 9)]:  # fmt: skip
        f = legend.style(style).textFormat()
        f.setSize(size)
        legend.rstyle(style).setTextFormat(f)
    layout.addLayoutItem(legend)
    legend.attemptMove(QgsLayoutPoint(6, PAGE_H - 70, QgsUnitTypes.LayoutMillimeters))

    scalebar(layout, m, map_w - 75, PAGE_H - 30, 50)

    north = QgsLayoutItemPicture(layout)
    svg = next((Path(p) / "arrows" / "NorthArrow_02.svg" for p in QgsApplication.svgPaths()
                if (Path(p) / "arrows" / "NorthArrow_02.svg").exists()), None)  # fmt: skip
    if svg:
        north.setPicturePath(str(svg))
        north.attemptMove(QgsLayoutPoint(map_w - 22, 6, QgsUnitTypes.LayoutMillimeters))
        north.attemptResize(QgsLayoutSize(14, 18, QgsUnitTypes.LayoutMillimeters))
        layout.addLayoutItem(north)
    label(layout, credit, map_w - 140, PAGE_H - 9, 136, 6, size=7, bold=False)


def new_layout(project, name):
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(name)
    layout.pageCollection().page(0).setPageSize(QgsLayoutSize(PAGE_W, PAGE_H))
    return layout


def export(layout, path: Path):
    s = QgsLayoutExporter.ImageExportSettings()
    s.dpi = DPI
    res = QgsLayoutExporter(layout).exportToImage(str(path), s)
    print(f"  {'wrote' if res == QgsLayoutExporter.Success else 'FAILED'} {path.name}")


def main() -> None:
    project = QgsProject.instance()
    project.setCrs(CRS)
    fp_all = QgsVectorLayer(f"{ROOT / 'data/processed/uav_footprints.gpkg'}|layername=footprints",
                            "flights", "ogr")  # fmt: skip

    # --- 1: field points on the 2026 ortho
    o26 = ortho(CO / "20260827_terra_rtk_rgb_om_20cm_local_coregistration.tif", "Aug 2026 ortho")
    pts = QgsVectorLayer(str(ROOT / "fieldwork/field_data/fieldwork_annotations.gpkg"),
                         "Field points 27 Aug 2026", "ogr")  # fmt: skip
    pts.setSubsetString("ST_X(geom) BETWEEN -180 AND 180 AND ST_Y(geom) BETWEEN -90 AND 90")
    n_all = QgsVectorLayer(str(ROOT / "fieldwork/field_data/fieldwork_annotations.gpkg"),
                           "all", "ogr").featureCount()  # fmt: skip
    print(f"  field points: {pts.featureCount()} of {n_all} shown (invalid coordinates left out)")
    categorized(pts, "type", FIELD_TYPES, other="No type")
    area26 = QgsVectorLayer(fp_all.source(), "Aug 2026 flight (Mavic 3M)", "ogr")
    area26.setSubsetString("\"flight\" = 'Aug 2026 (Mavic 3M)'")
    outline(area26)
    for lyr in (o26, pts, area26):
        project.addMapLayer(lyr, False)
    ext = area26.extent()
    ext.grow(40)
    layout = new_layout(project, "field points")
    m = add_map(layout, [pts, o26], ext, 0, 0, PAGE_W, PAGE_H)
    label(layout, f"Field campaign 27 Aug 2026: {pts.featureCount()} field points", 6, 6, 170, 10)
    decorations(
        layout,
        m,
        [pts],
        "Field points",
        "Background: UAV RGB ortho 27 Aug 2026 (Mavic 3M), 20 cm. EPSG:32632.",
    )
    export(layout, FIG / "map_fieldpoints_2026.png")

    # --- 2: Aug 2025 inventory with a zoom panel
    o25 = ortho(CO / "20250820_terra_ppk_rgb_om.tif", "Aug 2025 ortho")
    dpts = QgsVectorLayer(f"{ROOT / 'data/manual/sinkhole_points.gpkg'}|layername=sinkhole_points",
                          "Dolines (points)", "ogr")  # fmt: skip
    categorized(dpts, "type", DOLINE_TYPES)
    polys = QgsVectorLayer(f"{ROOT / 'data/manual/sinkhole_polygons.gpkg'}|layername=sinkholes",
                           "Outlines", "ogr")  # fmt: skip
    polys.setSubsetString("\"epoch\" = '2025-08-20'")
    categorized(polys, "type", DOLINE_TYPES, polygon=True)
    area25 = QgsVectorLayer(fp_all.source(), "Aug 2025 flight", "ogr")
    area25.setSubsetString("\"flight\" LIKE 'Aug 2025%'")
    outline(area25)
    for lyr in (o25, dpts, polys, area25):
        project.addMapLayer(lyr, False)

    # zoom window: the 64 x 40 m box holding the most dolines
    hw, hh = 32, 20
    centres = [f.geometry().asPoint() for f in dpts.getFeatures()]
    best = max(centres, key=lambda c: sum(abs(p.x() - c.x()) < hw and abs(p.y() - c.y()) < hh
                                          for p in centres))  # fmt: skip
    zoom = QgsRectangle(best.x() - hw, best.y() - hh, best.x() + hw, best.y() + hh)
    n_zoom = sum(1 for _ in polys.getFeatures(QgsFeatureRequest().setFilterRect(zoom)))

    layout = new_layout(project, "inventory")
    main_w = 280
    ext = area25.extent()
    ext.grow(30)
    m = add_map(layout, [dpts, o25], ext, 0, 0, main_w, PAGE_H)
    z = add_map(layout, [polys, o25], zoom, main_w + 4, 22, PAGE_W - main_w - 4, 85)
    scalebar(layout, z, main_w + 6, 112, 5, segments=4)
    box = QgsLayoutItemShape(layout)
    box.setShapeType(QgsLayoutItemShape.Rectangle)
    box.setSymbol(QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": "255,230,0",
                                              "outline_width": "0.8"}))  # fmt: skip
    layout.addLayoutItem(box)
    # place the zoom box on the main map
    mx = m.extent()
    sx, sy = main_w / mx.width(), PAGE_H / mx.height()
    box.attemptMove(QgsLayoutPoint((zoom.xMinimum() - mx.xMinimum()) * sx,
                                   (mx.yMaximum() - zoom.yMaximum()) * sy,
                                   QgsUnitTypes.LayoutMillimeters))  # fmt: skip
    box.attemptResize(QgsLayoutSize(zoom.width() * sx, zoom.height() * sy,
                                    QgsUnitTypes.LayoutMillimeters))  # fmt: skip
    z.setFrameStrokeColor(QColor(255, 230, 0))
    z.setFrameStrokeWidth(QgsLayoutMeasurement(0.8, QgsUnitTypes.LayoutMillimeters))
    label(layout, f"Doline inventory Aug 2025: {dpts.featureCount()} dolines", 6, 6, 150, 10)
    label(layout, f"Zoom: outlines ({n_zoom} dolines in 64 x 40 m)", main_w + 4, 6,
          PAGE_W - main_w - 4, 10, size=11)  # fmt: skip
    decorations(
        layout,
        m,
        [dpts],
        "Doline type",
        "Background: UAV RGB ortho 20 Aug 2025 (Zenmuse L1), 20 cm. EPSG:32632.",
        map_w=main_w,
    )
    export(layout, FIG / "map_inventory_2025.png")


if __name__ == "__main__":
    qgs = QgsApplication([], False)
    qgs.initQgis()
    main()
    qgs.exitQgis()
