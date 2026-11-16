"""Point layer of the Aug 2026 field photos, so they can be viewed while mapping.

Two sources, both under Master_Thesis/fieldwork/field_data/raw/ (gitignored):
- QField notes (sinkholes/notes.gpkg + sinkholes/DICM/): these photos carry
  no GPS in the image. QField links each photo to a note point instead
  (layer notes_attachments -> notes_point), so the photo gets the location of
  its note.
- Phone photos (phone_photos/*.jpg): located by the GPS position in their
  EXIF header. Photos without GPS are listed and skipped.

Output: fieldwork/field_data/field_photos.gpkg (EPSG:32632), one point per
photo with `photo` = path relative to Master_Thesis/qgis/projects/, so the
QGIS project can show the image (attachment widget / map tip).
"""

import os

import geopandas as gpd
import pandas as pd
from PIL import ExifTags, Image
from shapely.geometry import Point

from master_thesis.paths import FIELD_DATA

CRS = "EPSG:32632"
RAW = FIELD_DATA / "raw"
NOTES = RAW / "sinkholes" / "notes.gpkg"
PHONE = RAW / "phone_photos"
PROJECT_DIR = FIELD_DATA.parents[1] / "qgis" / "projects"
OUT = FIELD_DATA / "field_photos.gpkg"
GPS_IFD = 0x8825


def rel(path) -> str:
    return os.path.relpath(path, PROJECT_DIR).replace("\\", "/")


def qfield_photos() -> gpd.GeoDataFrame:
    points = gpd.read_file(NOTES, layer="notes_point")
    media = gpd.read_file(NOTES, layer="notes_attachments")
    df = media.merge(points, left_on="note_uuid", right_on="uuid", how="inner")
    df["file"] = df.media.str.replace("DCIM/", "", regex=False)
    df = df[[(RAW / "sinkholes" / "DICM" / f).exists() for f in df.file]]
    return gpd.GeoDataFrame({
        "photo": [rel(RAW / "sinkholes" / "DICM" / f) for f in df.file],
        "source": "QField note",
        "taken": df.timestamp_x.astype(str),
        "label": (df.title.fillna("") + " " + df.note.fillna("")).str.strip(),
    }, geometry=df.geometry.values, crs=points.crs).to_crs(CRS)  # fmt: skip


def _deg(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    return (d + m / 60 + s / 3600) * (-1 if ref in ("S", "W") else 1)


def phone_photos() -> gpd.GeoDataFrame:
    rows, missing = [], []
    for f in sorted(PHONE.glob("*.jp*g")) if PHONE.exists() else []:
        exif = Image.open(f).getexif()
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(GPS_IFD).items()}
        if "GPSLatitude" not in gps:
            missing.append(f.name)
            continue
        lat = _deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = _deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        rows.append({"photo": rel(f), "source": "phone", "taken": str(exif.get(306, "")),
                     "label": f.stem, "geometry": Point(lon, lat)})  # fmt: skip
    if missing:
        print(f"  {len(missing)} phone photos without GPS, skipped: {', '.join(missing[:10])}")
    if not rows:
        return gpd.GeoDataFrame(columns=["photo", "source", "taken", "label", "geometry"], crs=CRS)
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS)


def main() -> None:
    parts = [qfield_photos(), phone_photos()]
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    out.to_file(OUT, layer="field_photos", driver="GPKG")
    print(out.groupby("source").size().to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
