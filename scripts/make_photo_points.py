"""Point layer of the Aug 2026 field photos, so they can be viewed while mapping.

Two sources, both under Master_Thesis/fieldwork/field_data/raw/ (gitignored):
- QField notes (sinkholes/notes.gpkg + sinkholes/DICM/): these photos carry
  no GPS in the image. QField links each photo to a note point instead
  (layer notes_attachments -> notes_point), so the photo gets the location of
  its note.
- Phone photos (phone_photos/*.jpg): located by the GPS position in their
  EXIF header. The Aug 2026 ones have empty (NaN) GPS tags, so they fall back
  to time matching: the photo gets the position of the field point or QField
  note logged closest in time (phone clock = QField clock to a few seconds,
  checked on the QField photos). `dt_min` says how far apart that was; photos
  more than MAX_DT_MIN from any fix are skipped.

Output: fieldwork/field_data/field_photos.gpkg (EPSG:32632), one point per
photo with `photo` = path relative to Master_Thesis/qgis/projects/, so the
QGIS project can show the image (attachment widget / map tip).
"""

import math
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
ANNOTATIONS = FIELD_DATA / "fieldwork_annotations.gpkg"
TZ = "Europe/Berlin"  # camera EXIF time is local
MAX_DT_MIN = 15
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
        "located": "QField note",
        "dt_min": 0.0,
    }, geometry=df.geometry.values, crs=points.crs).to_crs(CRS)  # fmt: skip


def _deg(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    return (d + m / 60 + s / 3600) * (-1 if ref in ("S", "W") else 1)


def fixes() -> gpd.GeoDataFrame:
    """Every timestamped position logged in the field (field points + notes), UTC."""
    parts = [gpd.read_file(ANNOTATIONS), gpd.read_file(NOTES, layer="notes_point")]
    g = pd.concat([p.to_crs(CRS)[["timestamp", "geometry"]] for p in parts], ignore_index=True)
    g["timestamp"] = pd.to_datetime(g.timestamp, utc=True)
    return gpd.GeoDataFrame(g, crs=CRS).sort_values("timestamp").reset_index(drop=True)


def phone_photos() -> gpd.GeoDataFrame:
    rows, skipped = [], []
    track = fixes()
    for f in sorted(PHONE.glob("*.jp*g")) if PHONE.exists() else []:
        exif = Image.open(f).getexif()
        taken = str(exif.get(306, ""))
        row = {"photo": rel(f), "source": "phone", "taken": taken, "label": f.stem}
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(GPS_IFD).items()}
        try:
            lat = _deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
            lon = _deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        except (KeyError, ValueError, TypeError):
            lat = lon = math.nan
        if not math.isnan(lat):
            pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(CRS).iloc[0]
            rows.append(row | {"located": "EXIF GPS", "dt_min": 0.0, "geometry": pt})
            continue
        t = pd.Timestamp(taken.replace(":", "-", 2)).tz_localize(TZ).tz_convert("UTC")
        dt = (track.timestamp - t).abs()
        i = dt.idxmin()
        gap = dt.min()
        if gap > pd.Timedelta(minutes=MAX_DT_MIN):
            skipped.append(f.name)
            continue
        rows.append(row | {"located": "time match", "dt_min": round(gap.total_seconds() / 60, 1),
                           "geometry": track.geometry[i]})  # fmt: skip
    if skipped:
        print(f"  {len(skipped)} phone photos with no GPS and no fix within {MAX_DT_MIN} min: "
              f"{', '.join(skipped[:10])}")  # fmt: skip
    cols = ["photo", "source", "taken", "label", "located", "dt_min", "geometry"]
    return gpd.GeoDataFrame(rows or None, columns=cols, crs=CRS)


def main() -> None:
    parts = [qfield_photos(), phone_photos()]
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    out.to_file(OUT, layer="field_photos", driver="GPKG")
    print(out.groupby("source").size().to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
