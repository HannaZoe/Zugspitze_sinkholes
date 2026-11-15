"""Parse and merge the field-verification notes of 2026-08-27.

Two independent records from the same field day are combined; both are kept
unmodified in fieldwork/field_data/raw/. "lower" and "higher" refer to the
part of the AOI that was walked, not to GPS precision. The observer is not
carried into the output; the AOI part is kept as `aoi_section`.

1. Field-notes app export (`sinkholes_lower.zip`): 31 GPS points with fairly
   regular free-text dimension notes, lower part of the AOI. Parsed with
   regular expressions plus a few manual overrides.
2. QField layer export (`fieldwork_points_higher.shp`): 29 points, higher
   part of the AOI. These notes were typed in the field, with many typos and
   implicit units, so they were interpreted manually (HIGHER_PARSED) instead
   of with a generic pattern that could misread measurements.

Width, length and depth are only filled where a note states them
unambiguously. In the merged table (fieldwork/field_data/
fieldwork_annotations.gpkg), `type` carries no embedded "?" (field certainty
is not kept), `notes` is spelling-corrected but otherwise verbatim
(see the NOTE_FIXES dicts), and `flag` marks anything that needs a second
look: inferred units, vague or incomplete notes, a point marked wrong by a
later note, and one corrupted geometry in the raw QField export.
"""

import re
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

from master_thesis.paths import FIELD_DATA

ZIP_SOURCE = Path(r"C:\Users\Hanna\Downloads\sinkholes_lower.zip")
RAW_DIR = FIELD_DATA / "raw"
OUT_UNIFIED = FIELD_DATA / "fieldwork_annotations.gpkg"

# ---------------------------------------------------------------------------
# Source 1: field-notes app export (lower part of the AOI)
# ---------------------------------------------------------------------------

# fmt: off
MANUAL_OVERRIDES = {
    # "A bit less than 2m" -- single vague estimate, no height/width split
    "{5de309ec-5bd6-41bf-af14-81288970077b}": (None, None, None, None, True, None),
    # "Maybe 1,10m"
    "{a4d1862f-55ce-4b10-8865-eb679ecf6743}": (1.10, None, None, None, True, None),
    # "2,35m (pole) / Width: 3m and 4m"
    "{aa0d41d5-09c2-40a2-bd43-d4bf98981c22}": (2.35, 3.0, 4.0, "pole", False, None),
}
# fmt: on
NOTE_PATTERN = re.compile(
    r"(?P<depth>\d+[.,]?\d*)\s*(?:m|cm)?\s*(?:height|high)\s*\n?\s*(?:[\w,]+\s+)*?"
    # only "and"/"x"/"times" separate two widths; a bare comma is the decimal
    # mark (e.g. "2,50m"), and no note in this dataset uses one between widths
    r"(?P<w1>\d+[.,]?\d*)\s*m?\s*(?:and|x|times)\s*(?P<w2>\d+[.,]?\d*)\s*m?\s*(?:wide|wise)",
    re.IGNORECASE,
)
# fallback for a stated range instead of two measurements, e.g. "Pretty round, 5-6m wide"
RANGE_PATTERN = re.compile(
    r"(?P<w1>\d+[.,]?\d*)\s*-\s*(?P<w2>\d+[.,]?\d*)\s*m\s*(?:wide|wise)", re.IGNORECASE
)
# fallback for a single width, e.g. "2,50m wide" or "2,80m wide in both ways";
# width2_m stays null in both cases, since only one measurement was taken
SINGLE_WIDTH_PATTERN = re.compile(r"(?P<w1>\d+[.,]?\d*)\s*m\s*wide", re.IGNORECASE)
DEPTH_ONLY_PATTERN = re.compile(
    r"(?P<depth>\d+[.,]?\d*)\s*(?:m|cm)?\s*(?:height|high)", re.IGNORECASE
)
CM_PATTERN = re.compile(r"(\d+[.,]?\d*)\s*cm\s*(?:height|high)", re.IGNORECASE)
CAVE_PATTERN = re.compile(r"wide\s*and\s*min\.?\s*(?P<depth>\d+[.,]?\d*)\s*m\s*tief", re.IGNORECASE)
CAVE_WIDTH_PATTERN = re.compile(r"(?P<w>\d+[.,]?\d*)\s*m\s*wide", re.IGNORECASE)
APPROX_WORDS = re.compile(r"\b(maybe|approx|a bit|min\.|less than)\b", re.IGNORECASE)


def to_float(s: str) -> float:
    return float(s.replace(",", "."))


def parse_note_lower(title: str, note: str, uuid: str) -> dict:
    result = {
        "depth_m": None,
        "width1_m": None,
        "width2_m": None,
        "measurement_method": None,
        "approx": False,
        "parse_status": "unparsed",
    }
    if not isinstance(note, str) or not note.strip():
        result["parse_status"] = "no_note"
        return result

    result["approx"] = bool(APPROX_WORDS.search(note))

    if uuid in MANUAL_OVERRIDES:
        depth, w1, w2, method, approx, _ = MANUAL_OVERRIDES[uuid]
        result.update(
            depth_m=depth,
            width1_m=w1,
            width2_m=w2,
            measurement_method=method,
            approx=approx,
            parse_status="manual_override",
        )
        return result

    if title == "Cave Sinkhole?":
        cave_depth = CAVE_PATTERN.search(note)
        cave_width = CAVE_WIDTH_PATTERN.search(note)
        if cave_depth and cave_width:
            result.update(
                depth_m=to_float(cave_depth.group("depth")),
                width1_m=to_float(cave_width.group("w")),
                approx=True,
                parse_status="manual_override",
            )
        return result

    m = NOTE_PATTERN.search(note)
    if m:
        depth = to_float(m.group("depth"))
        if CM_PATTERN.search(note):
            depth = depth / 100
        result.update(
            depth_m=depth,
            width1_m=to_float(m.group("w1")),
            width2_m=to_float(m.group("w2")),
            parse_status="regex",
        )
        return result

    depth_m = DEPTH_ONLY_PATTERN.search(note)

    range_m = RANGE_PATTERN.search(note)
    if range_m and depth_m:
        depth = to_float(depth_m.group("depth"))
        if CM_PATTERN.search(note):
            depth = depth / 100
        result.update(
            depth_m=depth,
            width1_m=to_float(range_m.group("w1")),
            width2_m=to_float(range_m.group("w2")),
            approx=True,
            parse_status="regex_range",
        )
        return result

    single_m = SINGLE_WIDTH_PATTERN.search(note)
    if single_m and depth_m:
        depth = to_float(depth_m.group("depth"))
        if CM_PATTERN.search(note):
            depth = depth / 100
        result.update(
            depth_m=depth,
            width1_m=to_float(single_m.group("w1")),
            parse_status="regex_single_width",
        )
        return result

    result["parse_status"] = "no_dimensions"
    return result


def build_lower() -> gpd.GeoDataFrame:
    src_gpkg = RAW_DIR / "sinkholes" / "notes.gpkg"
    pts = gpd.read_file(src_gpkg, layer="notes_point")

    parsed = pts.apply(
        lambda row: parse_note_lower(row["title"], row["note"], row["uuid"]),
        axis=1,
        result_type="expand",
    )
    out = pd.concat([pts[["uuid", "title", "note", "timestamp", "geometry"]], parsed], axis=1)
    out = out.rename(columns={"note": "raw_note"})
    return gpd.GeoDataFrame(out, geometry="geometry", crs=pts.crs)


# ---------------------------------------------------------------------------
# Source 2: QField export (higher part of the AOI), interpreted manually.
# Keyed by row position (0-28): (depth_m, width1_m, width2_m, unit_inferred,
# review_flag). Dimensions are None unless the note states them, either
# explicitly or through an unambiguous unit inference (a bare "435" next to
# "4m" is read as 435 cm), which `unit_inferred` marks. `review_flag` covers
# cut-off text, ambiguous numbers, corrections by later notes, and
# rock-drop timings, which are not distance measurements.
# ---------------------------------------------------------------------------

# fmt: off
HIGHER_PARSED = {
    0:  (0.63, 1.30, 1.57, False, "Point 1 says this one 'was marked wrong' -- verify before trusting."),  # noqa: E501
    1:  (None, None, None, False, "Correction note about point 0, not a new location/measurement."),  # noqa: E501
    2:  (1.23, 4.00, 4.35, True,  None),  # noqa: E501
    3:  (0.84, 1.84, None, True,  "'184*2' -- text looks cut off, second width not usable."),  # noqa: E501
    4:  (1.22, 3.30, 3.70, True,  None),  # noqa: E501
    5:  (0.52, 1.65, 1.50, True,  "Order in note is width*width then depth. Also: 'use this data to calculate roughness' -- her own flagged intent, not yet done."),  # noqa: E501
    6:  (17.5, None, None, False, "Cave; depth given as a range '15-20m', used midpoint. Same cluster as points 7-8 (near-identical coords)."),  # noqa: E501
    7:  (None, None, None, False, "'14.4 seconds rock falling' -- a timed drop, not a direct depth measurement. Same cave as 6/8."),  # noqa: E501
    8:  (15.4, 10.0, 10.0, False, "Same cave as 6/7. 'Breite' = German for width."),  # noqa: E501
    9:  (1.80, 5.50, 3.90, False, "Located in the middle of the hiking path -- flag for the hazard/safety discussion."),  # noqa: E501
    10: (None, 8.00, None, True,  "Cave; only a 9.5s rock-drop timing + one width (8m), second width marked '??' by Hanna herself."),  # noqa: E501
    11: (0.86, 2.00, 0.90, True,  "Notes: 'almost no small rubble, a lot of frost weathering' -- relevant to the frost-weathering lit thread."),  # noqa: E501
    12: (1.30, 4.83, 5.40, True,  None),  # noqa: E501
    13: (1.85, 2.70, None, True,  "Only one width given."),  # noqa: E501
    14: (28.8, 11.8, None, False, "Cave, distinct from the 6/7/8 and 10 caves (different depth). '8s drop' corroborating note."),  # noqa: E501
    15: (None, None, None, False, "Descriptive only (gully leading to a sinkhole), no dimensions given."),  # noqa: E501
    16: (1.35, 3.60, 5.20, True,  None),  # noqa: E501
    17: (1.70, 4.60, 3.90, True,  None),  # noqa: E501
    18: (None, None, None, False, "Empty: no type, no note at all."),  # noqa: E501
    19: (None, None, None, False, "Type set (Doline confirmed) but no note/dimensions recorded."),  # noqa: E501
    20: (None, None, None, False, "Type set (Doline confirmed) but no note/dimensions recorded."),  # noqa: E501
    21: (2.10, 4.60, 4.60, True,  "'3er cluster' -- Hanna's note this is one of three close-together dolines; 'roughly round' supports the equal width1/width2."),  # noqa: E501
    22: (None, None, None, False, "'10*17' with no depth/width labels at all in the original note -- meaning is genuinely unclear, not guessed."),  # noqa: E501
    23: (None, 4.00, 2.60, False, "Cave; '5s drop, long silence' is a timing note not a depth. 'length' read as width1, explicit '2,60 wide' as width2."),  # noqa: E501
    24: (None, None, None, False, "'Same cave\\nDepth:' -- continues point 23, but the depth value was never typed in. Needs Hanna's recall or a field-notebook cross-check."),  # noqa: E501
    25: (None, None, None, False, "'15*11m depth / 10m' -- ambiguous which number is depth vs. width, not guessed."),  # noqa: E501
    26: (17.5, 39.0, None, True,  "Hanna's own description: 'huge multiple complex karst dolines, birds nesting in there, altered rock, super complex' -- likely not a single simple doline, treat dimensions as approximate for a complex merged feature."),  # noqa: E501
    27: (16.0, None, None, False, "'Frost weathering, small caves in between' -- relevant to the frost-weathering lit thread. No width given."),  # noqa: E501
    28: (1.00, 4.40, None, True,  "'440xm' -- read as a likely 'cm' typo; flagged, not fully certain."),  # noqa: E501
}
# fmt: on


LOWER_TYPE_MAP = {
    "Sinkhole?": "Doline",
    "Sinkhole": "Doline",
    "Sinkhole? (very small)": "Doline",
    "Cave Sinkhole?": "Cave Doline",
    "Rillenkarst": "Rillenkarst",
}
# all QField points typed "Other" are caves (see HIGHER_PARSED); the
# "(confirmed)" suffix is dropped
HIGHER_TYPE_MAP = {
    "Doline (confirmed)": "Doline",
    "Other": "Cave Doline",
    "Gully": "Gully",
}
# Spelling fixes: typos and autocorrect errors only, no rewording. Two
# ambiguous cases were resolved by the observer: "lite of frost weathering"
# -> "a lot of" (not "a little of") and "xace" -> "space". German words
# (schotter, Breite, sek, tief, "3er") are intentional and left unchanged.
LOWER_NOTE_FIXES = {
    "{43ac47e2-00ca-4927-aaaa-3b635be911b1}": "3m height \n4,20m and 5,20m wide",
}
HIGHER_NOTE_FIXES = {
    0: "63cm deep\n130 * 157 cm width\nGully behind it, lots of medium-sized gravel\n",
    1: "This is 63cm sinkhole, it was marked wrong",
    3: "4 dolines close by, all schotter filled\n84 deep\n184*2",
    5: "165*150\n52 depth\n\n\nUse this data to calculate roughness",
    6: (
        "Cave! Really deep, easily 15-20m\n"
        "On the summit side a lot of rock, other side more loose material"
    ),
    7: "14,4 seconds rock falling",
    8: "15,4 m depth measurable\nBreite 10m * 10m",
    9: "In the middle of the hiking path\nDepth 180cm\n5,50*3,90",
    10: (
        "Cave!!!\n\nMuch less angled than the other one, rillenkarst\n\n\n9,5 sek rock throw\n8m*??"
    ),
    11: "86cm deep\n2m*90\nReally clear cut, almost no small rubble, a lot of frost weathering",
    14: (
        "11,8 m width\n28.8 m deep\n\nCave!!\n"
        "Really deep, less angular, quite very vertical\n\n8s drop\n\n\n"
    ),
    21: "3er cluster\n210 deep\n460*460 roughly round\n",
    23: "Cave, 5s drop, long silence\n4m long, smaller space tho\n2,60 wide",
    26: (
        "17 5 wide 39m\nHuge multiple complex karst dolines\n\n"
        "Birds are nesting in there\nAltered rock, super complex"
    ),
}

# QField points left out of the merged table (positional index, see below)
DROPPED_HIGHER = {
    0: "marked wrong according to point 1",
    18: "empty: no type, no note",
    24: "duplicate of point 23, depth never typed in",
}


def build_unified(lower: gpd.GeoDataFrame, higher: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge both sources into one table.

    `aoi_section` ("lower"/"higher") replaces the per-tool source label.
    `type` carries no question marks (field certainty is not kept).
    `notes` has typos corrected but is otherwise
    verbatim. `flag` collects everything worth a second look, from doubts
    about the numbers to relevant context, in a single column.
    """
    lower_rows = []
    for row in lower.itertuples(index=False):
        type_ = LOWER_TYPE_MAP.get(row.title)
        note = LOWER_NOTE_FIXES.get(row.uuid, row.raw_note)
        if row.title == "Sinkhole? (very small)" and isinstance(note, str):
            note = f"[very small] {note}"

        flag = None
        if row.parse_status == "no_note":
            flag = "No note recorded."
        elif row.parse_status == "no_dimensions":
            flag = "No dimensions given in the note."
        elif (
            row.parse_status == "manual_override" and pd.isna(row.width1_m) and pd.isna(row.depth_m)
        ):
            flag = "Vague single estimate only, no clear width/depth split."
        elif row.measurement_method == "pole":
            flag = "Depth measured with a pole (likely more precise than a visual estimate)."
        elif row.approx:
            flag = "Note uses hedging language (maybe/approx/etc.) -- treat as approximate."

        lower_rows.append(
            {
                "aoi_section": "lower",
                "type": type_,
                "width_m": row.width1_m,
                "length_m": row.width2_m,
                "depth_m": row.depth_m,
                "notes": note,
                "flag": flag,
                "timestamp": row.timestamp,
                "geometry": row.geometry,
            }
        )

    higher_rows = []
    # positional index, not row.source_fid: HIGHER_PARSED and HIGHER_NOTE_FIXES
    # are keyed 0-28, while source_fid is the shapefile's 1-based fid (1-29)
    for position, row in enumerate(higher.itertuples(index=False)):
        if position in DROPPED_HIGHER:
            continue
        type_ = HIGHER_TYPE_MAP.get(row.type)
        note = HIGHER_NOTE_FIXES.get(position, row.raw_note)

        flag = row.review_flag
        if flag is None and row.unit_inferred:
            flag = "Unit inferred from magnitude, not stated explicitly in the note."
        if row.source_fid == 29:
            # corrupted geometry in the raw QField export (199257, 6974687, far
            # outside the site); the row is kept unchanged and flagged
            bad_geom_note = "Geometry is corrupted in the raw export (not a usable location)."
            flag = f"{flag} {bad_geom_note}" if flag else bad_geom_note

        higher_rows.append(
            {
                "aoi_section": "higher",
                "type": type_,
                "width_m": row.width1_m,
                "length_m": row.width2_m,
                "depth_m": row.depth_m,
                "notes": note,
                "flag": flag,
                "timestamp": row.created_at,
                "geometry": row.geometry,
            }
        )

    combined = gpd.GeoDataFrame(lower_rows + higher_rows, geometry="geometry", crs=lower.crs)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    return combined


def build_higher() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(RAW_DIR / "fieldwork_points_higher.shp")
    rows = []
    for fid, row in zip(HIGHER_PARSED.keys(), gdf.itertuples(index=False), strict=True):
        depth, w1, w2, inferred, flag = HIGHER_PARSED[fid]
        rows.append(
            {
                "source_fid": row.fid,
                "person": row.person,
                "type": row.type,
                "raw_note": row.notes,
                "depth_m": depth,
                "width1_m": w1,
                "width2_m": w2,
                "unit_inferred": inferred,
                "review_flag": flag,
                "created_at": row.created_at,
                "geometry": row.geometry,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not (RAW_DIR / "sinkholes").exists():
        shutil.unpack_archive(ZIP_SOURCE, RAW_DIR)
        print(f"extracted raw export to {RAW_DIR}")
    else:
        print(f"{RAW_DIR} already populated, skipping extraction")

    lower = build_lower()
    higher = build_higher()
    unified = build_unified(lower, higher)
    unified.to_file(OUT_UNIFIED, driver="GPKG")

    print(f"wrote {len(unified)} points to {OUT_UNIFIED}")
    print(unified["aoi_section"].value_counts().to_string())
    print("\ntype (no question marks):")
    print(unified["type"].value_counts(dropna=False).to_string())
    print(
        f"\n{unified['flag'].notna().sum()} flagged for a second look; "
        f"{unified['flag'].isna().sum()} clean reads"
    )


if __name__ == "__main__":
    main()
