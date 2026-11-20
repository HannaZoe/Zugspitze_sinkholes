"""Collect the Zugspitzplatt glacier outlines into one GeoPackage (EPSG:32632).

Sources (data/raw/glacier/, see Master_Thesis/docs/glacier_data_sources.md):
- Little Ice Age maximum: Reinthaler & Paul 2025 (Zenodo, CC BY 4.0), the
  Zugspitzplatt glacier LIA_ID 4148. A reconstruction (Hirtlreiter 1992 maps +
  1892 outlines + visible moraines), so check its edge against the moraines.
- Nördlicher / Südlicher Schneeferner outlines from bayerische-gletscher.de
  (BAdW glacier commission, free with attribution). Some files are in
  geographic coordinates; each is read with its own .prj and reprojected. The
  1954 file (unknown CRS) is skipped.

Output: data/processed/glacial/glacier_outlines.gpkg, layer `outlines` with
glacier, year, source, area_ha.
"""

import re

import geopandas as gpd
import pandas as pd

from master_thesis.paths import PROCESSED, RAW

CRS = "EPSG:32632"
SRC = RAW / "glacier"
KFG = "bayerische-gletscher.de (BAdW)"


def main() -> None:
    lia = gpd.read_file(
        f"zip://{SRC / 'Reinthaler_Paul_2025_LIA_Alps' / 'LIA_glacier_outlines_Alps.zip'}"
    )
    lia = lia[lia.LIA_ID == 4148].to_crs(CRS)
    rows = [lia.assign(glacier="Plattachferner (LIA max.)", year=1850,
                       source="Reinthaler & Paul 2025")]  # fmt: skip
    for folder, name in [("nsf_outlines", "Noerdlicher Schneeferner"),
                         ("ssf_outlines", "Suedlicher Schneeferner")]:  # fmt: skip
        for z in sorted((SRC / "bayerische-gletscher_KfG" / folder).glob("*.zip")):
            year = int(re.search(r"(\d{4})", z.stem).group(1))
            if "1954" in z.stem:
                continue  # unknown CRS
            g = gpd.read_file(f"zip://{z}")
            if g.crs is None:
                print(f"  skipped (no CRS): {z.name}")
                continue
            rows.append(g.to_crs(CRS).dissolve().assign(glacier=name, year=year, source=KFG))
    out = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True)[["glacier", "year", "source",
                                                                "geometry"]], crs=CRS)  # fmt: skip
    out["area_ha"] = (out.area / 1e4).round(1)
    path = PROCESSED / "glacial" / "glacier_outlines.gpkg"
    out.to_file(path, layer="outlines", driver="GPKG")
    print(out.drop(columns="geometry").to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
