"""Where data and outputs live.

This repo holds code only. Data, manual ground truth, field data and outputs
stay in the sibling Master_Thesis repo, so check both out side by side:

    Master/
        Master_Thesis/         data/, fieldwork/, outputs/
        Zugspitze_sinkholes/   this repo

Set the THESIS_ROOT environment variable if Master_Thesis lives elsewhere.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = Path(os.environ.get("THESIS_ROOT", REPO_ROOT.parent / "Master_Thesis"))

DATA = THESIS_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANUAL = DATA / "manual"
FIELD_DATA = THESIS_ROOT / "fieldwork" / "field_data"
FIGURES = THESIS_ROOT / "outputs" / "figures"
MAPS = THESIS_ROOT / "outputs" / "maps"
