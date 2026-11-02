# Zugspitze_sinkholes

Code companion to Hanna Schulten's M.Sc. thesis (EAGLE — Applied Earth
Observation and Geoanalysis, University of Würzburg): collapse dolines
(sinkholes) on the Zugspitzplatt, Bavarian Alps — surface change detection,
automated/semi-automated detection methods, and geohazard characterization
from UAV and airborne remote sensing data.

This repo holds only the code. No data is included — UAV photogrammetry/
LiDAR deliverables are third-party field data not cleared for redistribution,
and manually digitized ground-truth is kept in the private working repo.
Scripts that fetch public data (e.g. the historical Bavarian orthophoto
archive) are included and runnable on their own.

## Structure

- `src/master_thesis/` — installable package: terrain covariates, doline
  detection logic.
- `scripts/` — standalone, run-this-to-reproduce pipeline scripts (e.g.
  downloading the public historical orthophoto archive).
- `notebooks/` — exploratory analysis and method calibration.

## Stack

Python, managed with [uv](https://docs.astral.sh/uv/). Key dependencies:
`geopandas`, `rasterio`, `laspy`, `open3d`, `whitebox`, `scikit-learn` — see
`pyproject.toml` for the full list.

```bash
uv sync
uv run jupyter lab
```
