# Zugspitze_sinkholes

Code companion to my M.Sc. thesis on sinkholes (dolines) on the Zugspitzplatt, Bavarian Alps.

This repo holds only the code. No data is included.

Claude is used to support the author in generating code.

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
