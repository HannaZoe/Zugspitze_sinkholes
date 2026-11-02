# Zugspitze_sinkholes

Code for an M.Sc. thesis on collapse dolines on the Zugspitzplatt. **All
thesis code lives here** — notebooks, scripts and the `master_thesis`
package. Academic context, notes, the experiment log, data and fieldwork
material live in the sibling, private `Master_Thesis` repo; read its
`CLAUDE.md` for the research background and log conventions. This repo is
public, so nothing personal, no data and no unpublished field material goes
in here.

## Two repos, side by side

```
Master/
├── Master_Thesis/         data/, fieldwork/, outputs/, docs/ (incl. experiment_log.md)
└── Zugspitze_sinkholes/   this repo
```

Code never hard-codes `data/...` paths: import the locations from
`master_thesis.paths` (`RAW`, `INTERIM`, `PROCESSED`, `MANUAL`, `FIELD_DATA`,
`FIGURES`, `MAPS`). They resolve to `../Master_Thesis/...`; set the
`THESIS_ROOT` environment variable if that repo lives somewhere else.

## Working conventions

- Write code like a senior EO researcher: concise, correct, readable — not
  verbose, not over-engineered. Reach for well-maintained packages
  (`geopandas`, `rasterio`, `laspy`, `open3d`, `whitebox`, `scipy`,
  `scikit-learn`, ...) over hand-rolling something they already do well.
- Run `uv run ruff check --fix` and `uv run ruff format` before any commit —
  lint-clean is a precondition for committing, not a follow-up step.
- No AI attribution in git history — plain commit messages, no
  "Co-Authored-By: Claude" / "Generated with Claude Code" trailers.
- Commit and push after every meaningful unit of work, without being asked.
  Don't let more than one logical change pile up uncommitted.
- Analysis attempts get logged in `../Master_Thesis/docs/experiment_log.md`
  (see that repo's `CLAUDE.md`), not here.
- Comments: minimal, but a little lighthearted is fine — this doesn't need
  to read like enterprise code. No emojis anywhere in code.
- The doline detection code is meant to converge into one canonical script
  that represents the current best-known method (not a growing pile of
  per-date notebooks). Notebooks stay useful for the actual exploration/
  calibration work; once a method is validated, fold it into the script and
  keep iterating on that one place going forward.
- Coordinate reference systems: state the CRS explicitly in every script/
  notebook that reads or writes spatial data. The UAV DSMs are natively
  **EPSG:32632** (WGS84/UTM32N), the project's default working CRS. That is
  *not* the same datum as EPSG:25832 (ETRS89/UTM32N, Bavarian DGM1 and
  historical DOPs) — reproject explicitly when combining the two. Agisoft
  multispectral exports come out in EPSG:4326 and need reprojecting.

## Folder map

```
Zugspitze_sinkholes/
├── src/master_thesis/   installable package — see src/CLAUDE.md
├── notebooks/           exploratory analysis and calibration (Jupyter)
└── scripts/             run-this-to-reproduce scripts (downloads, imports,
                         and eventually the canonical detection pipeline)
```

Rule of thumb: once a function is needed in a second notebook or script, it
moves into `src/master_thesis/`.

## Stack

- **Python**: managed with `uv`, not conda, pinned to 3.12 (`.python-version`;
  `open3d` has no 3.13 wheels). Deliberate choice — see "Why uv, not conda"
  below before suggesting a switch.
- **Editor**: VS Code.
- **GIS**: QGIS (standalone install) for visualization, cartography, and manual
  vector/raster inspection. Analysis and processing live in Python scripts/
  notebooks, not QGIS's own Python console, so work stays scriptable and
  reproducible.
- **Point clouds**: `laspy` (+ `lazrs` for LAZ compression) for I/O, `open3d`
  for registration/filtering/visualization, `whitebox` for terrain/raster
  derivatives. No PDAL — see below.
- **Raster/vector**: `rasterio`, `geopandas`, `shapely`, `pyproj`, `fiona`.
- **Linting**: `ruff` (check + format), dev dependency — see Working conventions.

### Why uv, not conda

Geospatial packages (GDAL, PDAL) are notoriously hard to install via plain pip
on Windows because they wrap C++ libraries without official wheels. Conda-forge
normally solves this, but the explicit choice here is `uv`. Consequences:

- **PDAL is intentionally excluded** from dependencies — no reliable pip/uv
  install path on Windows. If a task genuinely needs PDAL (e.g. a specific
  filter only PDAL implements), the options are: (a) find an `open3d`/`laspy`/
  `whitebox` equivalent first, (b) call the PDAL CLI via subprocess if it's
  installed separately (e.g. bundled with QGIS/OSGeo4W), or (c) flag it to the
  user rather than silently trying to pip-install it.
- `rasterio`, `fiona`, `pyproj`, `shapely` all ship self-contained wheels on
  Windows, so they're safe to use freely.
- If GDAL-level functionality is needed beyond what `rasterio`/`fiona` expose,
  prefer calling QGIS's bundled GDAL/Python via subprocess over trying to pip
  install `gdal` directly.

## Environment setup

```powershell
uv sync              # creates .venv and installs locked dependencies
uv add <package>      # add a new dependency (updates pyproject.toml + lockfile)
uv run python ...     # run a script inside the project environment
uv run jupyter lab     # launch JupyterLab
uv run ruff check --fix && uv run ruff format   # lint + format before committing
```

Point VS Code's Python interpreter (and the notebook kernel) at
`.venv\Scripts\python.exe` in this folder.
