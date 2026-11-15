"""Terrain derivative rasters (slope, aspect, curvature, roughness, wetness) from a DEM.

Thin wrappers around WhiteboxTools. Kept apart from doline_detection.py, which
handles depression extraction; these rasters are covariates for the
susceptibility model.
"""

from pathlib import Path

from whitebox import WhiteboxTools


def _wbt(dem_path: Path) -> WhiteboxTools:
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(str(dem_path.parent.resolve()))
    return wbt


def compute_slope(dem_path: Path, output_path: Path) -> None:
    """Slope in degrees."""
    _wbt(dem_path).slope(str(dem_path.resolve()), str(output_path.resolve()), units="degrees")


def compute_aspect(dem_path: Path, output_path: Path) -> None:
    """Aspect in degrees (0-360, downslope direction)."""
    _wbt(dem_path).aspect(str(dem_path.resolve()), str(output_path.resolve()))


def compute_plan_curvature(dem_path: Path, output_path: Path) -> None:
    """Curvature perpendicular to slope direction -- convergence/divergence of flow."""
    _wbt(dem_path).plan_curvature(str(dem_path.resolve()), str(output_path.resolve()))


def compute_profile_curvature(dem_path: Path, output_path: Path) -> None:
    """Curvature parallel to slope direction -- acceleration/deceleration of flow."""
    _wbt(dem_path).profile_curvature(str(dem_path.resolve()), str(output_path.resolve()))


def compute_roughness(dem_path: Path, output_path: Path) -> None:
    """Ruggedness index: mean elevation difference to the 8 neighbouring cells."""
    _wbt(dem_path).ruggedness_index(str(dem_path.resolve()), str(output_path.resolve()))


def compute_wetness_index(dem_path: Path, output_dir: Path, output_path: Path) -> None:
    """Topographic wetness index (TWI = ln(SCA / tan(slope))).

    Flow routing runs on a depression-filled DEM. The filled DEM, the specific
    catchment area and a slope raster in radians (the unit wetness_index
    expects, unlike ``compute_slope``) are written to ``output_dir`` as
    temporary files and removed afterwards.
    """
    wbt = _wbt(dem_path)
    filled = output_dir / "_scratch_filled.tif"
    sca = output_dir / "_scratch_sca.tif"
    slope_rad = output_dir / "_scratch_slope_rad.tif"

    wbt.fill_depressions(str(dem_path.resolve()), str(filled.resolve()))
    wbt.d8_flow_accumulation(
        str(filled.resolve()), str(sca.resolve()), out_type="specific contributing area"
    )
    wbt.slope(str(filled.resolve()), str(slope_rad.resolve()), units="radians")
    wbt.wetness_index(str(sca.resolve()), str(slope_rad.resolve()), str(output_path.resolve()))

    filled.unlink(missing_ok=True)
    sca.unlink(missing_ok=True)
    slope_rad.unlink(missing_ok=True)
