"""Top-level package for PyGeoHydro."""
from importlib.metadata import PackageNotFoundError, version

if int(version("shapely").split(".")[0]) > 1:
    import os

    os.environ["USE_PYGEOS"] = "0"

from . import helpers, plot
from .exceptions import (
    DataNotAvailableError,
    DependencyError,
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingColumnError,
    MissingCRSError,
    ZeroMatchedError,
)
from .helpers import get_us_states
from .plot import interactive_map
from .print_versions import show_versions
from .pygeohydro import (
    NID,
    cover_statistics,
    get_camels,
    nlcd_bycoords,
    nlcd_bygeom,
    overland_roughness,
    soil_gnatsgo,
    soil_properties,
    ssebopeta_bycoords,
    ssebopeta_bygeom,
)
from .waterdata import NWIS, WBD, WaterQuality, huc_wb_full, irrigation_withdrawals

try:
    __version__ = version("pygeohydro")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "NID",
    "WBD",
    "NWIS",
    "WaterQuality",
    "cover_statistics",
    "get_camels",
    "overland_roughness",
    "huc_wb_full",
    "irrigation_withdrawals",
    "interactive_map",
    "nlcd_bygeom",
    "nlcd_bycoords",
    "ssebopeta_bygeom",
    "ssebopeta_bycoords",
    "soil_properties",
    "soil_gnatsgo",
    "helpers",
    "get_us_states",
    "plot",
    "DataNotAvailableError",
    "InputRangeError",
    "InputTypeError",
    "MissingCRSError",
    "MissingColumnError",
    "DependencyError",
    "InputValueError",
    "ZeroMatchedError",
    "show_versions",
    "__version__",
]
