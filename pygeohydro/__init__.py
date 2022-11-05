"""Top-level package for PyGeoHydro."""
from importlib.metadata import PackageNotFoundError, version

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
from .print_versions import show_versions
from .pygeohydro import (
    NID,
    WBD,
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
from .waterdata import NWIS, WaterQuality, interactive_map

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
    "interactive_map",
    "nlcd_bygeom",
    "nlcd_bycoords",
    "ssebopeta_bygeom",
    "ssebopeta_byloc",
    "ssebopeta_bycoords",
    "soil_properties",
    "soil_gnatsgo",
    "helpers",
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
