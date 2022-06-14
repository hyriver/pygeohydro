"""Top-level package for PyGeoHydro."""
import importlib.metadata

from . import helpers, plot
from .exceptions import (
    DataNotAvailable,
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingColumns,
    MissingCRS,
    ZeroMatched,
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
    ssebopeta_bycoords,
    ssebopeta_bygeom,
    ssebopeta_byloc,
)
from .waterdata import NWIS, WaterQuality, interactive_map

__version__ = importlib.metadata.version("pygeohydro")

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
    "helpers",
    "plot",
    "DataNotAvailable",
    "InvalidInputRange",
    "InvalidInputType",
    "MissingCRS",
    "MissingColumns",
    "InvalidInputValue",
    "ZeroMatched",
    "show_versions",
    "__version__",
]
