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
    cover_statistics,
    nlcd,
    nlcd_bycoords,
    nlcd_bygeom,
    ssebopeta_bycoords,
    ssebopeta_bygeom,
    ssebopeta_byloc,
)
from .waterdata import NWIS, WaterQuality, interactive_map

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore[no-redef]

try:
    __version__ = metadata.version("pygeohydro")
except Exception:
    __version__ = "999"

__all__ = [
    "NID",
    "NWIS",
    "WaterQuality",
    "cover_statistics",
    "interactive_map",
    "nlcd",
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
