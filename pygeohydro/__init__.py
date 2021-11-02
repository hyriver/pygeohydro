from . import helpers, plot
from .exceptions import (
    DataNotAvailable,
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    ZeroMatched,
)
from .print_versions import show_versions
from .pygeohydro import (
    NID,
    NWIS,
    cover_statistics,
    interactive_map,
    nlcd,
    ssebopeta_bygeom,
    ssebopeta_byloc,
)

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
    "cover_statistics",
    "interactive_map",
    "nlcd",
    "ssebopeta_bygeom",
    "ssebopeta_byloc",
    "helpers",
    "plot",
    "DataNotAvailable",
    "InvalidInputRange",
    "InvalidInputType",
    "InvalidInputValue",
    "ZeroMatched",
    "show_versions",
    "__version__",
]
