from pkg_resources import DistributionNotFound, get_distribution

from . import helpers, plot
from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue, ZeroMatched
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
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
