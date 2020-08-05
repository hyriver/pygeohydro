from pkg_resources import DistributionNotFound, get_distribution

from . import helpers, plot
from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue
from .hydrodata import (
    NWIS,
    cover_statistics,
    interactive_map,
    nlcd,
    ssebopeta_bygeom,
    ssebopeta_byloc,
)
from .print_versions import show_versions

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
