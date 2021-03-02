from pkg_resources import DistributionNotFound, get_distribution

from . import helpers, plot
from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue
from .print_versions import show_versions
from .pygeohydro import (
    NWIS,
    cover_statistics,
    get_nid,
    get_nid_codes,
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
