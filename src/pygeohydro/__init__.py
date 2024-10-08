"""Top-level package for PyGeoHydro."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pygeohydro import exceptions, helpers, plot
from pygeohydro.helpers import get_us_states
from pygeohydro.levee import NLD
from pygeohydro.nfhl import NFHL
from pygeohydro.nid import NID
from pygeohydro.nlcd import (
    cover_statistics,
    nlcd_area_percent,
    nlcd_bycoords,
    nlcd_bygeom,
    overland_roughness,
)
from pygeohydro.nwis import NWIS, streamflow_fillna
from pygeohydro.plot import interactive_map
from pygeohydro.print_versions import show_versions
from pygeohydro.pygeohydro import (
    EHydro,
    get_camels,
    soil_gnatsgo,
    soil_properties,
    soil_soilgrids,
    ssebopeta_bycoords,
    ssebopeta_bygeom,
)
from pygeohydro.stnfloodevents import STNFloodEventData, stn_flood_event
from pygeohydro.waterdata import SensorThings, WaterQuality
from pygeohydro.watershed import WBD, huc_wb_full, irrigation_withdrawals

try:
    __version__ = version("pygeohydro")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "EHydro",
    "NID",
    "WBD",
    "NWIS",
    "NFHL",
    "NLD",
    "WaterQuality",
    "streamflow_fillna",
    "cover_statistics",
    "get_camels",
    "overland_roughness",
    "huc_wb_full",
    "irrigation_withdrawals",
    "SensorThings",
    "STNFloodEventData",
    "stn_flood_event",
    "interactive_map",
    "nlcd_bygeom",
    "nlcd_bycoords",
    "nlcd_area_percent",
    "ssebopeta_bygeom",
    "ssebopeta_bycoords",
    "soil_properties",
    "soil_gnatsgo",
    "soil_soilgrids",
    "helpers",
    "get_us_states",
    "plot",
    "show_versions",
    "exceptions",
    "__version__",
]
