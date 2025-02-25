"""Top-level package for PyGeoHydro."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# for now export the plotting functions from hydrosignatures
# since they have been moved from pygeohydro.plot to hydrosignatures
from hydrosignatures import plot
from pygeohydro import exceptions, helpers
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
from pygeohydro.plot import cover_legends, descriptor_legends, interactive_map
from pygeohydro.print_versions import show_versions
from pygeohydro.pygeohydro import (
    EHydro,
    get_camels,
    soil_gnatsgo,
    soil_polaris,
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
    "NFHL",
    "NID",
    "NLD",
    "NWIS",
    "WBD",
    "EHydro",
    "STNFloodEventData",
    "SensorThings",
    "WaterQuality",
    "__version__",
    "cover_legends",
    "cover_statistics",
    "descriptor_legends",
    "exceptions",
    "get_camels",
    "get_us_states",
    "helpers",
    "huc_wb_full",
    "interactive_map",
    "irrigation_withdrawals",
    "nlcd_area_percent",
    "nlcd_bycoords",
    "nlcd_bygeom",
    "overland_roughness",
    "plot",
    "show_versions",
    "soil_gnatsgo",
    "soil_polaris",
    "soil_properties",
    "soil_soilgrids",
    "ssebopeta_bycoords",
    "ssebopeta_bygeom",
    "stn_flood_event",
    "streamflow_fillna",
]
