"""Accessing National Flood Hazard Layers (NLD) through web services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pygeohydro.exceptions import InputValueError
from pynhd import AGRBase

if TYPE_CHECKING:
    from pyproj import CRS

    CRSType = int | str | CRS

__all__ = ["NLD"]


class NLD(AGRBase):
    """Access National Levee Database (NLD) services.

    Notes
    -----
    For more info visit: https://geospatial.sec.usace.army.mil/server/rest/services/NLD2_PUBLIC/FeatureServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Valid layers are:

        - ``boreholes``
        - ``crossings``
        - ``levee_stations``
        - ``piezometers``
        - ``pump_stations``
        - ``relief_wells``
        - ``alignment_lines``
        - ``closure_structures``
        - ``cross_sections``
        - ``embankments``
        - ``floodwalls``
        - ``frm_lines``
        - ``pipe_gates``
        - ``toe_drains``
        - ``leveed_areas``
        - ``system_routes``
        - ``pipes``
        - ``channels``

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.

    Examples
    --------
    >>> from pygeohydro import NLD
    >>> nld = NLD("levee_stations")
    >>> levees = nld.bygeom((-105.914551, 37.437388, -105.807434, 37.522392))
    >>> levees.shape
    (1838, 12)
    """

    def __init__(
        self,
        layer: Literal[
            "boreholes",
            "crossings",
            "levee_stations",
            "piezometers",
            "pump_stations",
            "relief_wells",
            "alignment_lines",
            "closure_structures",
            "cross_sections",
            "embankments",
            "floodwalls",
            "frm_lines",
            "pipe_gates",
            "toe_drains",
            "leveed_areas",
            "system_routes",
            "pipes",
            "channels",
        ],
        outfields: str | list[str] = "*",
        crs: CRSType = 4326,
    ):
        self.valid_layers = {
            "boreholes": "0",
            "crossings": "1",
            "levee_stations": "2",
            "piezometers": "3",
            "pump_stations": "4",
            "relief_wells": "5",
            "alignment_lines": "6",
            "closure_structures": "7",
            "cross_sections": "8",
            "embankments": "9",
            "floodwalls": "10",
            "frm_lines": "11",
            "pipe_gates": "12",
            "toe_drains": "13",
            "leveed_areas": "14",
            "system_routes": "15",
            "pipes": "16",
            "channels": "17",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        base_url = (
            "https://geospatial.sec.usace.army.mil/server/rest/services/NLD2_PUBLIC/FeatureServer"
        )
        super().__init__(
            f"{base_url}/{_layer}",
            None,
            outfields,
            crs,
        )
