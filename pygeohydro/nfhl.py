"""Accessing National Flood Hazard Layers (NFHL) through web services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from pygeohydro.exceptions import InputValueError
from pygeoogc import ServiceURL
from pynhd import AGRBase

if TYPE_CHECKING:
    import pyproj

    CRSTYPE = Union[int, str, pyproj.CRS]

__all__ = ["NFHL"]


class NFHL(AGRBase):
    """Access National Flood Hazard Layers (NFHL).

    Parameters
    ----------
    service : str
        The service type. Valid services are:

        - ``NFHL``: Effective National Flood Hazard Layers
        - ``Prelim_CSLF``: Preliminary Changes Since Last Firm (CSLF)
        - ``Draft_CSLF``: Draft Changes Since Last Firm (CSLF)
        - ``Prelim_NFHL``: Preliminary National Flood Hazard Layers
        - ``Pending_NFHL``: Pending National Flood Hazard Layers
        - ``Draft_NFHL``: Draft National Flood Hazard Layers

    layer : str
        A valid service layer. Valid layers are service specific:

        - ``NFHL``: ``nfhl availability``, ``firm panels``, ``lomrs``, ``lomas``,
            ``political jurisdictions``, ``profile baselines``, ``water lines``,
            ``cross-sections``, ``base flood elevations``, ``levees``,
            ``seclusion boundaries``, ``coastal transects``, ``transect baselines``,
            ``general structures``, ``river mile markers``, ``water areas``, ``plss``,
            ``limit of moderate wave action``, ``flood hazard boundaries``,
            ``flood hazard zones``, ``primary frontal dunes``, ``base index``,
            ``topographic low confidence areas``, ``datum conversion points``,
            ``coastal gages``, ``gages``, ``nodes``, ``high water marks``,
            ``station start points``, ``hydrologic reaches``, ``alluvial fans``,
            and ``subbasins``
        - ``Prelim_CSLF``: ``preliminary``, ``coastal high hazard area change``,
            ``floodway change``, ``special flood hazard area change``,
            and ``non-special flood hazard area change``
        - ``Draft_CSLF``: ``draft``, ``coastal high hazard area change``,
            ``floodway change``, ``special flood hazard area change``, and
            ``non-special flood hazard area change``
        - ``Prelim_NFHL``: ``preliminary data availability``,
            ``preliminary firm panel index``, ``preliminary plss``,
            ``preliminary topographic low confidence areas``,
            ``preliminary river mile markers``, ``preliminary datum conversion points``,
            ``preliminary coastal gages``, ``preliminary gages``, ``preliminary nodes``,
            ``preliminary high water marks``, ``preliminary station start points``,
            ``preliminary cross-sections``, ``preliminary coastal transects``,
            ``preliminary base flood elevations``, ``preliminary profile baselines``,
            ``preliminary transect baselines``, ``preliminary limit of moderate wave action``,
            ``preliminary water lines``, ``preliminary political jurisdictions``,
            ``preliminary levees``, ``preliminary general structures``,
            ``preliminary primary frontal dunes``, ``preliminary hydrologic reaches``,
            ``preliminary flood hazard boundaries``, ``preliminary flood hazard zones``,
            ``preliminary submittal information``, ``preliminary alluvial fans``,
            ``preliminary subbasins``, and ``preliminary water areas``
        - ``Pending_NFHL``: ``pending submittal information``, ``pending water areas``,
            ``pending firm panel index``, ``pending data availability``,
            ``pending firm panels``, ``pending political jurisdictions``,
            ``pending profile baselines``, ``pending water lines``,
            ``pending cross-sections``, ``pending base flood elevations``,
            ``pending levees``, ``pending seclusion boundaries``,
            ``pending coastal transects``, ``pending transect baselines``,
            ``pending general structures``, ``pending river mile markers``,
            ``pending plss``, ``pending limit of moderate wave action``,
            ``pending flood hazard boundaries``, ``pending flood hazard zones``,
            ``pending primary frontal dunes``, ``pending topographic low confidence areas``,
            ``pending datum conversion points``, ``pending coastal gages``,
            ``pending gages``, ``pending nodes``, ``pending high water marks``,
            ``pending station start points``, ``pending hydrologic reaches``,
            ``pending alluvial fans``, and ``pending subbasins``
        - ``Draft_NFHL``: ``draft data availability``, ``draft firm panels``,
            ``draft political jurisdictions``, ``draft profile baselines``,
            ``draft water lines``, ``draft cross-sections``, ``draft base flood elevations``,
            ``draft levees``, ``draft submittal info``, ``draft coastal transects``,
            ``draft transect baselines``, ``draft general structures``,
            ``draft limit of moderate wave action``, ``draft flood hazard boundaries``,
            and ``draft flood hazard zones``

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference of output, default to ``EPSG:4326``.

    Examples
    --------
    >>> from pygeohydro import NFHL
    >>> nfhl = NFHL("NFHL", "cross-sections")
    >>> gdf_xs = nfhl.bygeom((-73.42, 43.28, -72.9, 43.52), geo_crs=4269)

    References
    ----------
    * `National Flood Hazard Layer <https://hazards.fema.gov/femaportal/wps/portal/NFHLWMS>`__

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.
    """

    def __init__(
        self, service: str, layer: str, outfields: str | list[str] = "*", crs: CRSTYPE = 4326
    ):
        # service URLs
        self.__valid_services = {
            "NFHL": ServiceURL().restful.fema_nfhl,
            "Prelim_CSLF": ServiceURL().restful.fema_prelim_cslf,
            "Draft_CSLF": ServiceURL().restful.fema_draft_cslf,
            "Prelim_NFHL": ServiceURL().restful.fema_prelim_nfhl,
            "Pending_NFHL": ServiceURL().restful.fema_pending_nfhl,
            "Draft_NFHL": ServiceURL().restful.fema_draft_nfhl,
        }
        url = self.valid_services.get(service)
        if url is None:
            raise InputValueError("service", list(self.valid_services))

        super().__init__(url, layer, outfields, crs)

    @property
    def valid_services(self) -> dict[str, str]:
        """A dictionary of valid services and their URLs."""
        return self.__valid_services
