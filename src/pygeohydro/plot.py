"""Plot hydrological signatures.

Plots include daily, monthly and annual hydrograph as well as regime
curve (monthly mean) and flow duration curve.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, TypeVar

import pygeoutils as geoutils
from pygeohydro import helpers
from pygeohydro.nwis import NWIS

if TYPE_CHECKING:
    import pandas as pd
    from folium import Map
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from pyproj import CRS

    CRSType = int | str | CRS
    DF = TypeVar("DF", pd.DataFrame, pd.Series)

__all__ = ["cover_legends", "descriptor_legends", "interactive_map"]


def descriptor_legends() -> tuple[ListedColormap, BoundaryNorm, list[int]]:
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    try:
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for descriptor legends. Please install it."
        ) from e
    nlcd_meta = helpers.nlcd_helper()
    bounds = [int(v) for v in nlcd_meta["descriptors"]]
    with contextlib.suppress(ValueError):
        bounds.remove(127)

    cmap = ListedColormap(list(nlcd_meta["colors"].values())[: len(bounds)])
    norm = BoundaryNorm(bounds, cmap.N)
    levels = [*bounds, 30]
    return cmap, norm, levels


def cover_legends() -> tuple[ListedColormap, BoundaryNorm, list[int]]:
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    try:
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for descriptor legends. Please install it."
        ) from e
    nlcd_meta = helpers.nlcd_helper()
    bounds = list(nlcd_meta["colors"])
    with contextlib.suppress(ValueError):
        bounds.remove(127)

    cmap = ListedColormap(list(nlcd_meta["colors"].values()))
    norm = BoundaryNorm(bounds, cmap.N)
    levels = [*bounds, 100]
    return cmap, norm, levels


def interactive_map(
    bbox: tuple[float, float, float, float],
    crs: CRSType = 4326,
    nwis_kwds: dict[str, Any] | None = None,
) -> Map:
    """Generate an interactive map including all USGS stations within a bounding box.

    Parameters
    ----------
    bbox : tuple
        List of corners in this order (west, south, east, north)
    crs : str, int, or pyproj.CRS, optional
        CRS of the input bounding box, defaults to EPSG:4326.
    nwis_kwds : dict, optional
        Additional keywords to include in the NWIS request as a dictionary like so:
        ``{"hasDataTypeCd": "dv,iv", "outputDataTypeCd": "dv,iv", "parameterCd": "06000"}``.
        Default to None.

    Returns
    -------
    folium.Map
        Interactive map within a bounding box.

    Examples
    --------
    >>> import pygeohydro as gh
    >>> nwis_kwds = {"hasDataTypeCd": "dv,iv", "outputDataTypeCd": "dv,iv"}
    >>> m = gh.interactive_map((-69.77, 45.07, -69.31, 45.45), nwis_kwds=nwis_kwds)
    >>> n_stations = len(m.to_dict()["children"]) - 1
    >>> n_stations
    10
    """
    try:
        import folium
    except ImportError as e:
        raise ImportError("folium is required for interactive map. Please install it.") from e
    nwis = NWIS()
    bbox = geoutils.geometry_reproject(bbox, crs, 4326)
    query = {"bBox": ",".join(f"{b:.06f}" for b in bbox)}
    if isinstance(nwis_kwds, dict):
        query.update(nwis_kwds)

    sites = nwis.get_info(query, expanded=True, nhd_info=True)

    sites["coords"] = list(sites[["dec_long_va", "dec_lat_va"]].itertuples(name=None, index=False))
    sites["altitude"] = (
        sites["alt_va"].astype("str") + " ft above " + sites["alt_datum_cd"].astype("str")
    )

    sites["drain_area_va"] = sites["drain_area_va"].astype("str") + " sqmi"
    sites["contrib_drain_area_va"] = sites["contrib_drain_area_va"].astype("str") + " sqmi"
    sites["nhd_areasqkm"] = sites["nhd_areasqkm"].astype("str") + " sqkm"
    for c in ("drain_area_va", "contrib_drain_area_va", "nhd_areasqkm"):
        sites.loc[sites[c].str.contains("nan"), c] = "N/A"

    cols_old = [
        "site_no",
        "station_nm",
        "coords",
        "altitude",
        "huc_cd",
        "drain_area_va",
        "contrib_drain_area_va",
        "nhd_areasqkm",
        "hcdn_2009",
    ]

    cols_new = [
        "Site No.",
        "Station Name",
        "Coordinate",
        "Altitude",
        "HUC8",
        "Drainage Area (NWIS)",
        "Contributing Drainage Area (NWIS)",
        "Drainage Area (GagesII)",
        "HCDN 2009",
    ]

    sites = (
        sites.groupby("site_no")[cols_old[1:]]
        .agg(set)
        .reset_index()
        .rename(columns=dict(zip(cols_old, cols_new)))
    )

    msgs = []
    base_url = "https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no="
    for row in sites.itertuples(index=False):
        site_no = row[sites.columns.get_loc(cols_new[0])]
        msg = f"<strong>{cols_new[0]}</strong>: {site_no}<br>"
        for col in cols_new[1:]:
            value = ", ".join(str(s) for s in row[sites.columns.get_loc(col)])
            msg += f"<strong>{col}</strong>: {value}<br>"
        msg += f'<a href="{base_url}{site_no}" target="_blank">More on USGS Website</a>'
        msgs.append(msg[:-4])

    sites["msg"] = msgs

    west, south, east, north = bbox
    lon = (west + east) * 0.5
    lat = (south + north) * 0.5

    imap = folium.Map(location=(lat, lon), zoom_start=10)

    for coords, msg in sites[["Coordinate", "msg"]].itertuples(name=None, index=False):
        folium.Marker(
            location=next(iter(coords))[::-1],
            popup=folium.Popup(msg, max_width=250),  # pyright: ignore[reportGeneralTypeIssues]
            icon=folium.Icon(),
        ).add_to(imap)

    return imap
