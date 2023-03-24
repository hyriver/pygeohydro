"""Plot hydrological signatures.

Plots include daily, monthly and annual hydrograph as well as regime
curve (monthly mean) and flow duration curve.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, Union

import folium
import matplotlib.pyplot as plt
import pandas as pd
import pyproj
from matplotlib.colors import BoundaryNorm, ListedColormap

import hydrosignatures as hs
from pygeohydro import helpers
from pygeohydro.exceptions import InputTypeError
from pygeohydro.nwis import NWIS
from pygeoogc import utils as ogc_utils

if TYPE_CHECKING:
    DF = TypeVar("DF", pd.DataFrame, pd.Series)
    CRSTYPE = Union[int, str, pyproj.CRS]

__all__ = ["signatures", "prepare_plot_data"]


class PlotDataType(NamedTuple):
    """Data structure for plotting hydrologic signatures."""

    daily: pd.DataFrame
    mean_monthly: pd.DataFrame
    ranked: pd.DataFrame
    titles: dict[str, str]
    units: dict[str, str]


def prepare_plot_data(daily: pd.DataFrame | pd.Series) -> PlotDataType:
    """Generae a structured data for plotting hydrologic signatures.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed

    Returns
    -------
    PlotDataType
        Containing ``daily, ``mean_monthly``, ``ranked``, ``titles``,
        and ``units`` fields.
    """
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()
    mean_month = hs.compute_mean_monthly(daily, True)
    ranked = hs.compute_exceedance(daily)

    _titles = [
        "Total Hydrograph (daily)",
        "Regime Curve (monthly mean)",
        "Flow Duration Curve",
    ]
    _units = [
        "mm/day",
        "mm/month",
        "mm/day",
    ]
    fields = PlotDataType._fields
    titles = dict(zip(fields[:-1], _titles))
    units = dict(zip(fields[:-1], _units))
    return PlotDataType(daily, mean_month, ranked, titles, units)


def _prepare_plot_data(
    daily: pd.DataFrame | pd.Series,
    precipitation: pd.DataFrame | pd.Series | None = None,
) -> tuple[PlotDataType, PlotDataType | None]:
    if not isinstance(daily, (pd.DataFrame, pd.Series)):
        raise InputTypeError("daily", "pandas.DataFrame or pandas.Series")

    discharge = prepare_plot_data(daily)
    if precipitation is not None:
        if isinstance(precipitation, pd.DataFrame) and precipitation.shape[1] == 1:
            precipitation = precipitation.squeeze()

        if not isinstance(precipitation, pd.Series):
            raise InputTypeError(
                "precipitation", "pandas.Series or pandas.DataFrame with one column"
            )

        prcp = prepare_plot_data(precipitation)
    else:
        prcp = None

    return discharge, prcp


def signatures(
    discharge: pd.DataFrame | pd.Series,
    precipitation: pd.Series | None = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    output: str | Path | None = None,
    close: bool = False,
) -> None:
    """Plot hydrological signatures with w/ or w/o precipitation.

    Plots includes daily hydrograph, regime curve (mean monthly) and
    flow duration curve. The input discharges are converted from cms
    to mm/day based on the watershed area, if provided.

    Parameters
    ----------
    discharge : pd.DataFrame or pd.Series
        The streamflows in mm/day. The column names are used as labels
        on the plot and the column values should be daily streamflow.
    precipitation : pd.Series, optional
        Daily precipitation time series in mm/day. If given, the data is
        plotted on the second x-axis at the top.
    title : str, optional
        The plot supertitle.
    figsize : tuple, optional
        The figure size in inches, defaults to (9, 5).
    output : str, optional
        Path to save the plot as png, defaults to ``None`` which means
        the plot is not saved to a file.
    close : bool, optional
        Whether to close the figure.
    """
    qdaily, prcp = _prepare_plot_data(discharge, precipitation)

    daily = qdaily.daily
    figsize = (9, 5) if figsize is None else figsize
    fig = plt.figure(constrained_layout=True, figsize=figsize, facecolor="w")
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, :])
    ax.grid(False)
    for c, q in daily.items():
        ax.plot(q, label=c)
    lines, labels = ax.get_legend_handles_labels()
    ax.set_ylabel("$Q$ (mm/day)")
    ax.text(0.02, 0.9, "(a)", transform=ax.transAxes, ha="left", va="center", fontweight="bold")

    if prcp is not None:
        _prcp = prcp.daily.squeeze()
        _prcp = _prcp.loc[daily.index[0] : daily.index[-1]]
        label = "$P$ (mm/day)"
        ax_p = ax.twinx()
        ax_p.grid(False)
        if _prcp.shape[0] > 1000:
            ax_p.plot(_prcp, alpha=0.7, color="g", label=label)
        else:
            ax_p.bar(
                _prcp.index,
                _prcp.to_numpy().ravel(),
                alpha=0.7,
                width=1,
                color="g",
                align="edge",
                label=label,
            )
        ax_p.set_ylim(_prcp.max() * 2.5, 0)
        ax_p.set_ylabel(label)
        ax_p.set_xmargin(0)
        lines_p, labels_p = ax_p.get_legend_handles_labels()
        lines.extend(lines_p)
        labels.extend(labels_p)

    ax.set_xmargin(0)
    ax.set_xlabel("")

    ax.legend(
        lines,
        labels,
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        ncol=len(lines),
    )

    ax = fig.add_subplot(gs[1, :-1])
    ax.plot(qdaily.mean_monthly)
    ax.set_xmargin(0)
    ax.grid(False)
    ax.set_ylabel("$Q$ (mm/month)")
    ax.text(0.02, 0.9, "(b)", transform=ax.transAxes, ha="left", va="center", fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    for col in daily:
        dc = qdaily.ranked[[col, f"{col}_rank"]]
        ax.plot(dc[f"{col}_rank"], dc[col], label=col)

    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% Exceedance")
    ax.set_ylabel(rf"$\log(Q)$ ({qdaily.units['ranked']})")
    ax.grid(False)
    ax.text(0.02, 0.9, "(c)", transform=ax.transAxes, ha="left", va="center", fontweight="bold")

    fig.suptitle(title)

    if output is not None:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(output, dpi=300)

    if close:
        plt.close(fig)


def descriptor_legends() -> tuple[ListedColormap, BoundaryNorm, list[int]]:
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    nlcd_meta = helpers.nlcd_helper()
    bounds = [int(v) for v in nlcd_meta["descriptors"]]
    with contextlib.suppress(ValueError):
        bounds.remove(127)

    cmap = ListedColormap(list(nlcd_meta["colors"].values())[: len(bounds)])
    norm = BoundaryNorm(bounds, cmap.N)
    levels = bounds + [30]
    return cmap, norm, levels


def cover_legends() -> tuple[ListedColormap, BoundaryNorm, list[int]]:
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    nlcd_meta = helpers.nlcd_helper()
    bounds = list(nlcd_meta["colors"])
    with contextlib.suppress(ValueError):
        bounds.remove(127)

    cmap = ListedColormap(list(nlcd_meta["colors"].values()))
    norm = BoundaryNorm(bounds, cmap.N)
    levels = bounds + [100]
    return cmap, norm, levels


def interactive_map(
    bbox: tuple[float, float, float, float],
    crs: CRSTYPE = 4326,
    nwis_kwds: dict[str, Any] | None = None,
) -> folium.Map:
    """Generate an interactive map including all USGS stations within a bounding box.

    Parameters
    ----------
    bbox : tuple
        List of corners in this order (west, south, east, north)
    crs : str, int, or pyproj.CRS, optional
        CRS of the input bounding box, defaults to EPSG:4326.
    nwis_kwds : dict, optional
        Optional keywords to include in the NWIS request as a dictionary like so:
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
    nwis = NWIS()
    bbox = ogc_utils.match_crs(bbox, crs, 4326)
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

    sites = sites.groupby("site_no")[cols_old[1:]].agg(set).reset_index()
    sites = sites.rename(columns=dict(zip(cols_old, cols_new)))

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

    imap = folium.Map(
        location=(lat, lon),
        tiles="Stamen Terrain",
        zoom_start=10,
    )

    for coords, msg in sites[["Coordinate", "msg"]].itertuples(name=None, index=False):
        folium.Marker(
            location=list(coords)[0][::-1],
            popup=folium.Popup(msg, max_width=250),
            icon=folium.Icon(),
        ).add_to(imap)

    return imap
