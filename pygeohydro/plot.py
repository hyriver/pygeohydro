"""Plot hydrological signatures.

Plots includes  daily, monthly and annual hydrograph as well as regime
curve (monthly mean) and flow duration curve.
"""
import calendar
import contextlib
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union

import pandas as pd
import proplot as pplt
from matplotlib.colors import BoundaryNorm, ListedColormap

from . import helpers
from .exceptions import InputTypeError

__all__ = ["signatures", "prepare_plot_data", "exceedance", "mean_monthly"]
DF = TypeVar("DF", pd.DataFrame, pd.Series)
pplt.rc["figure.facecolor"] = "w"


class PlotDataType(NamedTuple):
    """Data structure for plotting hydrologic signatures."""

    daily: pd.DataFrame
    mean_monthly: pd.DataFrame
    ranked: pd.DataFrame
    titles: Dict[str, str]
    units: Dict[str, str]


def exceedance(daily: DF, threshold: float = 1e-3) -> DF:
    """Compute exceedance probability from daily data.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed
    threshold : float, optional
        The threshold to compute exceedance probability, defaults to 1e-3.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Exceedance probability.
    """
    _daily = daily[daily > threshold].copy()
    ranks = _daily.rank(ascending=False, pct=True) * 100
    fdc = [
        pd.DataFrame({c: _daily[c], f"{c}_rank": ranks[c]})
        .sort_values(by=f"{c}_rank")
        .reset_index(drop=True)
        for c in daily
    ]
    return pd.concat(fdc, axis=1)


def mean_monthly(daily: DF, index_abbr: bool = False) -> DF:
    """Compute mean monthly summary from daily data.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Mean monthly summary.
    """
    monthly = daily.groupby(pd.Grouper(freq="M")).sum()
    mean_month = monthly.groupby(monthly.index.month).mean()
    mean_month.index.name = "Month"
    if index_abbr:
        month_abbr = dict(enumerate(calendar.month_abbr))
        mean_month.index = mean_month.index.map(month_abbr)
    return mean_month


def prepare_plot_data(daily: Union[pd.DataFrame, pd.Series]) -> PlotDataType:
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
    mean_month = mean_monthly(daily, True)
    ranked = exceedance(daily)

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
    daily: Union[pd.DataFrame, pd.Series],
    precipitation: Optional[Union[pd.DataFrame, pd.Series]] = None,
) -> Tuple[PlotDataType, Optional[PlotDataType]]:
    if not isinstance(daily, (pd.DataFrame, pd.Series)):
        raise InputTypeError("daily", "pd.DataFrame or pd.Series")

    discharge = prepare_plot_data(daily)

    if not isinstance(precipitation, (pd.DataFrame, pd.Series)) and precipitation is not None:
        raise InputTypeError("precipitation", "pd.DataFrame or pd.Series")

    prcp = None if precipitation is None else prepare_plot_data(precipitation)

    return discharge, prcp


def signatures(
    discharge: Union[pd.DataFrame, pd.Series],
    precipitation: Optional[pd.Series] = None,
    title: Optional[str] = None,
    threshold: float = 1e-3,
    output: Union[str, Path, None] = None,
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
    threshold : float, optional
        The threshold for cutting off the discharge for the flow duration
        curve to deal with log 0 issue, defaults to :math:`1^{-3}` mm/day.
    output : str, optional
        Path to save the plot as png, defaults to ``None`` which means
        the plot is not saved to a file.
    close : bool, optional
        Whether to close the figure.
    """
    qdaily, prcp = _prepare_plot_data(discharge, precipitation)

    cycle = "colorblind10"
    fig, axs = pplt.subplots(
        [[1, 1, 1, 1], [2, 2, 3, 3]],
        refwidth=6.5,
        refheight=2.2,
        share=0,
        facecolor="w",
    )
    axs.format(titleloc="uc", toplabels=(title,), lw=0.7)

    daily = qdaily.daily
    fdc = qdaily.ranked
    rc = qdaily.mean_monthly

    hs = axs[0, 0].plot(daily, cycle=cycle)
    axs[0, 0].format(xlabel="", ylabel=f"$Q$ ({qdaily.units['daily']})", xrotation=0)
    axs[0, 0].margins(x=0)

    if len(hs) > 1:
        fig.legend(hs, ncols=1, frame=False, loc="r")

    axs[1, 0].plot(rc, cycle=cycle)
    axs[1, 0].format(xlabel="", ylabel=f"$Q$ ({qdaily.units['mean_monthly']})")

    if prcp is not None:
        ox = axs[0, 0].alty(reverse=True, label=f"$P$ ({prcp.units['daily']})")
        ox.bar(prcp.daily, color="g", alpha=0.2, width=1)
        ox.margins(x=0)

        ox = axs[1, 0].alty(
            reverse=True,
            lim=(0, prcp.mean_monthly.max()[0] * 2.5),
            label=f"$P$ ({prcp.units['mean_monthly']})",
        )
        ox.bar(prcp.mean_monthly, color="g", alpha=0.2, width=1)

    for col in daily:
        _fdc = fdc[[col, f"{col}_rank"]]
        _fdc = _fdc[_fdc > threshold]
        axs[1, 2].plot(_fdc[f"{col}_rank"], _fdc[col], cycle=cycle)
    logq = "$\\log \\left( Q \\right)$"
    axs[1, 2].format(
        yscale="symlog",
        xlabel="Exceedance Probability",
        ylabel=f"{logq} ({qdaily.units['ranked']})",
    )

    if output is not None:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(output, dpi=300)

    if close:
        pplt.close(fig)


def descriptor_legends() -> Tuple[ListedColormap, BoundaryNorm, List[int]]:
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    nlcd_meta = helpers.nlcd_helper()
    bounds = [int(v) for v in nlcd_meta["descriptors"]]
    with contextlib.suppress(ValueError):
        bounds.remove(127)

    cmap = ListedColormap(list(nlcd_meta["colors"].values())[: len(bounds)])
    norm = BoundaryNorm(bounds, cmap.N)
    levels = bounds + [30]
    return cmap, norm, levels


def cover_legends() -> Tuple[ListedColormap, BoundaryNorm, List[int]]:
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    nlcd_meta = helpers.nlcd_helper()
    bounds = list(nlcd_meta["colors"])
    with contextlib.suppress(ValueError):
        bounds.remove(127)

    cmap = ListedColormap(list(nlcd_meta["colors"].values()))
    norm = BoundaryNorm(bounds, cmap.N)
    levels = bounds + [100]
    return cmap, norm, levels
