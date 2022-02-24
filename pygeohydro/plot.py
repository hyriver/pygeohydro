"""Plot hydrological signatures.

Plots includes  daily, monthly and annual hydrograph as well as regime
curve (monthly mean) and flow duration curve.
"""
import calendar
import contextlib
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from . import helpers
from .exceptions import InvalidInputType


class PlotDataType(NamedTuple):
    """Data structure for plotting hydrologic signatures."""

    daily: pd.DataFrame
    monthly: pd.DataFrame
    annual: pd.DataFrame
    mean_monthly: pd.DataFrame
    ranked: pd.DataFrame
    bar_width: Dict[str, int]
    titles: Dict[str, str]
    units: Dict[str, str]


def exceedance(daily: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Compute Flow duration (rank, sorted obs)."""
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()

    ranks = daily.rank(ascending=False, pct=True) * 100
    fdc = [
        pd.DataFrame({c: daily[c], f"{c}_rank": ranks[c]})
        .sort_values(by=f"{c}_rank")
        .reset_index(drop=True)
        for c in daily
    ]
    return pd.concat(fdc, axis=1)


def prepare_plot_data(daily: Union[pd.DataFrame, pd.Series]) -> PlotDataType:
    """Generae a structured data for plotting hydrologic signatures.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed

    Returns
    -------
    PlotDataType
        Containing ``daily, ``monthly``, ``annual``, ``mean_monthly``, ``ranked`` fields.
    """
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()
    daily.index = daily.index.tz_localize(None)

    monthly = daily.groupby(pd.Grouper(freq="M")).sum()

    annual = daily.groupby(pd.Grouper(freq="Y")).sum()

    month_abbr = dict(enumerate(calendar.month_abbr))
    mean_month = daily.groupby(pd.Grouper(freq="M")).sum()
    mean_month = mean_month.groupby(mean_month.index.month).mean()
    mean_month.index = mean_month.index.map(month_abbr)

    ranked = exceedance(daily)
    _titles = [
        "Total Hydrograph (daily)",
        "Total Hydrograph (monthly)",
        "Total Hydrograph (annual)",
        "Regime Curve (monthly mean)",
        "Flow Duration Curve",
    ]
    _units = [
        "mm/day",
        "mm/month",
        "mm/year",
        "mm/month",
        "mm/day",
    ]
    fields = PlotDataType._fields
    titles = dict(zip(fields[:-1], _titles))
    units = dict(zip(fields[:-1], _units))
    bar_width = dict(zip(fields[:-2], [1, 30, 365, 1]))
    return PlotDataType(daily, monthly, annual, mean_month, ranked, bar_width, titles, units)


def _prepare_plot_data(
    daily: Union[pd.DataFrame, pd.Series],
    precipitation: Optional[Union[pd.DataFrame, pd.Series]] = None,
) -> Tuple[PlotDataType, Optional[PlotDataType]]:
    if not isinstance(daily, (pd.DataFrame, pd.Series)):
        raise InvalidInputType("daily", "pd.DataFrame or pd.Series")

    discharge = prepare_plot_data(daily)

    if not isinstance(precipitation, (pd.DataFrame, pd.Series)) and precipitation is not None:
        raise InvalidInputType("precipitation", "pd.DataFrame or pd.Series")

    prcp = None if precipitation is None else prepare_plot_data(precipitation)

    return discharge, prcp


def signatures(
    daily: Union[pd.DataFrame, pd.Series],
    precipitation: Optional[pd.Series] = None,
    title: Optional[str] = None,
    title_ypos: float = 1.02,
    figsize: Tuple[int, int] = (14, 13),
    threshold: float = 1e-3,
    output: Optional[Union[str, Path]] = None,
) -> None:
    """Plot hydrological signatures with w/ or w/o precipitation.

    Plots includes daily, monthly and annual hydrograph as well as
    regime curve (mean monthly) and flow duration curve. The input
    discharges are converted from cms to mm/day based on the watershed
    area, if provided.

    Parameters
    ----------
    daily : pd.DataFrame or pd.Series
        The streamflows in mm/day. The column names are used as labels
        on the plot and the column values should be daily streamflow.
    precipitation : pd.Series, optional
        Daily precipitation time series in mm/day. If given, the data is
        plotted on the second x-axis at the top.
    title : str, optional
        The plot supertitle.
    title_ypos : float
        The vertical position of the plot title, default to 1.02
    figsize : tuple, optional
        Width and height of the plot in inches, defaults to (14, 13) inches.
    threshold : float, optional
        The threshold for cutting off the discharge for the flow duration
        curve to deal with log 0 issue, defaults to :math:`1^{-3}` mm/day.
    output : str, optional
        Path to save the plot as png, defaults to ``None`` which means
        the plot is not saved to a file.
    """
    discharge, prcp = _prepare_plot_data(daily, precipitation)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2)
    sub_ax = [gs[0, :], gs[1, :], gs[2, 0], gs[2, 1], gs[3, :]]

    for sp, f in zip(sub_ax[:-1], discharge._fields[:-2]):
        ax = fig.add_subplot(sp)
        _discharge = getattr(discharge, f)  # noqa: B009
        _title = discharge.titles[f]
        _unit = discharge.units[f]
        qxval = _discharge.index

        ax.plot(qxval, _discharge.to_numpy())
        ax.set_ylabel(f"$Q$ ({_unit})")

        if prcp is not None:
            _prcp = getattr(prcp, f)  # noqa: B009
            _prcp = _prcp.loc[_prcp.index.intersection(qxval)]

            ax_p = ax.twinx()
            if _prcp.shape[0] > 1000:
                ax_p.plot(_prcp.index, _prcp.to_numpy().ravel(), alpha=0.7, color="g")
            else:
                ax_p.bar(
                    _prcp.index,
                    _prcp.to_numpy().ravel(),
                    alpha=0.7,
                    width=prcp.bar_width[f],
                    color="g",
                    align="edge",
                )
            ax_p.set_ylim(_prcp.max().to_numpy()[0] * 2.5, 0)
            ax_p.set_ylabel(f"$P$ ({_unit})")

        ax.set_xlim(qxval[0], qxval[-1])
        ax.set_xlabel("")
        ax.set_title(_title)

        if len(_discharge.columns) > 1 and f == "daily":
            ax.legend(
                _discharge.columns,
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc="lower right",
                ncol=len(_discharge),
            )

        if f == "annual":
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    ax = fig.add_subplot(sub_ax[-1])
    for col in discharge.daily:
        dc = discharge.ranked[[col, f"{col}_rank"]]
        dc = dc[dc > threshold]
        ax.plot(dc[f"{col}_rank"], dc[col], label=col)

    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% Exceedance")
    ax.set_ylabel(rf"$\log(Q)$ ({discharge.units['ranked']})")
    ax.set_title("Flow Duration Curve")

    plt.tight_layout()
    plt.suptitle(title, size=16, y=title_ypos)

    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches="tight")


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
