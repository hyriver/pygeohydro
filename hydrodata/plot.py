"""Plot hydrological signatures.

Plots includes  daily, monthly and annual hydrograph as well as regime
curve (monthly mean) and flow duration curve.
"""
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from . import helpers, utils
from .exceptions import InvalidInputType


def signatures(
    daily: Union[pd.DataFrame, pd.Series],
    daily_unit: str = "cms",
    precipitation: Optional[pd.Series] = None,
    prcp_unit: str = "mm/day",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (13, 13),
    threshold: float = 1e-3,
    output: Union[str, Path] = None,
) -> None:
    """Plot hydrological signatures with w/ or w/o precipitation.

    Plots includes daily, monthly and annual hydrograph as well as
    regime curve (mean monthly) and flow duration curve. The input
    discharges are converted from cms to mm/day based on the watershed
    area, if provided.

    Parameters
    ----------
    daily : pd.DataFrame or pd.Series
        The column names are used as labels on the plot and the column values should be
        daily streamflow.
    daily_unit : str, optional
        The unit of the daily streamflow to appear on the plots, defaults to cms.
    precipitation : pd.Series, optional
        Daily precipitation time series in :math:`mm/day`. If given, the data is
        plotted on the second x-axis at the top.
    prcp_unit : str, optional
        The unit of the precipitation to appear on the plots, defaults to mm/day.
    title : str, optional
        The plot supertitle.
    figsize : tuple, optional
        Width and height of the plot in inches, defaults to (13, 13) inches.
    threshold : float, optional
        The threshold for cutting off the discharge for the flow duration
        curve to deal with log 0 issue, defaults to :math:`1e-3 mm/day`.
    output : str, optional
        Path to save the plot as png, defaults to ``None`` which means
        the plot is not saved to a file.
    """
    pd.plotting.register_matplotlib_converters()

    if not isinstance(daily, (pd.DataFrame, pd.Series)):
        raise InvalidInputType("daily", "pd.DataFrame, pd.Series")

    if precipitation is None:
        prcp = None
    else:
        if not isinstance(precipitation, pd.Series):
            raise InvalidInputType("daily", "pd.DataFrame, pd.Series")
        prcp = prepare_plot_data(precipitation)

    discharge = prepare_plot_data(daily)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2)
    sub_ax = [gs[0, :], gs[1, :], gs[2, 0], gs[2, 1], gs[3, :]]

    for sp, f in zip(sub_ax[:-1], discharge._fields[:-2]):
        ax = fig.add_subplot(sp)
        _discharge = getattr(discharge, f)  # noqa: B009
        _title = discharge.titles[f]
        qxval = _discharge.index

        ax.plot(qxval, _discharge)
        ax.set_ylabel(f"$Q$ ({daily_unit})")

        if prcp is not None:
            _prcp = getattr(prcp, f)  # noqa: B009
            _prcp = _prcp.loc[_prcp.index.intersection(qxval)]

            ax_p = ax.twinx()
            ax_p.bar(
                _prcp.index, _prcp.to_numpy().ravel(), alpha=0.7, width=prcp.bar_width[f], color="g"
            )
            ax_p.set_ylim(_prcp.max().to_numpy()[0] * 2.5, 0)
            ax_p.set_ylabel(f"$P$ ({prcp_unit})")

        ax.set_xlim(qxval[0], qxval[-1])
        ax.set_xlabel("")
        ax.set_title(_title)
        if len(_discharge.columns) > 1 and f == "daily":
            ax.legend(_discharge.columns, loc="best")

    ax = fig.add_subplot(sub_ax[-1])
    for col in discharge.daily:
        dc = discharge.ranked[[col, f"{col}_rank"]]
        dc = dc[dc > threshold]
        ax.plot(dc[f"{col}_rank"], dc[col], label=col)

    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% Exceedance")
    ax.set_ylabel(fr"$\log(Q)$ ({daily_unit})")
    ax.set_title("Flow Duration Curve")

    plt.tight_layout()
    plt.suptitle(title, size=16, y=1.02)

    if output is not None:
        utils.check_dir(output)
        plt.savefig(output, dpi=300, bbox_inches="tight")


class PlotDataType(NamedTuple):
    """Data structure for plotting hydrologic signatures."""

    daily: pd.DataFrame
    monthly: pd.DataFrame
    annual: pd.DataFrame
    mean_monthly: pd.DataFrame
    ranked: pd.DataFrame
    bar_width: Dict[str, int]
    titles: Dict[str, str]


def prepare_plot_data(daily: Union[pd.DataFrame, pd.Series]) -> PlotDataType:
    """Generae a structured data for plotting hydrologic signatures.

    Parameters
    ----------
    daily : pandas.Series or pandas.DataFrame
        The data to be processed
    ranked : bool, optional
        Whether to sort the data by rank for plotting flow duration curve, defaults to False.

    Returns
    -------
    NamedTuple
        Containing ``daily, ``monthly``, ``annual``, ``mean_monthly``, ``ranked`` fields.
    """
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()

    monthly = daily.groupby(pd.Grouper(freq="M")).sum()
    annual = daily.groupby(pd.Grouper(freq="Y")).sum()
    mean_monthly = utils.mean_monthly(daily)
    ranked = utils.exceedance(daily)
    _titles = [
        "Total Hydrograph (daily)",
        "Total Hydrograph (monthly)",
        "Total Hydrograph (annual)",
        "Regime Curve (monthly mean)",
        "Flow Duration Curve",
    ]
    fields = PlotDataType._fields
    titles = dict(zip(fields[:-1], _titles))
    bar_width = dict(zip(fields[:-2], [1, 30, 365, 1]))
    return PlotDataType(daily, monthly, annual, mean_monthly, ranked, bar_width, titles)


def cover_legends() -> Tuple[ListedColormap, BoundaryNorm, List[float]]:
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    nlcd_meta = helpers.nlcd_helper()
    bounds = list(nlcd_meta["colors"].keys())

    cmap = ListedColormap(list(nlcd_meta["colors"].values()))
    norm = BoundaryNorm(bounds, cmap.N)
    levels = bounds + [100]
    return cmap, norm, levels
