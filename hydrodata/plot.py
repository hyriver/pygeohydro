#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot hydrological signatures.

Plots includes  daily, monthly and annual hydrograph as well as
regime curve (monthly mean) and flow duration curve.
"""
from hydrodata import utils


def signatures(
    Q_daily, prcp=None, title=None, figsize=(13, 13), threshold=1e-3, output=None,
):
    """Plot hydrological signatures with w/ or w/o precipitation.

    Plots includes daily, monthly and annual hydrograph as well as
    regime curve (mean monthly) and flow duration curve. The input
    discharges are converted from cms to mm/day based on the watershed
    area, if provided.

    Parameters
    ----------
    Q_daily : dict of tuple
        The first element is a series containing daily discharges in m$^3$/s
        and the second element is the contributing drainage area in km$^2$.
        The dict keys are the labels on the plot.
    prcp : series, optional
        Daily precipitation time series in mm/day. If given, the data is
        plotted on the second x-axis at the top.
    title : str, optional
        The plot supertitle.
    figsize : tuple, optional
        Width and height of the plot in inches, defaults to (13, 13) inches.
    threshold : float, optional
        The threshold for cutting off the discharge for the flow duration
        curve to deal with log 0 issue, defaults to 1e-3 mm/day.
    output : string, optional
        Path to save the plot as png, defaults to `None` which means
        the plot is not saved to a file.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    pd.plotting.register_matplotlib_converters()
    mpl.rcParams["figure.dpi"] = 300

    if not isinstance(Q_daily, dict):
        raise TypeError("The daily_dict argument should be a dictionary.")

    for _, v in Q_daily.items():
        if isinstance(v, tuple):
            if not len(v) == 2:
                raise ValueError(
                    "The tuple should have exactly two elemenets, (Q, area)."
                )
        else:
            raise TypeError("The values of the input dictionary should be tuples.")

    # convert cms to mm/day
    daily_dict = {
        k: v[0] * 1000.0 * 24.0 * 3600.0 / (v[1] * 1.0e6) for k, v in Q_daily.items()
    }

    month_Q_dict, year_Q_dict, mean_month_Q_dict, Q_fdc_dict = {}, {}, {}, {}
    for label, daily in daily_dict.items():
        month_Q_dict[label] = daily.groupby(pd.Grouper(freq="M")).sum()
        year_Q_dict[label] = daily.groupby(pd.Grouper(freq="Y")).sum()
        mean_month_Q_dict[label] = utils.mean_monthly(daily)
        Q_fdc_dict[label] = utils.exceedance(daily[daily > threshold])

    if prcp is not None:
        month_P = prcp.groupby(pd.Grouper(freq="M")).sum()
        year_P = prcp.groupby(pd.Grouper(freq="Y")).sum()
        mean_month_P = utils.mean_monthly(prcp)

    plt.close("all")
    fig = plt.figure(1, figsize=figsize)

    ax1 = plt.subplot(4, 2, (1, 2))
    dates = get_daterange(daily_dict)

    for label, daily in daily_dict.items():
        ax1.plot(daily.index.to_pydatetime(), daily, label=label)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylabel("$Q$ (mm/day)")
    ax1.set_xlabel("")
    ax1.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax1.set_title("Total Hydrograph (daily)")
    if len(daily_dict) > 1:
        ax1.legend(list(daily_dict.keys()), loc="best")

    if prcp is not None:
        ax12 = ax1.twinx()
        ax12.bar(prcp.index.to_pydatetime(), prcp.values, alpha=0.7, width=1, color="g")
        ax12.set_ylim(0, prcp.max() * 2.5)
        ax12.set_ylim(ax12.get_ylim()[::-1])
        ax12.set_ylabel("$P$ (mm/day)")
        ax12.set_xlabel("")

    ax2 = plt.subplot(4, 2, (3, 4))
    dates = get_daterange(month_Q_dict)

    for label, month_Q in month_Q_dict.items():
        ax2.plot(month_Q.index.to_pydatetime(), month_Q, label=label)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.set_xlabel("")
    ax2.set_ylabel("$Q$ (mm/month)")
    ax2.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax2.set_title("Total Hydrograph (monthly)")

    if prcp is not None:
        ax22 = ax2.twinx()
        ax22.bar(
            month_P.index.to_pydatetime(),
            month_P.values,
            alpha=0.7,
            width=30,
            color="g",
        )
        ax22.set_ylim(0, month_P.max() * 2.5)
        ax22.set_ylim(ax22.get_ylim()[::-1])
        ax22.set_ylabel("$P$ (mm/day)")
        ax22.set_xlabel("")

    ax3 = plt.subplot(4, 2, 5)
    dates = list(mean_month_Q_dict.values())[0].index.astype("O")

    for label, mean_month_Q in mean_month_Q_dict.items():
        ax3.plot(dates, mean_month_Q, label=label)
    ax3.set_xlim(dates[0], dates[-1])
    ax3.set_xlabel("")
    ax3.set_ylabel(r"$\overline{Q}$ (mm/month)")
    ax3.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax3.set_title("Regime Curve (monthly mean)")

    if prcp is not None:
        ax32 = ax3.twinx()
        ax32.bar(dates, mean_month_P.values, alpha=0.7, width=1, color="g")
        ax32.set_ylim(0, mean_month_P.max() * 2.5)
        ax32.set_ylim(ax32.get_ylim()[::-1])
        ax32.set_ylabel("$P$ (mm/day)")
        ax32.set_xlabel("")

    ax4 = plt.subplot(4, 2, 6)
    dates = get_daterange(year_Q_dict)

    for label, year_Q in year_Q_dict.items():
        ax4.plot(year_Q.index.to_pydatetime(), year_Q, label=label)
    ax4.set_xlim(dates[0], dates[-1])
    ax4.set_xlabel("")
    ax4.set_ylabel("$Q$ (mm/year)")
    ax4.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax4.set_title("Total Hydrograph (annual)")

    if prcp is not None:
        ax42 = ax4.twinx()
        ax42.bar(
            year_P.index.to_pydatetime(), year_P.values, alpha=0.7, width=365, color="g"
        )
        ax42.set_xlim(dates[0], dates[-1])
        ax42.set_ylim(0, year_P.max() * 2.5)
        ax42.set_ylim(ax42.get_ylim()[::-1])
        ax42.set_ylabel("$P$ (mm/day)")
        ax42.set_xlabel("")

    ax5 = plt.subplot(4, 2, (7, 8))
    for label, Q_fdc in Q_fdc_dict.items():
        ax5.plot(Q_fdc.index.values, Q_fdc, label=label)
    ax5.set_yscale("log")
    ax5.set_xlim(0, 100)
    ax5.set_xlabel("% Exceedence")
    ax5.set_ylabel(r"$\log(Q)$ (mm/day)")
    ax5.set_title("Flow Duration Curve")

    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=True)
    plt.tight_layout()
    plt.suptitle(title, size=16, y=1.02)

    if output is None:
        return
    else:
        from pathlib import Path

        output = Path(output)
        if not output.parent.is_dir():
            try:
                import os

                os.makedirs(output.parent)
            except OSError:
                print(f"output directory cannot be created: {output.parent}")

        plt.savefig(output, dpi=300, bbox_inches="tight")
        return


def get_daterange(Q_dict):
    """Find data range of several data series."""
    import pandas as pd

    return pd.date_range(
        min([q.index[0] for q in list(Q_dict.values())]),
        max([q.index[-1] for q in list(Q_dict.values())]),
    ).to_pydatetime()


def cover_lenegeds():
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = [
        (0.33, 0.46, 0.66),
        (1.0, 1.0, 1.0),
        (0.91, 0.82, 0.82),
        (0.89, 0.62, 0.55),
        (1.0, 0.0, 0.0),
        (0.71, 0.0, 0.0),
        (0.82, 0.8, 0.75),
        (0.52, 0.78, 0.49),
        (0.22, 0.51, 0.31),
        (0.83, 0.91, 0.69),
        (0.69, 0.59, 0.24),
        (0.86, 0.79, 0.56),
        (0.99, 0.91, 0.67),
        (0.82, 0.82, 0.51),
        (0.64, 0.8, 0.32),
        (0.51, 0.73, 0.62),
        (0.98, 0.96, 0.36),
        (0.79, 0.57, 0.27),
        (0.78, 0.9, 0.97),
        (0.39, 0.7, 0.84),
    ]
    values = [
        11,
        12,
        21,
        22,
        23,
        24,
        31,
        41,
        42,
        43,
        51,
        52,
        71,
        72,
        73,
        74,
        81,
        82,
        90,
        95,
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(values, cmap.N)
    return cmap, norm
