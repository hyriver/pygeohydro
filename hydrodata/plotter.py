def RC(daily):
    """
    Compute monthly mean over the whole time series for the regime curve.
    """
    import calendar

    d = dict(enumerate(calendar.month_abbr))
    mean_month = daily.groupby(daily.index.month).mean()
    mean_month.index = mean_month.index.map(d)
    return mean_month


def FDC(daily):
    """
    Computes Flow duration (rank, sorted obs). The zero discharges are handled
    by dropping since log 0 is undefined.
    """
    import pandas as pd

    if not isinstance(daily, pd.Series):
        msg = "The input should be of type pandas Series."
        raise TypeError(msg)

    rank = daily.rank(ascending=False, pct=True) * 100
    fdc = pd.concat([daily, rank], axis=1)
    fdc.columns = ["Q", "rank"]
    fdc.sort_values(by=["rank"], inplace=True)
    fdc.set_index("rank", inplace=True, drop=True)
    return fdc


def plot(daily_dict,
         prcp,
         area,
         title,
         figsize=(8, 10),
         threshold=1e-3,
         output=None):
    """Plot hydrological signatures with precipitation as the second axis.
       Plots includes  daily, monthly and annual hydrograph as well as
       regime curve (monthly mean) and flow duration curve.

       Arguments:
           daily_dict (dataframe): Daily discharge timeseries in mm/day.
                                   A dataframe or a dictionary of dataframes
                                   can be passed where keys are lables and
                                   values are dataframes.
           prcp (dataframe): Daily precipitation timeseries in mm/day.
           area (float): Watershed area in km$^2$ (for converting
                         cms to mm/day)
           title (str): Plot's supertitle.
           figsize (tuple): Width and height of the plot in inches.
                            The default is (8, 10)
           threshold (float): The threshold for cutting off the discharge for
                              the flow duration curve to deal with log 0 issue.
                              The default is 1e-3.
           output (str): Path to save the plot as png. The default is `None`
                           which means the plot is not saved to a file.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    pd.plotting.register_matplotlib_converters()
    mpl.rcParams["figure.dpi"] = 300

    if isinstance(daily_dict, pd.Series):
        daily_dict = {'Q': daily_dict}
    elif not isinstance(daily_dict, dict):
        raise TypeError(
            'The daily_dict argument can be either a Pandas series ' +
            'or a dictionary of Pandas series.')

    # convert cms to mm/day
    daily_dict = {
        k: v * 1000.0 * 24.0 * 3600.0 / (area * 1.0e6)
        for k, v in daily_dict.items()
    }

    month_Q_dict, year_Q_dict, mean_month_Q_dict, Q_fdc_dict = {}, {}, {}, {}
    for label, daily in daily_dict.items():
        month_Q_dict[label] = daily.groupby(pd.Grouper(freq="M")).sum()
        year_Q_dict[label] = daily.groupby(pd.Grouper(freq="Y")).sum()
        mean_month_Q_dict[label] = RC(daily)
        Q_fdc_dict[label] = FDC(daily[daily > threshold])

    month_P = prcp.groupby(pd.Grouper(freq="M")).sum()
    year_P = prcp.groupby(pd.Grouper(freq="Y")).sum()
    mean_month_P = RC(prcp)

    plt.close("all")
    fig = plt.figure(1, figsize=figsize)

    ax1 = plt.subplot(4, 2, (1, 2))
    ax12 = ax1.twinx()
    dates = get_daterange(daily_dict)

    for label, daily in daily_dict.items():
        ax1.plot(daily.index.to_pydatetime(), daily, label=label)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylabel("$Q$ (mm/day)")
    ax1.set_xlabel("")
    ax1.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax1.set_title("Total Hydrograph (daily)")
    if len(daily_dict) > 1:
        ax1.legend(list(daily_dict.keys()), loc='best')

    ax12.bar(prcp.index.to_pydatetime(),
             prcp.values,
             alpha=0.7,
             width=1,
             color="g")
    ax12.set_ylim(0, prcp.max() * 2.5)
    ax12.set_ylim(ax12.get_ylim()[::-1])
    ax12.set_ylabel("$P$ (mm/day)")
    ax12.set_xlabel("")

    ax2 = plt.subplot(4, 2, (3, 4))
    ax22 = ax2.twinx()
    dates = get_daterange(month_Q_dict)

    for label, month_Q in month_Q_dict.items():
        ax2.plot(month_Q.index.to_pydatetime(), month_Q, label=label)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.set_xlabel("")
    ax2.set_ylabel("$Q$ (mm/month)")
    ax2.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax2.set_title("Total Hydrograph (monthly)")

    ax22.bar(month_P.index.to_pydatetime(),
             month_P.values,
             alpha=0.7,
             width=30,
             color="g")
    ax22.set_ylim(0, month_P.max() * 2.5)
    ax22.set_ylim(ax22.get_ylim()[::-1])
    ax22.set_ylabel("$P$ (mm/day)")
    ax22.set_xlabel("")

    ax3 = plt.subplot(4, 2, 5)
    ax32 = ax3.twinx()
    dates = list(mean_month_Q_dict.values())[0].index.astype("O")

    for label, mean_month_Q in mean_month_Q_dict.items():
        ax3.plot(dates, mean_month_Q, label=label)
    ax3.set_xlim(dates[0], dates[-1])
    ax3.set_xlabel("")
    ax3.set_ylabel(r"$\overline{Q}$ (mm/month)")
    ax3.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax3.set_title("Regime Curve (monthly mean)")

    ax32.bar(dates, mean_month_P.values, alpha=0.7, width=1, color="g")
    ax32.set_ylim(0, mean_month_P.max() * 2.5)
    ax32.set_ylim(ax32.get_ylim()[::-1])
    ax32.set_ylabel("$P$ (mm/day)")
    ax32.set_xlabel("")

    ax4 = plt.subplot(4, 2, 6)
    ax42 = ax4.twinx()
    dates = get_daterange(year_Q_dict)

    for label, year_Q in year_Q_dict.items():
        ax4.plot(year_Q.index.to_pydatetime(), year_Q, label=label)
    ax4.set_xlim(dates[0], dates[-1])
    ax4.set_xlabel("")
    ax4.set_ylabel("$Q$ (mm/year)")
    ax4.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax4.set_title("Total Hydrograph (annual)")

    ax42.bar(year_P.index.to_pydatetime(),
             year_P.values,
             alpha=0.7,
             width=365,
             color="g")
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
    ax5.set_ylabel("$\log(Q)$ (mm/day)")
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
                print("output directory cannot be created: {:s}".format(
                    output.parent))

        plt.savefig(output, dpi=300, bbox_inches="tight")
        return


def plot_discharge(daily_dict,
                   area,
                   title,
                   figsize=(8, 10),
                   threshold=1e-3,
                   output=None):
    """Plot hydrological signatures without precipitation; daily, monthly and
       annual hydrograph as well as regime curve (monthly mean) and
       flow duration curve.

       Arguments:
           daily_dict (dataframe): Daily discharge timeseries in mm/day.
                                   A dataframe or a dictionary of dataframes
                                   can be passed where keys are lables and
                                   values are dataframes.
           area (float): Watershed area in km$^2$ (for converting
                         cms to mm/day).
           title (str): Plot's supertitle.
           figsize (tuple): Width and height of the plot in inches.
                            The default is (8, 10)
           threshold (float): The threshold for cutting off the discharge for
                              the flow duration curve to deal with log 0 issue.
                              The default is 1e-3.
           output (str): Path to save the plot as png. The default is `None`
                           which means the plot is not saved to a file.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    pd.plotting.register_matplotlib_converters()
    mpl.rcParams["figure.dpi"] = 300

    if isinstance(daily_dict, pd.Series):
        daily_dict = {'Q': daily_dict}
    elif not isinstance(daily_dict, dict):
        raise TypeError(
            'The daily_dict argument can be either a Pandas series ' +
            'or a dictionary of Pandas series.')

    # convert cms to mm/day
    daily_dict = {
        k: v * 1000.0 * 24.0 * 3600.0 / (area * 1.0e6)
        for k, v in daily_dict.items()
    }

    month_Q_dict, year_Q_dict, mean_month_Q_dict, Q_fdc_dict = {}, {}, {}, {}
    for label, daily in daily_dict.items():
        month_Q_dict[label] = daily.groupby(pd.Grouper(freq="M")).sum()
        year_Q_dict[label] = daily.groupby(pd.Grouper(freq="Y")).sum()
        mean_month_Q_dict[label] = RC(daily)
        Q_fdc_dict[label] = FDC(daily[daily > threshold])

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
        ax1.legend(list(daily_dict.keys()), loc='best')

    ax2 = plt.subplot(4, 2, (3, 4))
    dates = get_daterange(month_Q_dict)

    for label, month_Q in month_Q_dict.items():
        ax2.plot(month_Q.index.to_pydatetime(), month_Q, label=label)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.set_xlabel("")
    ax2.set_ylabel("$Q$ (mm/month)")
    ax2.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax2.set_title("Total Hydrograph (monthly)")

    ax3 = plt.subplot(4, 2, 5)
    dates = list(mean_month_Q_dict.values())[0].index.astype("O")

    for label, mean_month_Q in mean_month_Q_dict.items():
        ax3.plot(dates, mean_month_Q, label=label)
    ax3.set_xlim(dates[0], dates[-1])
    ax3.set_xlabel("")
    ax3.set_ylabel(r"$\overline{Q}$ (mm/month)")
    ax3.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax3.set_title("Regime Curve (monthly mean)")

    ax4 = plt.subplot(4, 2, 6)
    dates = get_daterange(year_Q_dict)

    for label, year_Q in year_Q_dict.items():
        ax4.plot(year_Q.index.to_pydatetime(), year_Q, label=label)
    ax4.set_xlim(dates[0], dates[-1])
    ax4.set_xlabel("")
    ax4.set_ylabel("$Q$ (mm/year)")
    ax4.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    ax4.set_title("Total Hydrograph (annual)")

    ax5 = plt.subplot(4, 2, (7, 8))
    for label, Q_fdc in Q_fdc_dict.items():
        ax5.plot(Q_fdc.index.values, Q_fdc, label=label)
    ax5.set_yscale("log")
    ax5.set_xlim(0, 100)
    ax5.set_xlabel("% Exceedence")
    ax5.set_ylabel("$\log(Q)$ (mm/day)")
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
                print("output directory cannot be created: {:s}".format(
                    output.parent))

        plt.savefig(output, dpi=300, bbox_inches="tight")
        return


def get_daterange(Q_dict):
    import pandas as pd

    return pd.date_range(min([q.index[0] for q in list(Q_dict.values())]),
                         max([q.index[-1] for q in list(Q_dict.values())
                              ])).to_pydatetime()
