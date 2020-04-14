#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some utilities for Hydrodata"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features as rio_features
import rasterio.warp as rio_warp
from hydrodata import helpers
from owslib.wms import WebMapService
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    RetryError,
    Timeout,
)
from shapely.geometry import Point, mapping


def retry_requests(
    retries=3,
    backoff_factor=0.5,
    status_to_retry=(500, 502, 504),
    prefixes=("http://", "https://"),
):
    """Configures the passed-in session to retry on failed requests.

    The fails can be due to connection errors, specific HTTP response
    codes and 30X redirections. The original code is taken from:
    https://github.com/bustawin/retry-requests

    Parameters
    ---------
    retries : int
        The number of maximum retries before raising an exception.
    backoff_factor : float
        A factor used to compute the waiting time between retries.
    status_to_retry : tuple of ints
        A tuple of status codes that trigger the reply behaviour.

    Returns
    -------
    session
        A session object with retry configurations.
    """

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3 import Retry

    session = requests.Session()

    r = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_to_retry,
        method_whitelist=False,
    )
    adapter = HTTPAdapter(max_retries=r)
    for prefix in prefixes:
        session.mount(prefix, adapter)
    session.hooks = {"response": lambda r, *args, **kwargs: r.raise_for_status()}

    return session


def get_url(session, url, payload=None):
    """Retrieve data from a url by GET using a requests session"""
    try:
        return session.get(url, params=payload)
    except (ConnectionError, HTTPError, RequestException, RetryError, Timeout):
        raise


def post_url(session, url, payload=None):
    """Retrieve data from a url by POST using a requests session"""
    try:
        return session.post(url, data=payload)
    except (ConnectionError, HTTPError, RequestException, RetryError, Timeout):
        raise


def check_dir(fpath):
    """Create parent directory for a file if doesn't exist"""
    parent = Path(fpath).parent
    if not parent.is_dir():
        try:
            os.makedirs(parent)
        except OSError:
            raise OSError(f"Parent directory cannot be created: {parent}")


def daymet_dates(start, end):
    """Correct dates for Daymet when leap years.

    Daymet doesn't account for leap years and removes
    Dec 31 when it's leap year. This function returns all
    the dates in the Daymet database within the provided date range.
    """

    period = pd.date_range(start, end)
    nl = period[~period.is_leap_year]
    lp = period[
        (period.is_leap_year) & (~period.strftime("%Y-%m-%d").str.endswith("12-31"))
    ]
    period = period[(period.isin(nl)) | (period.isin(lp))]
    years = [period[period.year == y] for y in period.year.unique()]
    return [(y[0], y[-1]) for y in years]


def elevation_byloc(lon, lat):
    """Get elevation from USGS 3DEP service for a coordinate.

    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude

    Returns
    -------
    float
        Elevation in meter
    """

    url = "https://nationalmap.gov/epqs/pqs.php"
    session = retry_requests()
    payload = {"output": "json", "x": lon, "y": lat, "units": "Meters"}
    r = get_url(session, url, payload)
    root = r.json()["USGS_Elevation_Point_Query_Service"]
    elevation = root["Elevation_Query"]["Elevation"]
    if elevation == -1000000:
        raise ValueError(
            f"The altitude of the requested coordinate ({lon}, {lat}) cannot be found."
        )
    else:
        return elevation


def elevation_bybbox(bbox, resolution, coords, crs="4326"):
    """Get elevation from DEM data for a list of coordinates.

    This function is intended for getting elevations for a gridded dataset.

    Parameters
    ----------
    bbox : list or tuple
        Bounding box with coordinates in [west, south, east, north] format.
    resolution : float
        Gridded data resolution
    coords : list of tuples
        A list of coordinates in (lon, lat) format to extract the elevations.

    Returns
    -------
    numpy.array
        An array of elevations in meters
    """
    if isinstance(bbox, list) or isinstance(bbox, tuple):
        if len(bbox) != 4:
            raise TypeError(
                "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
            )
    else:
        raise TypeError(
            "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
        )

    url = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"
    layers = ["3DEPElevation:None"]
    version = "1.3.0"

    west, south, east, north = bbox
    width = int((east - west) * 3600 / resolution)
    height = int(abs(north - south) / abs(east - west) * width)

    wms = WebMapService(url, version=version)
    img = wms.getmap(
        layers=layers, srs=crs, bbox=bbox, size=(width, height), format="image/geotiff",
    )

    with rio.MemoryFile() as memfile:
        memfile.write(img.read())
        with memfile.open() as src:
            elevations = np.array([e[0] for e in src.sample(coords)], dtype=np.float32)

    return elevations


def pet_fao_byloc(df, lon, lat):
    """Compute Potential EvapoTranspiration using Daymet dataset for a single location.

    The method is based on `FAO-56 <http://www.fao.org/docrep/X0490E/X0490E00.htm>`.

    Parameters
    ----------
    df : DataFrame
        A dataframe with columns named as follows:
        ``tmin (deg c)``, ``tmax (deg c)``, ``vp (Pa)``, ``srad (W/m^2)``, ``dayl (s)``
    lon : float
        Longitude of the location of interest
    lat : float
        Latitude of the location of interest

    Returns
    -------
    DataFrame
        The input DataFrame with an additional column named ``pet (mm/day)``
    """

    keys = [v for v in df.columns]
    reqs = ["tmin (deg c)", "tmax (deg c)", "vp (Pa)", "srad (W/m^2)", "dayl (s)"]

    missing = [r for r in reqs if r not in keys]
    if len(missing) > 0:
        msg = "These required variables are not in the dataset: "
        msg += ", ".join(x for x in missing)
        msg += f'\nRequired variables are {", ".join(x for x in reqs)}'
        raise KeyError(msg)

    dtype = df.dtypes[0]
    df["tmean (deg c)"] = 0.5 * (df["tmax (deg c)"] + df["tmin (deg c)"])
    Delta = (
        4098
        * (
            0.6108
            * np.exp(
                17.27 * df["tmean (deg c)"] / (df["tmean (deg c)"] + 237.3), dtype=dtype
            )
        )
        / ((df["tmean (deg c)"] + 237.3) ** 2)
    )
    elevation = elevation_byloc(lon, lat)

    P = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
    gamma = P * 0.665e-3

    G = 0.0  # recommended for daily data
    df["vp (Pa)"] = df["vp (Pa)"] * 1e-3

    e_max = 0.6108 * np.exp(
        17.27 * df["tmax (deg c)"] / (df["tmax (deg c)"] + 237.3), dtype=dtype
    )
    e_min = 0.6108 * np.exp(
        17.27 * df["tmin (deg c)"] / (df["tmin (deg c)"] + 237.3), dtype=dtype
    )
    e_s = (e_max + e_min) * 0.5
    e_def = e_s - df["vp (Pa)"]

    u_2 = 2.0  # recommended when no data is available

    jday = df.index.dayofyear
    R_s = df["srad (W/m^2)"] * df["dayl (s)"] * 1e-6

    alb = 0.23

    jp = 2.0 * np.pi * jday / 365.0
    d_r = 1.0 + 0.033 * np.cos(jp, dtype=dtype)
    delta = 0.409 * np.sin(jp - 1.39, dtype=dtype)
    phi = lat * np.pi / 180.0
    w_s = np.arccos(-np.tan(phi, dtype=dtype) * np.tan(delta, dtype=dtype))
    R_a = (
        24.0
        * 60.0
        / np.pi
        * 0.082
        * d_r
        * (
            w_s * np.sin(phi, dtype=dtype) * np.sin(delta, dtype=dtype)
            + np.cos(phi, dtype=dtype)
            * np.cos(delta, dtype=dtype)
            * np.sin(w_s, dtype=dtype)
        )
    )
    R_so = (0.75 + 2e-5 * elevation) * R_a
    R_ns = (1.0 - alb) * R_s
    R_nl = (
        4.903e-9
        * (
            ((df["tmax (deg c)"] + 273.16) ** 4 + (df["tmin (deg c)"] + 273.16) ** 4)
            * 0.5
        )
        * (0.34 - 0.14 * np.sqrt(df["vp (Pa)"]))
        * ((1.35 * R_s / R_so) - 0.35)
    )
    R_n = R_ns - R_nl

    df["pet (mm/day)"] = (
        0.408 * Delta * (R_n - G)
        + gamma * 900.0 / (df["tmean (deg c)"] + 273.0) * u_2 * e_def
    ) / (Delta + gamma * (1 + 0.34 * u_2))
    df["vp (Pa)"] = df["vp (Pa)"] * 1.0e3

    return df


def pet_fao_gridded(ds):
    """Compute Potential EvapoTranspiration using Daymet dataset.

    The method is based on `FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`.
    The following variables are required:
    tmin (deg c), tmax (deg c), lat, lon, vp (Pa), srad (W/m2), dayl (s/day)
    The computed PET's unit is mm/day.

    Parameters
    ----------
    ds : xarray.DataArray
        The dataset should include the following variables:
        ``tmin``, ``tmax``, ``lat``, ``lon``, ``vp``, ``srad``, ``dayl``

    Returns
    -------
    xarray.DataArray
        The input dataset with an additional variable called ``pet``.
    """

    keys = [v for v in ds.keys()]
    reqs = ["tmin", "tmax", "lat", "lon", "vp", "srad", "dayl"]

    missing = [r for r in reqs if r not in keys]
    if len(missing) > 0:
        msg = "These required variables are not in the dataset: "
        msg += ", ".join(x for x in missing)
        msg += f'\nRequired variables are {", ".join(x for x in reqs)}'
        raise KeyError(msg)

    dtype = ds.tmin.dtype
    dates = ds["time"]
    ds["tmean"] = 0.5 * (ds["tmax"] + ds["tmin"])
    ds["tmean"].attrs["units"] = "degree C"
    ds["delta"] = (
        4098
        * (0.6108 * np.exp(17.27 * ds["tmean"] / (ds["tmean"] + 237.3), dtype=dtype))
        / ((ds["tmean"] + 237.3) ** 2)
    )

    if "elevation" not in keys:
        coords = [
            (i, j)
            for i, j in zip(
                ds.sel(time=ds["time"][0]).lon.values.flatten(),
                ds.sel(time=ds["time"][0]).lat.values.flatten(),
            )
        ]
        margine = 0.05
        bbox = [
            ds.lon.min().values - margine,
            ds.lat.min().values - margine,
            ds.lon.max().values + margine,
            ds.lat.max().values + margine,
        ]
        resolution = (3600.0 * 180.0) / (6371000.0 * np.pi) * ds.res[0] * 1e3
        elevation = elevation_bybbox(bbox, resolution, coords).reshape(
            ds.dims["y"], ds.dims["x"]
        )
        ds["elevation"] = ({"y": ds.dims["y"], "x": ds.dims["x"]}, elevation)
        ds["elevation"].attrs["units"] = "m"

    P = 101.3 * ((293.0 - 0.0065 * ds["elevation"]) / 293.0) ** 5.26
    ds["gamma"] = P * 0.665e-3

    G = 0.0  # recommended for daily data
    ds["vp"] *= 1e-3

    e_max = 0.6108 * np.exp(17.27 * ds["tmax"] / (ds["tmax"] + 237.3), dtype=dtype)
    e_min = 0.6108 * np.exp(17.27 * ds["tmin"] / (ds["tmin"] + 237.3), dtype=dtype)
    e_s = (e_max + e_min) * 0.5
    ds["e_def"] = e_s - ds["vp"]

    u_2 = 2.0  # recommended when no wind data is available

    lat = ds.sel(time=ds["time"][0]).lat
    ds["time"] = pd.to_datetime(ds.time.values).dayofyear.astype(dtype)
    R_s = ds["srad"] * ds["dayl"] * 1e-6

    alb = 0.23

    jp = 2.0 * np.pi * ds["time"] / 365.0
    d_r = 1.0 + 0.033 * np.cos(jp, dtype=dtype)
    delta = 0.409 * np.sin(jp - 1.39, dtype=dtype)
    phi = lat * np.pi / 180.0
    w_s = np.arccos(-np.tan(phi, dtype=dtype) * np.tan(delta, dtype=dtype))
    R_a = (
        24.0
        * 60.0
        / np.pi
        * 0.082
        * d_r
        * (
            w_s * np.sin(phi, dtype=dtype) * np.sin(delta, dtype=dtype)
            + np.cos(phi, dtype=dtype)
            * np.cos(delta, dtype=dtype)
            * np.sin(w_s, dtype=dtype)
        )
    )
    R_so = (0.75 + 2e-5 * ds["elevation"]) * R_a
    R_ns = (1.0 - alb) * R_s
    R_nl = (
        4.903e-9
        * (((ds["tmax"] + 273.16) ** 4 + (ds["tmin"] + 273.16) ** 4) * 0.5)
        * (0.34 - 0.14 * np.sqrt(ds["vp"]))
        * ((1.35 * R_s / R_so) - 0.35)
    )
    ds["R_n"] = R_ns - R_nl

    ds["pet"] = (
        0.408 * ds["delta"] * (ds["R_n"] - G)
        + ds["gamma"] * 900.0 / (ds["tmean"] + 273.0) * u_2 * ds["e_def"]
    ) / (ds["delta"] + ds["gamma"] * (1 + 0.34 * u_2))
    ds["pet"].attrs["units"] = "mm/day"

    ds["time"] = dates
    ds["vp"] *= 1.0e3

    ds = ds.drop_vars(["delta", "gamma", "e_def", "R_n"])

    return ds


def mean_monthly(daily):
    """Compute monthly mean for the regime curve."""
    import calendar

    d = dict(enumerate(calendar.month_abbr))
    mean_month = daily.groupby(daily.index.month).mean()
    mean_month.index = mean_month.index.map(d)
    return mean_month


def exceedance(daily):
    """Compute Flow duration (rank, sorted obs).

    The zero discharges are handled by dropping since log 0 is undefined.
    """

    if not isinstance(daily, pd.Series):
        msg = "The input should be of type pandas Series."
        raise TypeError(msg)

    rank = daily.rank(ascending=False, pct=True) * 100
    fdc = pd.concat([daily, rank], axis=1)
    fdc.columns = ["Q", "rank"]
    fdc.sort_values(by=["rank"], inplace=True)
    fdc.set_index("rank", inplace=True, drop=True)
    return fdc


def subbasin_delineation(station_id):
    """Delineate subbasins of a watershed based on HUC12 pour points."""

    import hydrodata.datasets as hds

    trib_ids = hds.nhdplus_navigate_byid("nwissite", station_id).comid.tolist()
    trib = hds.nhdplus_byid(trib_ids, layer="nhdflowline_network")
    trib_fl = prepare_nhdplus(trib, 0, 0, purge_non_dendritic=True, warn=False)

    main_fl = trib_fl[trib_fl.streamleve == 1].copy()

    main_ids = main_fl.sort_values(by="hydroseq").comid.tolist()
    len_cs = main_fl.set_index("comid").sort_values(by="hydroseq").lengthkm.cumsum()

    pp = hds.nhdplus_navigate_byid("nwissite", station_id, dataSource="huc12pp")
    station = Point(
        mapping(main_fl[main_fl.comid == main_ids[0]].geometry.values[0])[
            "coordinates"
        ][0][-1][:-1]
    )
    pp = pp.append([{"geometry": station, "comid": main_ids[0]}], ignore_index=True)
    headwater = Point(
        mapping(main_fl[main_fl.comid == main_ids[-1]].geometry.values[0])[
            "coordinates"
        ][0][-1][:-1]
    )
    pp = pp.append([{"geometry": headwater, "comid": main_ids[-1]}], ignore_index=True)
    pp_sorted = pp[pp.comid.isin(main_ids)].comid.tolist()
    pp = pp.set_index("comid").loc[pp_sorted].reset_index()

    pp_dist = len_cs[len_cs.index.isin(pp_sorted)]
    pp_dist = pp_dist.diff()

    pp_idx = [main_ids[0]] + pp_dist[pp_dist > 1].index.tolist()

    if len(pp_idx) < 2:
        msg = "There are not enough pour points in the watershed for automatic delineation."
        msg += " Try passing the desired number of subbasins."
        raise ValueError(msg)

    pour_points = pp[pp.comid.isin(pp_idx[1:-1])]

    pp_idx = [main_ids.index(i) for i in pp_idx]
    idx_subs = [main_ids[pp_idx[i - 1] : pp_idx[i]] for i in range(1, len(pp_idx))]

    catchemnts = []
    sub = [trib_fl]
    fm = main_fl.copy()
    for idx_max in idx_subs:
        sub.append(
            hds.nhdplus_navigate_byid("comid", idx_max[-1], "upstreamTributaries")
        )
        idx_catch = sub[-2][~sub[-2].comid.isin(sub[-1].comid)].comid.tolist()
        catchemnts.append(hds.nhdplus_byid(idx_catch, "catchmentsp"))
        fm = fm[~fm.comid.isin(idx_max)]

    catchemnts[-1] = catchemnts[-1].append(
        hds.nhdplus_byid(sub[-1].comid.tolist(), "catchmentsp")
    )

    return catchemnts, pour_points


def interactive_map(bbox):
    """An interactive map including all USGS stations within a bounding box.

    Only stations that record(ed) daily streamflow data are included.

    Parameters
    ----------
    bbox : list
        List of corners in this order [west, south, east, north]

    Returns
    -------
    folium.Map
    """
    import folium
    import hydrodata.datasets as hds

    if not isinstance(bbox, list):
        raise ValueError("bbox should be a list: [west, south, east, north]")

    df = hds.nwis_siteinfo(bbox=bbox)
    df = df[
        [
            "site_no",
            "station_nm",
            "dec_lat_va",
            "dec_long_va",
            "alt_va",
            "alt_datum_cd",
            "huc_cd",
            "begin_date",
            "end_date",
        ]
    ]
    df["coords"] = [
        (lat, lon)
        for _, lat, lon in df[["dec_lat_va", "dec_long_va"]].itertuples(name=None)
    ]
    df["altitude"] = (
        df["alt_va"].astype(str) + " ft above " + df["alt_datum_cd"].astype(str)
    )
    df = df.drop(columns=["dec_lat_va", "dec_long_va", "alt_va", "alt_datum_cd"])

    dr = hds.nwis_siteinfo(bbox=bbox, expanded=True)[
        ["site_no", "drain_area_va", "contrib_drain_area_va"]
    ]
    df = df.merge(dr, on="site_no").dropna()

    df["drain_area_va"] = df["drain_area_va"].astype(str) + " square miles"
    df["contrib_drain_area_va"] = (
        df["contrib_drain_area_va"].astype(str) + " square miles"
    )

    df = df[
        [
            "site_no",
            "station_nm",
            "coords",
            "altitude",
            "huc_cd",
            "drain_area_va",
            "contrib_drain_area_va",
            "begin_date",
            "end_date",
        ]
    ]
    df.columns = [
        "Site No.",
        "Station Name",
        "Coordinate",
        "Altitude",
        "HUC8",
        "Drainage Area",
        "Contributing Drainga Area",
        "Begin date",
        "End data",
    ]

    msgs = []
    for row in df.itertuples(index=False):
        msg = ""
        for col in df.columns:
            msg += "".join(
                ["<strong>", col, "</strong> : ", f"{row[df.columns.get_loc(col)]}<br>"]
            )
        msgs.append(msg[:-4])

    df["msg"] = msgs
    df = df[["Coordinate", "msg"]]

    west, south, east, north = bbox
    lon = (west + east) * 0.5
    lat = (south + north) * 0.5

    m = folium.Map(location=(lat, lon), tiles="Stamen Terrain", zoom_start=12)

    for _, coords, msg in df.itertuples(name=None):
        folium.Marker(
            location=coords, popup=folium.Popup(msg, max_width=250), icon=folium.Icon()
        ).add_to(m)

    return m


def prepare_nhdplus(
    fl,
    min_network_size,
    min_path_length,
    min_path_size=0,
    purge_non_dendritic=True,
    warn=True,
):
    """Cleaning up and fixing issue in NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`_

    Parameters
    ----------
    fl : (Geo)DataFrame
        NHDPlus flowlines with at least the following columns:
        COMID, LENGTHKM, FTYPE, TerminalFl, FromNode, ToNode, TotDASqKM,
        StartFlag, StreamOrde, StreamCalc, TerminalPa, Pathlength,
        Divergence, Hydroseq, LevelPathI
    min_network_size : float
        Minimum size of drainage network in sqkm
    min_path_length : float
        Minimum length of terminal level path of a network in km.
    min_path_size : float
        Minimum size of outlet level path of a drainage basin in km.
        Drainage basins with an outlet drainage area smaller than
        this value will be removed.
    purge_non_dendritic : bool
        Whether to remove non dendritic paths.
    warn : bool
        Whether to show a message about the removed features, defaults to True.

    Return
    ------
    (Geo)DataFrame
        With an additional column named ``tocomid`` that represents downstream
        comid of features.
    """
    req_cols = [
        "comid",
        "lengthkm",
        "ftype",
        "terminalfl",
        "fromnode",
        "tonode",
        "totdasqkm",
        "startflag",
        "streamorde",
        "streamcalc",
        "terminalpa",
        "pathlength",
        "divergence",
        "hydroseq",
        "levelpathi",
    ]
    fl.columns = map(str.lower, fl.columns)
    if any(c not in fl.columns for c in req_cols):
        msg = "The required columns are not in the provided flowline dataframe."
        msg += f" The required columns are {', '.join(c for c in req_cols)}"
        raise ValueError(msg)

    extra_cols = ["comid"] + [c for c in fl.columns if c not in req_cols]
    int_cols = [
        "comid",
        "terminalfl",
        "fromnode",
        "tonode",
        "startflag",
        "streamorde",
        "streamcalc",
        "terminalpa",
        "divergence",
        "hydroseq",
        "levelpathi",
    ]
    for c in int_cols:
        fl[c] = fl[c].astype("Int64")

    fls = fl[req_cols].copy()
    if not any(fls.terminalfl == 1):
        if all(fls.terminalpa == fls.terminalpa.iloc[0]):
            fls.loc[fls.hydroseq == fls.hydroseq.min(), "terminalfl"] = 1
        else:
            raise ValueError("No terminal flag were found in the dataframe.")

    if purge_non_dendritic:
        fls = fls[
            ((fls.ftype != "Coastline") | (fls.ftype != 566))
            & (fls.streamorde == fls.streamcalc)
        ]
    else:
        fls = fls[(fls.ftype != "Coastline") | (fls.ftype != 566)]
        fls.loc[fls.divergence == 2, "fromnode"] = pd.NA

    if min_path_size > 0:
        short_paths = fls.groupby("levelpathi").apply(
            lambda x: (x.hydroseq == x.hydroseq.min())
            & (x.totdasqkm < min_path_size)
            & (x.totdasqkm >= 0)
        )
        short_paths = short_paths.index.get_level_values("levelpathi")[
            short_paths
        ].tolist()
        fls = fls[~fls.levelpathi.isin(short_paths)]
    terminal_filter = (fls.terminalfl == 1) & (fls.totdasqkm < min_network_size)
    start_filter = (fls.startflag == 1) & (fls.pathlength < min_path_length)
    if any(terminal_filter.dropna()) or any(start_filter.dropna()):
        tiny_networks = fls[terminal_filter].append(fls[start_filter])
        fls = fls[~fls.terminalpa.isin(tiny_networks.terminalpa.unique())]

    n_rm = fl.shape[0] - fls.shape[0]
    if n_rm > 0:
        print(f"Removed {n_rm} rows from the flowlines database.")

    if fls.shape[0] > 0:
        fl_gr = fls.groupby("terminalpa")
        fl_li = [fl_gr.get_group(g) for g in fl_gr.groups]

        def tocomid(ft):
            def toid(row):
                try:
                    return ft[ft.fromnode == row.tonode].comid.values[0]
                except IndexError:
                    return pd.NA

            return ft.apply(toid, axis=1)

        fls["tocomid"] = pd.concat(map(tocomid, fl_li))

    return gpd.GeoDataFrame(pd.merge(fls, fl[extra_cols], on="comid", how="left"))


def accumulate_attr(fl, attr, graph=False):
    """Compute accumulation of an attribute in a river network

    The paths for each node are determined using Breadth-First Search method.

    Parameters
    ----------
    fl : DataFrame
        A dataframe including ID, toID and the attribute columns.
    attr : string
        The column name of an atrribute
    graph : bool
        Whether to return the generated networkx graph object

    Return
    ------
    DataFrame
        The input data frame with an additional column named CumAttr
    """
    import networkx as nx

    req_cols = ["ID", "toID", attr]
    if any(c not in fl.columns for c in req_cols):
        msg = "The required columns are not in the provided flowlines."
        msg += f" The required columns are {', '.join(c for c in req_cols)}"
        raise ValueError(msg)

    G = nx.from_pandas_edgelist(
        fl[req_cols],
        source="ID",
        target="toID",
        edge_attr=attr,
        create_using=nx.DiGraph,
    )
    fl["CumAttr"] = fl.apply(
        lambda x: x[attr]
        + sum(
            [
                G.get_edge_data(f, t)[attr]
                for t, f in nx.bfs_edges(G, x.ID, reverse=True)
            ]
        ),
        axis=1,
    )

    if graph:
        return fl, G
    else:
        return fl


def traverse_json(obj, path):
    """Extracts an element from a JSON file along a specified path.

    Note
    ----
    From `bcmullins <https://bcmullins.github.io/parsing-json-python/>`_

    Parameters
    ----------
    obj : dict
        The input json dictionary
    path : list of strings
        The path to the requested element

    Returns
    -------
    list
    """

    def extract(obj, path, ind, arr):
        key = path[ind]
        if ind + 1 < len(path):
            if isinstance(obj, dict):
                if key in obj.keys():
                    extract(obj.get(key), path, ind + 1, arr)
                else:
                    arr.append(None)
            elif isinstance(obj, list):
                if not obj:
                    arr.append(None)
                else:
                    for item in obj:
                        extract(item, path, ind, arr)
            else:
                arr.append(None)
        if ind + 1 == len(path):
            if isinstance(obj, list):
                if not obj:
                    arr.append(None)
                else:
                    for item in obj:
                        arr.append(item.get(key, None))
            elif isinstance(obj, dict):
                arr.append(obj.get(key, None))
            else:
                arr.append(None)
        return arr

    if isinstance(obj, dict):
        return extract(obj, path, 0, [])
    elif isinstance(obj, list):
        outer_arr = []
        for item in obj:
            outer_arr.append(extract(item, path, 0, []))
        return outer_arr


def cover_statistics(ds):
    """Percentages of the categorical NLCD cover data

    Parameters
    ----------
    ds : xarray.Dataset

    Return
    ------
    dict
        Percentages of classes and categories
    """
    nlcd_meta = helpers.nlcd_helper()
    cover_arr = ds.values
    total_pix = np.count_nonzero(~np.isnan(cover_arr))

    class_percentage = dict(
        zip(
            list(nlcd_meta["classes"].values()),
            [
                cover_arr[cover_arr == int(cat)].shape[0] / total_pix * 100.0
                for cat in list(nlcd_meta["classes"].keys())
            ],
        )
    )

    cat_list = (
        np.array([np.count_nonzero(cover_arr // 10 == c) for c in range(10) if c != 6])
        / total_pix
        * 100.0
    )

    category_percentage = dict(zip(list(nlcd_meta["categories"].keys()), cat_list))

    return {"classes": class_percentage, "categories": category_percentage}


def create_dataset(content, mask, transform, width, height, name, fpath):
    """Create dataset from a response clipped by a geometry

    Parameter
    ---------
    content : requests.Response
        The response to be processed
    geometry : Polygon
        The geometry to clip the data
    name : string
        Variable name in the dataset
    nodata : int or float
        The value to be set for nodata. If set to None, uses the source
        nodata value or if not available, automatically sets NAN for float
        datasets and maximum value in the data type range. For example, if the
        source data has ``uint8`` type nodata is set to 255.
    fpath : string or Path
        The path save the file

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr

    with rio.MemoryFile() as memfile:
        memfile.write(content)
        with memfile.open() as src:
            if src.nodata is None:
                try:
                    nodata = np.iinfo(src.dtypes[0]).max
                except ValueError:
                    nodata = np.nan
            else:
                nodata = np.dtype(src.dtypes[0]).type(src.nodata)

            meta = src.meta
            meta.update(
                {
                    "width": width,
                    "height": height,
                    "transform": transform,
                    "nodata": nodata,
                }
            )

            if fpath is not None:
                with rio.open(fpath, "w", **meta) as dest:
                    dest.write_mask(~mask)
                    dest.write(src.read())

            with rio.vrt.WarpedVRT(src, **meta) as vrt:
                ds = xr.open_rasterio(vrt)
                try:
                    ds = ds.squeeze("band", drop=True)
                except ValueError:
                    pass
                ds = ds.where(~mask, other=vrt.nodata)
                ds.name = name

                ds.attrs["transform"] = transform
                ds.attrs["res"] = (transform[0], transform[4])
                ds.attrs["bounds"] = tuple(vrt.bounds)
                ds.attrs["nodatavals"] = vrt.nodatavals
                ds.attrs["crs"] = vrt.crs
    return ds


def geom_mask(
    geometry, width, height, geo_crs="epsg:4326", ds_crs="epsg:4326", all_touched=True
):
    """Create a mask array and transform for a given geometry.

    Params
    ------
    geometry : Polygon
        A shapely Polygon geometry
    width : int
        x-dimension of the data
    heigth : int
        y-dimension of the data
    geo_crs : string, CRS
        CRS of the geometry, defaults to epsg:4326
    ds_crs : string, CRS
        CRS of the dataset to be masked, defaults to epsg:4326
    all_touched : bool
        Wether to include all the elements where the geometry touchs
        rather than only the element center, defaults to True

    Returns
    -------
    (numpy.ndarray, tuple)
        mask, transform
    """
    if geo_crs != ds_crs:
        geom = rio_warp.transform_geom(geo_crs, ds_crs, mapping(geometry))
        bnds = rio_warp.transform_bounds(geo_crs, ds_crs, *geometry.bounds)
    else:
        geom = geometry
        bnds = geometry.bounds

    transform, _, _ = rio_warp.calculate_default_transform(
        ds_crs, ds_crs, width, height, *bnds
    )
    mask = rio_features.geometry_mask(
        [geom], (height, width), transform, all_touched=all_touched
    )
    return mask, transform
