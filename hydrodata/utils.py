#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some utilities for Hydrodata"""

import json
import xml.etree.ElementTree as ET
from itertools import zip_longest
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService
from pqdm.threads import pqdm
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
from shapely.geometry import Point, Polygon, mapping


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
        A session object with the retry setup.
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
    except HTTPError or ConnectionError or Timeout or RequestException:
        raise


def post_url(session, url, payload=None):
    """Retrieve data from a url by POST using a requests session"""
    try:
        return session.post(url, data=payload)
    except HTTPError or ConnectionError or Timeout or RequestException:
        raise


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


def get_elevation(lon, lat):
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


def get_elevation_bybbox(bbox, coords):
    """Get elevation from DEM data for a list of coordinates.

    The elevations are extracted from SRTM1 (30-m resolution) data.
    This function is intended for getting elevations for a gridded dataset.

    Parameters
    ----------
    bbox : list or tuple
        Bounding box with coordinates in [west, south, east, north] format.
    coords : list of tuples
        A list of coordinates in (lon, lat) format to extract the elevations.

    Returns
    -------
    array_like
        A numpy array of elevations in meters
    """

    import rasterio

    if isinstance(bbox, list) or isinstance(bbox, tuple):
        if len(bbox) == 4:
            envelope = dict(zip(["west", "south", "east", "north"], bbox))
        else:
            raise TypeError(
                "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
            )
    else:
        raise TypeError(
            "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
        )

    url = "https://portal.opentopography.org/otr/getdem"

    payload = {"demtype": "SRTMGL1", "outputFormat": "GTiff", **envelope}

    session = retry_requests()
    r = get_url(session, url, payload)

    with rasterio.MemoryFile() as memfile:
        memfile.write(r.content)
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
    elevation = get_elevation(lon, lat)

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
        elevation = get_elevation_bybbox(bbox, coords).reshape(
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
    """Compute monthly mean over the whole time series for the regime curve."""
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


def clip_daymet(ds, geometry):
    """Clip a xarray dataset by a geometry """

    from xarray import apply_ufunc

    if not isinstance(geometry, Polygon):
        raise TypeError("The geometry argument should be of Shapely's Polygon type.")

    def _within(x, y, g):
        return np.array([Point(i, j).within(g) for i, j in np.nditer((x, y))]).reshape(
            x.shape
        )

    def within(da, shape):
        return apply_ufunc(_within, da.lon, da.lat, kwargs={"g": shape})

    return ds.where(within(ds, geometry), drop=True)


def nlcd_helper():
    """Helper for NLCD cover data"""
    import xml.etree.cElementTree as ET

    url = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/metadata/NLCD_2016_Land_Cover_Science_product_L48.xml"
    session = retry_requests()
    r = get_url(session, url)

    root = ET.fromstring(r.content)

    colors = root[4][1][1].text.split("\n")[2:]
    colors = [i.split() for i in colors]
    colors = dict((int(c), (float(r), float(g), float(b))) for c, r, g, b in colors)

    classes = dict(
        (root[4][0][3][i][0][0].text, root[4][0][3][i][0][1].text.split("-")[0].strip())
        for i in range(3, len(root[4][0][3]))
    )

    nlcd_meta = dict(
        impervious_years=[2016, 2011, 2006, 2001],
        canopy_years=[2016, 2011],
        cover_years=[2016, 2013, 2011, 2008, 2006, 2004, 2001],
        classes=classes,
        categories={
            "Unclassified": ("0"),
            "Water": ("11", "12"),
            "Developed": ("21", "22", "23", "24"),
            "Barren": ("31",),
            "Forest": ("41", "42", "43", "45", "46"),
            "Shrubland": ("51", "52"),
            "Herbaceous": ("71", "72", "73", "74"),
            "Planted/Cultivated": ("81", "82"),
            "Wetlands": ("90", "95"),
        },
        roughness={
            "11": 0.0250,
            "12": 0.0220,
            "21": 0.0400,
            "22": 0.1000,
            "23": 0.0800,
            "24": 0.1500,
            "31": 0.0275,
            "41": 0.1600,
            "42": 0.1800,
            "43": 0.1700,
            "45": 0.1000,
            "46": 0.0350,
            "51": 0.1600,
            "52": 0.1000,
            "71": 0.0350,
            "72": 0.0350,
            "81": 0.0325,
            "82": 0.0375,
            "90": 0.1200,
            "95": 0.0700,
        },
        colors=colors,
    )

    return nlcd_meta


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


def get_nhdplus_fcodes():
    """Get NHDPlus FCode lookup table"""
    url = "https://nhd.usgs.gov/userGuide/Robohelpfiles/NHD_User_Guide/Feature_Catalog/Hydrography_Dataset/Complete_FCode_List.htm"
    return pd.concat(pd.read_html(url, header=0)).set_index("FCode")


def get_nwis_errors():
    """Get USGS daily values site web service's error code lookup table"""
    return pd.read_html("https://waterservices.usgs.gov/rest/DV-Service.html")[0]


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


class ArcGISREST:
    """Base class for web services based on ArcGIS REST."""

    def __init__(self, host, site, folder=None, serviceName=None, layer=None):
        """Form the base url and get the service information.

        The general url is in the following form:
        https://<host>/<site>/rest/services/<folder>/<serviceName>/<serviceType>/<layer>/
        or
        https://<host>/<site>/rest/services/<serviceName>/<serviceType>/<layer>/
        For more information visit:
        https://developers.arcgis.com/rest/services-reference/get-started-with-the-services-directory.htm
        """
        self.root = f"https://{host}/{site}/rest/services"
        self.folder = folder
        self.serviceName = serviceName
        self.layer = layer

        self.session = retry_requests()

        self.generate_url()

    def get_layers(self, serviceName=None):
        """Find the available sublayers and their parent layers

        Parameter
        ---------
        serviceName : string, optional
            The geoserver serviceName, defaults to the serviceName instance variable of the class

        Returns
        -------
        dict
            Two dictionaries sublayers and parent_layers
        """
        _, services = self.get_fs(self.folder)
        serviceName = self.serviceName if serviceName is None else serviceName
        if serviceName is None:
            raise ValueError(
                "serviceName should be either passed as an argument "
                + "or be set as a class instance variable"
            )
        try:
            if self.folder is None:
                url = f"{self.root}/{serviceName}/{services[serviceName]}"
            else:
                url = f"{self.root}/{self.folder}/{serviceName}/{services[serviceName]}"
        except KeyError:
            raise KeyError(
                "The serviceName was not found. Check if folder is set correctly."
            )

        info = get_url(self.session, url, {"f": "pjson"}).json()
        try:
            layers = {
                i: n
                for i, n in zip(
                    traverse_json(info, ["layers", "id"]),
                    traverse_json(info, ["layers", "name"]),
                )
            }
            layers.update({-1: "HEAD LAYER"})
        except TypeError:
            raise TypeError(f"The url doesn't have layers.")

        parent_layers = {
            i: p
            for i, p in zip(
                traverse_json(info, ["layers", "id"]),
                traverse_json(info, ["layers", "parentLayerId"]),
            )
        }

        def get_parents(lid):
            lid = [parent_layers[lid]]
            while lid[-1] > -1:
                lid.append(parent_layers[lid[-1]])
            return lid

        parent_layers = {
            i: [layers[l] for l in get_parents(i)[:-1]]
            for i in traverse_json(info, ["layers", "id"])
        }
        sublayers = [
            i
            for i, s in zip(
                traverse_json(info, ["layers", "id"]),
                traverse_json(info, ["layers", "subLayerIds"]),
            )
            if s is None
        ]
        sublayers = {i: layers[i] for i in sublayers}
        return sublayers, parent_layers

    def get_fs(self, folder=None):
        """Get folders and services of the geoserver's (folder) root url"""
        url = self.root if folder is None else f"{self.root}/{folder}"
        info = get_url(self.session, url, {"f": "pjson"}).json()
        try:
            folders = info["folders"]
            services = {
                f"/{s}".split("/")[-1]: t
                for s, t in zip(
                    traverse_json(info, ["services", "name"]),
                    traverse_json(info, ["services", "type"]),
                )
            }
        except TypeError or KeyError:
            raise KeyError(f"The url doesn not include folder and services : {url}")
        return folders, services

    def generate_url(self):
        reqs = {"serviceName": self.serviceName, "layer": self.layer}
        missing = [m for m, v in reqs.items() if v is None]
        if len(missing) > 0:
            self.base_url = None
            msg = (
                "The base_url set to None since the following variables are missing:\n"
            )
            msg += ", ".join(m for m in missing) + "\n"
            msg += "URL's form is https://<host>/<site>/rest/services/<folder>/<serviceName>/<serviceType>/<layer>/\n"
            msg += "Use get_info(<url>) or get_layers(<url>) to get the available options at each level."
            warn(msg)
        elif not isinstance(self.layer, int):
            raise ValueError(
                f"Layer should be an integer. Use self.get_layers('{self.serviceName}') to find the available options."
            )
        else:
            _, services = self.get_fs(self.folder)
            sublayers, _ = self.get_layers()
            self.layer_name = sublayers[self.layer]
            if self.folder is None:
                try:
                    self.base_url = f"{self.root}/{self.serviceName}/{services[self.serviceName]}/{self.layer}"
                except KeyError:
                    raise KeyError(
                        f"The requetsed service is not available on the server: {self.serviceName}"
                    )
            else:
                try:
                    self.base_url = f"{self.root}/{self.folder}/{self.serviceName}/{services[self.serviceName]}/{self.layer}"
                except KeyError:
                    raise KeyError(
                        f"The requetsed service is not available on the server: {self.serviceName}"
                    )
            try:
                self.url_info = get_url(
                    self.session, self.base_url, {"f": "pjson"}
                ).json()
                self.maxRecordCount = self.url_info["maxRecordCount"]
            except KeyError:
                raise KeyError(f"The requested url is not correct: {self.base_url}")


class RESTByGeom(ArcGISREST):
    """For getting data by geometry from an Arc GIS REST sercive."""

    def __init__(
        self, host, site, folder=None, serviceName=None, layer=None, n_threads=4
    ):
        super().__init__(host, site, folder, serviceName, layer)
        self.n_threads = min(n_threads, 8)
        if n_threads > 8:
            warn("No. of threads was reduced to 8.")

    def get_featureids(self, geom):
        if self.base_url is None:
            raise ValueError(
                "The base_url is not set yet, use "
                + "self.generate_url(<layer>) to form the url"
            )

        if isinstance(geom, list) or isinstance(geom, tuple):
            if len(geom) != 4:
                raise TypeError(
                    "The bounding box should be a list or tuple of form [west, south, east, north]"
                )
            geometryType = "esriGeometryEnvelope"
            bbox = dict(zip(["xmin", "ymin", "xmax", "ymax"], geom))
            geom_json = {**bbox, "spatialReference": {"wkid": 4326}}
            geometry = json.dumps(geom_json)
        elif isinstance(geom, Polygon):
            geometryType = "esriGeometryPolygon"
            geom_json = {
                "rings": [[[x, y] for x, y in zip(*geom.exterior.coords.xy)]],
                "spatialReference": {"wkid": 4326},
            }
            geometry = json.dumps(geom_json)
        else:
            raise ValueError("The geometry should be either a bbox (list) or a Polygon")

        payload = {
            "geometryType": geometryType,
            "geometry": geometry,
            "intSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "returnGeometry": "false",
            "returnIdsOnly": "true",
            "f": "geojson",
        }
        r = post_url(self.session, f"{self.base_url}/query", payload)
        try:
            oids = r.json()["objectIds"]
        except KeyError:
            raise KeyError("No feature ID were found on the geoserver within the .")

        oid_list = list(zip_longest(*[iter(oids)] * self.maxRecordCount))
        oid_list[-1] = [i for i in oid_list[-1] if i is not None]
        self.splitted_ids = oid_list

    def get_features(self):
        def _get_features(ids):
            payload = {
                "objectIds": ",".join(str(i) for i in ids),
                "returnGeometry": "true",
                "outSR": "4326",
                "outFields": "*",
                "f": "geojson",
            }
            r = post_url(self.session, f"{self.base_url}/query", payload)
            try:
                return gpd.GeoDataFrame.from_features(r.json(), crs="epsg:4326")
            except TypeError:
                return ids

        feature_list = pqdm(
            self.splitted_ids,
            _get_features,
            n_jobs=self.n_threads,
            desc=f"{self.layer_name}",
        )

        # Find the failed batches and retry
        fails = [ids for ids in feature_list if isinstance(ids, tuple)]
        success = [ids for ids in feature_list if isinstance(ids, gpd.GeoDataFrame)]
        if len(fails) > 0:
            fails = ([x] for y in fails for x in y)
            retry = pqdm(
                fails,
                _get_features,
                n_jobs=min(self.n_threads * 2, 8),
                desc="Retry failed batches",
            )
            success += [ids for ids in retry if isinstance(ids, gpd.GeoDataFrame)]
        return gpd.GeoDataFrame(pd.concat(success))


class Geoserver:
    def __init__(self, root_url, owsType, feature=None, version=None, outFormat=None):
        self.root_url = root_url

        if owsType in ["wfs", "wms"]:
            self.owsType = owsType
            if self.owsType == "wms":
                self.version = "1.1.1" if version is None else version
                f_ows = WebMapService(
                    url=f"{self.root_url}/{owsType}", version=self.version
                )
                self.outFormat = "image/geotiff" if outFormat is None else outFormat
            else:
                self.version = "1.1.0" if version is None else version
                f_ows = WebFeatureService(
                    url=f"{self.root_url}/{owsType}", version=self.version
                )
                self.outFormat = "application/json" if outFormat is None else outFormat
            self.valid_features = {
                k: v
                for v, k in [c.split(":") for c in list(f_ows.contents) if ":" in c]
            }
        else:
            raise ValueError("Acceptable value for owsType are wfs and wms.")

        if feature not in list(self.valid_features.keys()):
            msg = (
                f"The given feature, {feature}, is not valid. The valid features are:\n"
            )
            msg += ", ".join(str(f) for f in list(self.valid_features.keys()))
            raise ValueError(msg)
        else:
            self.feature = feature
        self.base_url = (
            f"{self.root_url}/{self.valid_features[self.feature]}/{self.owsType}"
        )

    def getfeature_byid(self, featurename, featureids, crs="epsg:4326"):
        if self.owsType == "wms":
            raise ValueError("For wms use getmap class function.")

        if not isinstance(featureids, list):
            featureids = [featureids]

        if len(featureids) == 0:
            raise ValueError("The feature ID list is empty!")

        featureids = [str(i) for i in featureids]

        def filter_xml(pname, pid):
            fstart = '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc"><ogc:Or>'
            fend = "</ogc:Or></ogc:Filter>"
            return (
                fstart
                + "".join(
                    [
                        f"<ogc:PropertyIsEqualTo><ogc:PropertyName>{pname}"
                        + f"</ogc:PropertyName><ogc:Literal>{p}"
                        + "</ogc:Literal></ogc:PropertyIsEqualTo>"
                        for p in pid
                    ]
                )
                + fend
            )

        payload = {
            "service": f"{self.owsType}",
            "version": f"{self.version}",
            "outputFormat": f"{self.outFormat}",
            "request": "GetFeature",
            "typeName": f"{self.valid_features[self.feature]}:{self.feature}",
            "srsName": crs,
            "filter": filter_xml(featurename, featureids),
        }

        session = retry_requests()
        r = post_url(session, self.base_url, payload)
        if r.headers["Content-Type"] == "application/xml":
            root = ET.fromstring(r.text)
            raise ValueError(root[0][0].text.strip())

        return r

    def getfeature_bybox(self, bbox, crs="epsg:4326"):
        if self.owsType == "wms":
            raise ValueError("For WMS use getmap class function.")

        if isinstance(bbox, list) or isinstance(bbox, tuple):
            if len(bbox) == 4:
                query = {"bbox": ",".join(f"{b:.06f}" for b in bbox) + f",{crs}"}
            else:
                raise TypeError(
                    "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
                )
        else:
            raise TypeError(
                "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
            )

        payload = {
            "service": f"{self.owsType}",
            "version": f"{self.version}",
            "outputFormat": f"{self.outFormat}",
            "request": "GetFeature",
            "typeName": f"{self.valid_features[self.feature]}:{self.feature}",
            **query,
        }

        session = retry_requests()
        r = get_url(session, self.base_url, payload)
        if r.headers["Content-Type"] == "application/xml":
            root = ET.fromstring(r.text)
            raise ValueError(root[0][0].text.strip())

        return r

    def getmap(self, bbox, width, crs="epsg:4326"):
        if self.owsType == "wfs":
            raise ValueError(
                "For WFS use getfeature_byid or getfeature_bybox class function."
            )

        if isinstance(bbox, list) or isinstance(bbox, tuple):
            if len(bbox) == 4:
                query = {"bbox": ",".join(f"{b:.06f}" for b in bbox)}
            else:
                raise TypeError(
                    "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
                )
        else:
            raise TypeError(
                "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
            )

        height = int(np.abs(bbox[1] - bbox[3]) / np.abs(bbox[0] - bbox[2]) * width)
        payload = {
            "service": f"{self.owsType}",
            "version": f"{self.version}",
            "format": f"{self.outFormat}",
            "request": "GetMap",
            "layers": f"{self.valid_features[self.feature]}:{self.feature}",
            **query,
            "width": width,
            "height": height,
            "srs": crs,
        }

        session = retry_requests()
        r = get_url(session, self.base_url, payload)
        if "xml" in r.headers["Content-Type"]:
            root = ET.fromstring(r.text)
            raise ValueError(root[0].text.strip())

        return r
