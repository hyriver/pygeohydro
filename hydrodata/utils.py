#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some utilities for Hydrodata"""

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout


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


def get_data(stations):
    """Instantiate Station class in batch."""
    from hydrodata import Station

    default = dict(
        start=None,
        end=None,
        station_id=None,
        coords=None,
        data_dir="./data",
        rain_snow=False,
        phenology=False,
        width=2000,
        climate=False,
        nlcd=False,
        yreas={"impervious": 2016, "cover": 2016, "canopy": 2016},
    )

    params = list(stations.keys())

    if "station_id" in params and "coords" in params:
        if stations["station_id"] is not None and stations["coords"] is not None:
            raise KeyError("Either coords or station_id should be provided.")

    for k in list(default.keys()):
        if k not in params:
            stations[k] = default[k]

    station = Station(
        start=stations["start"],
        end=stations["end"],
        station_id=stations["station_id"],
        coords=stations["coords"],
        data_dir=stations["data_dir"],
        rain_snow=stations["rain_snow"],
        phenology=stations["phenology"],
        width=stations["width"],
    )

    if stations["climate"]:
        station.get_climate()

    if stations["nlcd"]:
        station.get_nlcd(stations["years"])

    return station.data_dir


def batch(stations):
    """Process queries in batch in parallel.

    Parameters
    ----------
    stations : list of dict
        A list of dictionary containing the input variables:
        [{
        "start" : 'YYYY-MM-DD', [Requaired]
        "end" : 'YYYY-MM-DD', [Requaired]
        "station_id" : '<ID>', OR "coords" : (<lon>, <lat>), [Requaired]
        "data_dir" : '<path/to/store/data>',  [Optional] Default : ./data
        "dem" : True or Flase, [Optional] Default : False
        "climate" : True or Flase, [Optional] Default : False
        "nlcd" : True or False, [Optional] Default : False
        "years" : {'impervious': <YYYY>, 'cover': <YYYY>, 'canopy': <YYYY>}, [Optional] Default is 2016
        "width" : 2000, [Optional] Default : 200
        },
        ...]
    """

    from concurrent import futures

    with futures.ThreadPoolExecutor() as executor:
        data_dirs = list(executor.map(get_data, stations))

    print("All the jobs finished successfully.")
    return data_dirs


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

    url = "https://nationalmap.gov/epqs/pqs.php?"
    session = retry_requests()

    try:
        payload = {"output": "json", "x": lon, "y": lat, "units": "Meters"}
        r = session.get(url, params=payload)
    except HTTPError or ConnectionError or Timeout or RequestException:
        raise
    elevation = r.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"][
        "Elevation"
    ]
    if elevation == -1000000:
        raise ValueError(
            f"The altitude of the requested coordinate ({lon}, {lat}) cannot be found."
        )
    else:
        return elevation


def get_elevation_bybbox(bbox, coords):
    """Get elevation from DEM data for a list of coordinates.

    The elevations are extracted from SRTM1 (30-m resolution) data.
    This function is intended for getting elevations for ds data.

    Parameters
    ----------
    bbox : list
        Bounding box with coordinates in [west, south, east, north] format.
    coords : list of tuples
        A list of coordinates in (lon, lat) foramt to extract the elevation.

    Returns
    -------
    array_like
        A numpy array of elevations in meters
    """

    import rasterio

    west, south, east, north = bbox
    url = "http://opentopo.sdsc.edu/otr/getdem?"
    payload = dict(
        demtype="SRTMGL1",
        west=west,
        south=south,
        east=east,
        north=north,
        outputFormat="GTiff",
    )
    session = retry_requests()
    try:
        r = session.get(url, params=payload)
    except HTTPError or ConnectionError or Timeout or RequestException:
        raise

    with rasterio.MemoryFile() as memfile:
        memfile.write(r.content)
        with memfile.open() as src:
            elevations = np.array([e[0] for e in src.sample(coords)], dtype=np.float32)

    return elevations


def pet_fao(df, lon, lat):
    """Compute Potential EvapoTranspiration using Daymet dataset.

    The method is based on `FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`.
    The following variables are required:
    tmin (deg c), tmax (deg c), lat, lon, vp (Pa), srad (W/m^2), dayl (s)
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

    u_2 = 2.0  # recommended when no data is available

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
