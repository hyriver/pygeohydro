#!/usr/bin/env python
"""Some utilities for Hydrodata"""

import numbers
import os
from concurrent import futures
from pathlib import Path
from typing import Iterable
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features as rio_features
import rasterio.warp as rio_warp
import simplejson as json
from owslib.wms import WebMapService
from shapely.geometry import LineString, Point, box, mapping

from hydrodata.connection import RetrySession


def threading(func, iter_list, param_list=[], max_workers=8):
    """Run a function using threading

    Parameters
    ----------
    func : function
        The function to be ran in threads
    iter_list : list
        The iterator for the function
    param_list : list, optional
        List of other parameters, defaults to an empty list
    max_workers : int, optional
        Maximum number of threads, defaults to 8

    Returns
    -------
    list
        A list of function returns for each iterator. The list is not ordered.
    """
    data = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_itr = {
            executor.submit(func, itr, *param_list): itr for itr in iter_list
        }
        for future in futures.as_completed(future_to_itr):
            itr = future_to_itr[future]
            try:
                data.append(future.result())
            except Exception as exc:
                raise Exception(f"{itr}: {exc}")
    return data


def check_dir(fpath_itr):
    """Create parent directory for a file if doesn't exist"""
    if isinstance(fpath_itr, str):
        fpath_itr = [fpath_itr]
    elif not isinstance(fpath_itr, Iterable):
        raise ValueError("Input should be either a string or an iterable object.")

    for f in fpath_itr:
        parent = Path(f).parent
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


def get_ssebopeta_urls(start=None, end=None, years=None):
    """Get list of URLs for SSEBop dataset within a period"""
    if years is None and start is not None and end is not None:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if start < pd.to_datetime("2000-01-01") or end > pd.to_datetime("2018-12-31"):
            raise ValueError("SSEBop database ranges from 2000 till 2018.")
        dates = pd.date_range(start, end)
    elif years is not None and start is None and end is None:
        years = years if isinstance(years, list) else [years]
        seebop_yrs = np.arange(2000, 2019)

        if any(y not in seebop_yrs for y in years):
            raise ValueError("SSEBop database ranges from 2000 till 2018.")

        d_list = [pd.date_range(f"{y}0101", f"{y}1231") for y in years]
        dates = d_list[0] if len(d_list) == 1 else d_list[0].union_many(d_list[1:])
    else:
        raise ValueError("Either years or start and end arguments should be provided.")

    base_url = (
        "https://edcintl.cr.usgs.gov/downloads/sciweb1/"
        + "shared/uswem/web/conus/eta/modis_eta/daily/downloads"
    )
    f_list = [
        (d, f"{base_url}/det{d.strftime('%Y%j')}.modisSSEBopETactual.zip")
        for d in dates
    ]

    return f_list


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
    payload = {"output": "json", "x": lon, "y": lat, "units": "Meters"}
    r = RetrySession().get(url, payload)
    root = r.json()["USGS_Elevation_Point_Query_Service"]
    elevation = root["Elevation_Query"]["Elevation"]
    if elevation == -1000000:
        raise ValueError(
            f"The altitude of the requested coordinate ({lon}, {lat}) cannot be found."
        )

    return elevation


def elevation_bybbox(bbox, resolution, coords, crs="epsg:4326"):
    """Get elevation from DEM data for a list of coordinates.

    This function is intended for getting elevations for a gridded dataset.

    Parameters
    ----------
    bbox : list or tuple
        Bounding box with coordinates in [west, south, east, north] format.
    resolution : float
        Gridded data resolution in arc-second
    coords : list of tuples
        A list of coordinates in (lon, lat) format to extract the elevations.
    crs : str, optional
        The spatial reference system of the input region, defaults to
        epsg:4326.

    Returns
    -------
    numpy.ndarray
        An array of elevations in meters
    """
    import pyproj
    import shapely.ops as ops

    if isinstance(bbox, (list, tuple)):
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

    if crs != "epsg:4326":
        prj = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
        _bbox = ops.transform(prj.transform, box(*bbox))
        res_bbox = _bbox.bounds
    else:
        res_bbox = bbox

    west, south, east, north = res_bbox
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


def pet_fao_byloc(clm, lon, lat):
    """Compute Potential EvapoTranspiration using Daymet dataset for a single location.

    The method is based on `FAO-56 <http://www.fao.org/docrep/X0490E/X0490E00.htm>`.

    Parameters
    ----------
    clm : DataFrame
        A dataframe with columns named as follows:
        ``tmin (deg c)``, ``tmax (deg c)``, ``vp (Pa)``, ``srad (W/m^2)``, ``dayl (s)``
    lon : float
        Longitude of the location of interest
    lat : float
        Latitude of the location of interest

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column named ``pet (mm/day)``
    """

    reqs = ["tmin (deg c)", "tmax (deg c)", "vp (Pa)", "srad (W/m^2)", "dayl (s)"]

    check_requirements(reqs, clm.columns)

    dtype = clm.dtypes[0]
    clm["tmean (deg c)"] = 0.5 * (clm["tmax (deg c)"] + clm["tmin (deg c)"])
    Delta = (
        4098
        * (
            0.6108
            * np.exp(
                17.27 * clm["tmean (deg c)"] / (clm["tmean (deg c)"] + 237.3),
                dtype=dtype,
            )
        )
        / ((clm["tmean (deg c)"] + 237.3) ** 2)
    )
    elevation = elevation_byloc(lon, lat)

    P = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
    gamma = P * 0.665e-3

    G = 0.0  # recommended for daily data
    clm["vp (Pa)"] = clm["vp (Pa)"] * 1e-3

    e_max = 0.6108 * np.exp(
        17.27 * clm["tmax (deg c)"] / (clm["tmax (deg c)"] + 237.3), dtype=dtype
    )
    e_min = 0.6108 * np.exp(
        17.27 * clm["tmin (deg c)"] / (clm["tmin (deg c)"] + 237.3), dtype=dtype
    )
    e_s = (e_max + e_min) * 0.5
    e_def = e_s - clm["vp (Pa)"]

    u_2 = 2.0  # recommended when no data is available

    jday = clm.index.dayofyear
    R_s = clm["srad (W/m^2)"] * clm["dayl (s)"] * 1e-6

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
            ((clm["tmax (deg c)"] + 273.16) ** 4 + (clm["tmin (deg c)"] + 273.16) ** 4)
            * 0.5
        )
        * (0.34 - 0.14 * np.sqrt(clm["vp (Pa)"]))
        * ((1.35 * R_s / R_so) - 0.35)
    )
    R_n = R_ns - R_nl

    clm["pet (mm/day)"] = (
        0.408 * Delta * (R_n - G)
        + gamma * 900.0 / (clm["tmean (deg c)"] + 273.0) * u_2 * e_def
    ) / (Delta + gamma * (1 + 0.34 * u_2))
    clm["vp (Pa)"] = clm["vp (Pa)"] * 1.0e3

    return clm


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

    keys = list(ds.keys())
    reqs = ["tmin", "tmax", "lat", "lon", "vp", "srad", "dayl"]

    check_requirements(reqs, keys)

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

    month_abbr = dict(enumerate(calendar.month_abbr))
    mean_month = daily.groupby(daily.index.month).mean()
    mean_month.index = mean_month.index.map(month_abbr)
    return mean_month


def exceedance(daily):
    """Compute Flow duration (rank, sorted obs).

    The zero discharges are handled by dropping since log 0 is undefined.
    """

    if not isinstance(daily, pd.Series):
        raise TypeError("The input should be of type pandas Series.")

    rank = daily.rank(ascending=False, pct=True) * 100
    fdc = pd.concat([daily, rank], axis=1)
    fdc.columns = ["Q", "rank"]
    fdc = fdc.sort_values(by=["rank"]).set_index("rank", drop=True)
    return fdc


def check_columns(df, req_cols):
    """Check if a dataframe has a list of required columns

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas DataFrame.
    req_cols : list or tuple
        A list of required column names

    Raises
    ------
    ValueError
        Shows the missing columns
    """
    missing = [c for c in req_cols if c not in df]
    if missing:
        raise ValueError(
            "The following columns are missing:\n" + f"{', '.join(m for m in missing)}"
        )


def prepare_nhdplus(
    flw,
    min_network_size,
    min_path_length,
    min_path_size=0,
    purge_non_dendritic=False,
    verbose=False,
):
    """Cleaning up and fixing issue in NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`_

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        COMID, LENGTHKM, FTYPE, TerminalFl, FromNode, ToNode, TotDASqKM,
        StartFlag, StreamOrde, StreamCalc, TerminalPa, Pathlength,
        Divergence, Hydroseq, LevelPathI
    min_network_size : float
        Minimum size of drainage network in sqkm
    min_path_length : float
        Minimum length of terminal level path of a network in km.
    min_path_size : float, optional
        Minimum size of outlet level path of a drainage basin in km.
        Drainage basins with an outlet drainage area smaller than
        this value will be removed. Defaults to 0.
    purge_non_dendritic : bool, optional
        Whether to remove non dendritic paths, defaults to False
    verbose : bool, optional
        Whether to show a message about the removed features, defaults to True.

    Returns
    -------
    geopandas.GeoDataFrame
        Note that all column names are converted to lower case.
    """

    flw.columns = flw.columns.str.lower()
    nrows = flw.shape[0]

    req_cols = [
        "comid",
        "terminalfl",
        "terminalpa",
        "hydroseq",
        "streamorde",
        "streamcalc",
        "divergence",
        "fromnode",
        "ftype",
    ]

    check_columns(flw, req_cols)
    flw[req_cols[:-1]] = flw[req_cols[:-1]].astype("Int64")

    if not any(flw.terminalfl == 1):
        if all(flw.terminalpa == flw.terminalpa.iloc[0]):
            flw.loc[flw.hydroseq == flw.hydroseq.min(), "terminalfl"] = 1
        else:
            raise ValueError("No terminal flag were found in the dataframe.")

    if purge_non_dendritic:
        flw = flw[
            ((flw.ftype != "Coastline") | (flw.ftype != 566))
            & (flw.streamorde == flw.streamcalc)
        ]
    else:
        flw = flw[(flw.ftype != "Coastline") | (flw.ftype != 566)]
        flw.loc[flw.divergence == 2, "fromnode"] = pd.NA

    flw = remove_tinynetworks(flw, min_path_size, min_path_length, min_network_size)

    if verbose:
        print(f"Removed {nrows - flw.shape[0]} paths from the flowlines.")

    if flw.shape[0] > 0:
        flw = add_tocomid(flw)

    return flw


def remove_tinynetworks(flw, min_path_size, min_path_length, min_network_size):
    """Remove small paths in NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`_

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        levelpathi, hydroseq, totdasqkm, terminalfl, startflag,
        pathlength, terminalpa
    min_network_size : float
        Minimum size of drainage network in sqkm.
    min_path_length : float
        Minimum length of terminal level path of a network in km.
    min_path_size : float
        Minimum size of outlet level path of a drainage basin in km.
        Drainage basins with an outlet drainage area smaller than
        this value will be removed.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    req_cols = [
        "levelpathi",
        "hydroseq",
        "terminalfl",
        "startflag",
        "terminalpa",
        "totdasqkm",
        "pathlength",
    ]
    check_columns(flw, req_cols)

    flw[req_cols[:-2]] = flw[req_cols[:-2]].astype("Int64")

    if min_path_size > 0:
        short_paths = flw.groupby("levelpathi").apply(
            lambda x: (x.hydroseq == x.hydroseq.min())
            & (x.totdasqkm < min_path_size)
            & (x.totdasqkm >= 0)
        )
        short_paths = short_paths.index.get_level_values("levelpathi")[
            short_paths
        ].tolist()
        flw = flw[~flw.levelpathi.isin(short_paths)]

    terminal_filter = (flw.terminalfl == 1) & (flw.totdasqkm < min_network_size)
    start_filter = (flw.startflag == 1) & (flw.pathlength < min_path_length)

    if any(terminal_filter.dropna()) or any(start_filter.dropna()):
        tiny_networks = flw[terminal_filter].append(flw[start_filter])
        flw = flw[~flw.terminalpa.isin(tiny_networks.terminalpa.unique())]

    return flw


def add_tocomid(flw):
    """Find the downstream comid(s) of each comid in NHDPlus flowline database.

    Ported from `nhdplusTools <https://github.com/USGS-R/nhdplusTools>`_

    Parameters
    ----------
    flw : geopandas.GeoDataFrame
        NHDPlus flowlines with at least the following columns:
        comid, terminalpa, fromnode, tonode

    Returns
    -------
    geopandas.GeoDataFrame
        The input dataframe With an additional column named ``tocomid``.
    """

    req_cols = ["comid", "terminalpa", "fromnode", "tonode"]
    check_columns(flw, req_cols)

    flw[req_cols] = flw[req_cols].astype("Int64")

    def tocomid(group):
        def toid(row):
            try:
                return group[group.fromnode == row.tonode].comid.to_numpy()[0]
            except IndexError:
                return pd.NA

        return group.apply(toid, axis=1)

    flw["tocomid"] = pd.concat([tocomid(g) for _, g in flw.groupby("terminalpa")])
    return flw


def traverse_json(obj, path):
    """Extracts an element from a JSON file along a specified path.

    Notes
    -----
    From `bcmullins <https://bcmullins.github.io/parsing-json-python/>`_

    Parameters
    ----------
    obj : dict
        The input json dictionary
    path : list of strs
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

    if isinstance(obj, list):
        outer_arr = []
        for item in obj:
            outer_arr.append(extract(item, path, 0, []))
        return outer_arr


def cover_statistics(ds):
    """Percentages of the categorical NLCD cover data

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    dict
    """
    from hydrodata.helpers import nlcd_helper

    nlcd_meta = nlcd_helper()
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

    Parameters
    ----------
    content : requests.Response
        The response to be processed
    mask : numpy.ndarray
        The mask to clip the data
    transform : tuple
        Transform of the mask
    width : int
        x-dimension of the data
    heigth : int
        y-dimension of the data
    name : str
        Variable name in the dataset
    fpath : str or Path
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
                ds.attrs["crs"] = vrt.crs.to_string()
    return ds


def json_togeodf(content, in_crs, crs="epsg:4326"):
    """Create GeoDataFrame from (Geo)JSON

    Parameters
    ----------
    content : dict
        A (Geo)JSON dictionary e.g., r.json()
    in_crs : str
        CRS of the content
    crs : str, optional
        CRS of the output GeoDataFrame, defaults to ``epsg:4326``

    Returns
    -------
    geopandas.GeoDataFrame
    """
    try:
        geodf = gpd.GeoDataFrame.from_features(content, crs=in_crs)
    except TypeError:
        geodf = gpd.GeoDataFrame.from_features(arcgis_togeojson(content), crs=in_crs)

    geodf.crs = in_crs
    if in_crs != crs:
        geodf = geodf.to_crs(crs)
    return geodf


def arcgis_togeojson(arcgis, idAttribute=None):
    """Convert ESRIGeoJSON format to GeoJSON.

    Notes
    -----
    Based on https://github.com/chris48s/arcgis_togeojson

    Parameters
    ----------
    arcgis : str or binary
        The ESRIGeoJSON format str (or binary)
    idAttribute : str
        ID of the attribute of interest

    Returns
    -------
    dict
        A GeoJSON file readable by GeoPandas
    """

    def convert(arcgis, idAttribute=None):
        """Convert an ArcGIS JSON object to a GeoJSON object"""
        geojson = {}

        if "features" in arcgis and arcgis["features"]:
            geojson["type"] = "FeatureCollection"
            geojson["features"] = []
            geojson["features"] = [
                convert(feature, idAttribute) for feature in arcgis["features"]
            ]

        if (
            "x" in arcgis
            and isinstance(arcgis["x"], numbers.Number)
            and "y" in arcgis
            and isinstance(arcgis["y"], numbers.Number)
        ):
            geojson["type"] = "Point"
            geojson["coordinates"] = [arcgis["x"], arcgis["y"]]
            if "z" in arcgis and isinstance(arcgis["z"], numbers.Number):
                geojson["coordinates"].append(arcgis["z"])

        if "points" in arcgis:
            geojson["type"] = "MultiPoint"
            geojson["coordinates"] = arcgis["points"]

        if "paths" in arcgis:
            if len(arcgis["paths"]) == 1:
                geojson["type"] = "LineString"
                geojson["coordinates"] = arcgis["paths"][0]
            else:
                geojson["type"] = "MultiLineString"
                geojson["coordinates"] = arcgis["paths"]

        if "rings" in arcgis:
            geojson = rings_togeojson(arcgis["rings"])

        if (
            "xmin" in arcgis
            and isinstance(arcgis["xmin"], numbers.Number)
            and "ymin" in arcgis
            and isinstance(arcgis["ymin"], numbers.Number)
            and "xmax" in arcgis
            and isinstance(arcgis["xmax"], numbers.Number)
            and "ymax" in arcgis
            and isinstance(arcgis["ymax"], numbers.Number)
        ):
            geojson["type"] = "Polygon"
            geojson["coordinates"] = [
                [
                    [arcgis["xmax"], arcgis["ymax"]],
                    [arcgis["xmin"], arcgis["ymax"]],
                    [arcgis["xmin"], arcgis["ymin"]],
                    [arcgis["xmax"], arcgis["ymin"]],
                    [arcgis["xmax"], arcgis["ymax"]],
                ]
            ]

        if "geometry" in arcgis or "attributes" in arcgis:
            geojson["type"] = "Feature"
            if "geometry" in arcgis:
                geojson["geometry"] = convert(arcgis["geometry"])
            else:
                geojson["geometry"] = None

            if "attributes" in arcgis:
                geojson["properties"] = arcgis["attributes"]
                try:
                    attributes = arcgis["attributes"]
                    keys = (
                        [idAttribute, "OBJECTID", "FID"]
                        if idAttribute
                        else ["OBJECTID", "FID"]
                    )
                    for key in keys:
                        if key in attributes and (
                            isinstance(attributes[key], (numbers.Number, str))
                        ):
                            geojson["id"] = attributes[key]
                            break
                except KeyError:
                    warn("No valid id attribute found.")
            else:
                geojson["properties"] = None

        if "geometry" in geojson and not geojson["geometry"]:
            geojson["geometry"] = None

        return geojson

    def rings_togeojson(rings):
        """Checks for holes in the ring and fill them"""

        outerRings = []
        holes = []
        x = None  # iterator
        outerRing = None  # current outer ring being evaluated
        hole = None  # current hole being evaluated

        for ring in rings:
            if not all(np.isclose(ring[0], ring[-1])):
                ring.append(ring[0])

            if len(ring) < 4:
                continue

            total = sum(
                (pt2[0] - pt1[0]) * (pt2[1] + pt1[1])
                for pt1, pt2 in zip(ring[:-1], ring[1:])
            )
            # Clock-wise check
            if total >= 0:
                outerRings.append(
                    [ring[::-1]]
                )  # wind outer rings counterclockwise for RFC 7946 compliance
            else:
                holes.append(
                    ring[::-1]
                )  # wind inner rings clockwise for RFC 7946 compliance

        uncontainedHoles = []

        # while there are holes left...
        while holes:
            # pop a hole off out stack
            hole = holes.pop()

            # loop over all outer rings and see if they contain our hole.
            contained = False
            x = len(outerRings) - 1
            while x >= 0:
                outerRing = outerRings[x][0]
                l1, l2 = LineString(outerRing), LineString(hole)
                p2 = Point(hole[0])
                intersects = l1.intersects(l2)
                contains = l1.contains(p2)
                if not intersects and contains:
                    # the hole is contained push it into our polygon
                    outerRings[x].append(hole)
                    contained = True
                    break
                x = x - 1

            # ring is not contained in any outer ring
            # sometimes this happens https://github.com/Esri/esri-leaflet/issues/320
            if not contained:
                uncontainedHoles.append(hole)

        # if we couldn't match any holes using contains we can try intersects...
        while uncontainedHoles:
            # pop a hole off out stack
            hole = uncontainedHoles.pop()

            # loop over all outer rings and see if any intersect our hole.
            intersects = False
            x = len(outerRings) - 1
            while x >= 0:
                outerRing = outerRings[x][0]
                l1, l2 = LineString(outerRing), LineString(hole)
                intersects = l1.intersects(l2)
                if intersects:
                    # the hole is contained push it into our polygon
                    outerRings[x].append(hole)
                    intersects = True
                    break
                x = x - 1

            if not intersects:
                outerRings.append([hole[::-1]])

        if len(outerRings) == 1:
            return {"type": "Polygon", "coordinates": outerRings[0]}

        return {"type": "MultiPolygon", "coordinates": outerRings}

    if isinstance(arcgis, str):
        return json.dumps(convert(json.loads(arcgis), idAttribute))

    return convert(arcgis, idAttribute)


def geom_mask(
    geometry, width, height, geo_crs="epsg:4326", ds_crs="epsg:4326", all_touched=True
):
    """Create a mask array and transform for a given geometry.

    Parameters
    ----------
    geometry : Polygon
        A shapely Polygon geometry
    width : int
        x-dimension of the data
    heigth : int
        y-dimension of the data
    geo_crs : str, CRS
        CRS of the geometry, defaults to epsg:4326
    ds_crs : str, CRS
        CRS of the dataset to be masked, defaults to epsg:4326
    all_touched : bool
        Wether to include all the elements where the geometry touchs
        rather than only the element's center, defaults to True

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


def check_requirements(reqs, cols):
    """Check for all the required data.

    Parameters
    ----------
    reqs : list
        A list of required data names as strs
    cols : list
        A list of data names as strs
    """

    if not isinstance(reqs, Iterable):
        raise ValueError("Inputs should be list of strs")

    missing = [r for r in reqs if r not in cols]
    if len(missing) > 1:
        raise ValueError(
            "The following required data are missing:\n" + ", ".join(m for m in missing)
        )


def topoogical_sort(flowlines):
    """Topological sorting of a river network.

    Parameters
    ----------
    flowlines : pandas.DataFrame
        A dataframe with columns ID and toID

    Returns
    -------
    (list, dict , networkx.DiGraph)
        A list of topologically sorted IDs, a dictionary
        with keys as IDs and values as its upstream nodes,
        and the generated networkx object. Note that the
        terminal node ID is set to pd.NA.
    """
    import networkx as nx

    upstream_nodes = {
        i: flowlines[flowlines.toID == i].ID.tolist() for i in flowlines.ID.tolist()
    }
    upstream_nodes[pd.NA] = flowlines[flowlines.toID.isna()].ID.tolist()

    G = nx.from_pandas_edgelist(
        flowlines[["ID", "toID"]], source="ID", target="toID", create_using=nx.DiGraph,
    )
    topo_sorted = list(nx.topological_sort(G))
    return topo_sorted, upstream_nodes, G


def vector_accumulation(
    flowlines, func, attr_col, arg_cols, id_col="comid", toid_col="tocomid",
):
    """Flow accumulation using vector river network data.

    Parameters
    ----------
    flowlines : pandas.DataFrame
        A dataframe containing comid, tocomid, attr_col and all the columns
        that ara required for passing to ``func``.
    func : function
        The function that routes the flow in a signle river segment.
        Positions of the arguments in the function should be as follows:
        ``func(qin, *arg_cols)``
        ``qin`` is computed in this function and the rest are in the order
        of the ``arg_cols``. For example, if ``arg_cols = ["slope", "roughness"]``
        then the functions is called this way:
        func(qin, slope, roughness)
        where slope and roughness are elemental values read from the flowlines.
    attr_col : str
        The column name of the attribute being accumulated in the network.
        The column should contain the initial condition for the attribute for
        each river segment. It can be a scalar or an array (e.g., time series).
    arg_cols : list of strs
        List of the flowlines columns that contain all the required
        data for a routing a single river segment such as slope, length,
        lateral flow, etc.
    id_name : str, optional
        Name of the flowlines column containing IDs, defaults to comid
    toid_name : str, optional
        Name of the flowlines column containing toIDs, defaults to tocomid

    Returns
    -------
    pandas.Series
        Accumulated flow for all the nodes. The dataframe is sorted from upstream
        to downstream (topological sorting). Depending on the given initial
        condition in the attr_col, the outflow for each river segment can be
        a scalar or an array.
    """

    sorted_nodes, upstream_nodes, _ = topoogical_sort(
        flowlines[[id_col, toid_col]].rename(columns={id_col: "ID", toid_col: "toID"})
    )
    topo_sorted = sorted_nodes[:-1]

    outflow = flowlines.set_index(id_col)[attr_col].to_dict()

    init = flowlines.iloc[0][attr_col]
    if isinstance(init, numbers.Number):
        outflow[0] = 0.0
    elif isinstance(init, (np.ndarray, list)):
        outflow[0] = np.zeros_like(init)
    else:
        raise ValueError(
            "The elements in the attribute column can be either scalars or arrays"
        )

    upstream_nodes.update({k: [0] for k, v in upstream_nodes.items() if len(v) == 0})

    for i in topo_sorted:
        outflow[i] = func(
            np.sum([outflow[u] for u in upstream_nodes[i]], axis=0),
            *flowlines.loc[flowlines[id_col] == i, arg_cols].to_numpy()[0],
        )

    outflow.pop(0)
    qsim = pd.Series(outflow).loc[sorted_nodes[:-1]]
    qsim = qsim.rename_axis("comid").rename("acc")
    return qsim
