#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accessing data from the supported databases through their APIs."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.mask
from hydrodata import utils

import xarray as xr

MARGINE = 15


def nwis_streamflow(station_ids, start, end, raw=False):
    """Get daily streamflow observations from USGS.

    Parameters
    ----------
    station_ids : string, list
        The gage ID(s)  of the USGS station
    start : string or datetime
        Start date
    end : string or datetime
        End date
    raw : bool
        Whether to return the raw data without cleanup as a Dataframe or
        remove all the columns except for ``qobs`` as a Series, default to
        False.

    Returns
    -------
        qobs : dataframe
        Streamflow data observations in cubic meter per second (cms)
    """

    if isinstance(station_ids, str):
        station_ids = [station_ids]
    elif isinstance(station_ids, list):
        station_ids = [str(i) for i in station_ids]
    else:
        raise ValueError(
            "the ids argument should be either a string or a list or strings"
        )
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    siteinfo = nwis_siteinfo(station_ids)
    check_dates = siteinfo.loc[
        (siteinfo.stat_cd == "00003")
        & (start < siteinfo.begin_date)
        & (end > siteinfo.end_date),
        "site_no",
    ].tolist()
    nas = [s for s in station_ids if s in check_dates]
    if len(nas) > 0:
        msg = "Daily Mean data unavailable for the specified time period for the following stations:\n"
        msg += ", ".join(str(s) for s in nas)
        raise ValueError(msg)

    url = "https://waterservices.usgs.gov/nwis/dv"
    payload = {
        "format": "json",
        "sites": ",".join(str(s) for s in station_ids),
        "startDT": start.strftime("%Y-%m-%d"),
        "endDT": end.strftime("%Y-%m-%d"),
        "parameterCd": "00060",
        "statCd": "00003",
        "siteStatus": "all",
    }

    session = utils.retry_requests()
    r = utils.post_url(session, url, payload)

    ts = r.json()["value"]["timeSeries"]
    r_ts = {
        t["sourceInfo"]["siteCode"][0]["value"]: t["values"][0]["value"]
        for t in ts["value"]["timeSeries"]
    }

    def to_df(col, dic):
        q = pd.DataFrame.from_records(dic, exclude=["qualifiers"], index=["dateTime"])
        q.index = pd.to_datetime(q.index)
        q.columns = [col]
        q[col] = q[col].astype("float64") * 0.028316846592  # Convert cfs to cms
        return q

    qobs = pd.concat([to_df(f"USGS-{s}", t) for s, t in r_ts.items()], axis=1)
    return qobs


def nwis_siteinfo(ids=None, bbox=None, expanded=False):
    """Get NWIS stations by a list of IDs or within a bounding box.

    Only stations that record(ed) daily streamflow data are returned.
    The following columns are included in the dataframe:
    site_no         -- Site identification number
    station_nm      -- Site name
    site_tp_cd      -- Site type
    dec_lat_va      -- Decimal latitude
    dec_long_va     -- Decimal longitude
    coord_acy_cd    -- Latitude-longitude accuracy
    dec_coord_datum_cd -- Decimal Latitude-longitude datum
    alt_va          -- Altitude of Gage/land surface
    alt_acy_va      -- Altitude accuracy
    alt_datum_cd    -- Altitude datum
    huc_cd          -- Hydrologic unit code
    parm_cd         -- Parameter code
    stat_cd         -- Statistical code
    ts_id           -- Internal timeseries ID
    loc_web_ds      -- Additional measurement description
    medium_grp_cd   -- Medium group code
    parm_grp_cd     -- Parameter group code
    srs_id          -- SRS ID
    access_cd       -- Access code
    begin_date      -- Begin date
    end_date        -- End date
    count_nu        -- Record count

    Parameters
    ----------
    ids : string or list of strings
        Station ID(s)
    bbox : list
        List of corners in this order [west, south, east, north]
    expanded : bool, optional
        Wether to get expanded sit information for example drainage area.

    Returns
    -------
    pandas.DataFrame
    """
    if bbox is not None and ids is None:
        if isinstance(bbox, list) or isinstance(bbox, tuple):
            if len(bbox) == 4:
                query = {"bBox": ",".join(f"{b:.06f}" for b in bbox)}
            else:
                raise TypeError(
                    "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
                )
        else:
            raise TypeError(
                "The bounding box should be a list or tuple of length 4: [west, south, east, north]"
            )

    elif ids is not None and bbox is None:
        if isinstance(ids, str):
            query = {"sites": ids}
        elif isinstance(ids, list):
            query = {"sites": ",".join(str(i) for i in ids)}
        else:
            raise ValueError(
                "the ids argument should be either a string or a list or strings"
            )
    else:
        raise ValueError("Either ids or bbox argument should be provided.")

    url = "https://waterservices.usgs.gov/nwis/site"

    if expanded:
        outputType = {"siteOutput": "expanded"}
    else:
        outputType = {"outputDataTypeCd": "dv"}

    payload = {
        **query,
        **outputType,
        "format": "rdb",
        "parameterCd": "00060",
        "siteStatus": "all",
        "hasDataTypeCd": "dv",
    }

    session = utils.retry_requests()
    r = utils.post_url(session, url, payload)

    r_text = r.text.split("\n")
    r_list = [l.split("\t") for l in r_text if "#" not in l]
    r_dict = [dict(zip(r_list[0], st)) for st in r_list[2:]]

    df = pd.DataFrame.from_dict(r_dict).dropna()
    df = df.drop(df[df.alt_va == ""].index)
    try:
        df = df[df.parm_cd == "00060"]
        df["begin_date"] = pd.to_datetime(df["begin_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
    except AttributeError:
        pass

    df[["dec_lat_va", "dec_long_va", "alt_va"]] = df[
        ["dec_lat_va", "dec_long_va", "alt_va"]
    ].astype("float64")

    df = df[df.site_no.apply(len) == 8]

    return df


def nwis_basin(station_ids):
    """Get USGS stations basins using NLDI service.

    Parameters
    ----------
    station_ids : string or list
        The NWIS stations ID(s).

    Returns
    -------
    GeoDataFrame
    """
    if not isinstance(station_ids, list):
        station_ids = [station_ids]

    station_ids = ["USGS-" + str(f) for f in station_ids]

    if len(station_ids) == 0:
        raise ValueError("The featureID list is empty!")

    crs = "epsg:4326"
    base_url = "https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite"
    session = utils.retry_requests()

    def get_url(sid):
        url = f"{base_url}/{sid}/basin"
        r = utils.get_url(session, url)
        gdf = gpd.GeoDataFrame.from_features(r.json(), crs=crs)
        gdf["site_no"] = sid
        return gdf

    gdf = gpd.GeoDataFrame(pd.concat([get_url(i) for i in station_ids]))
    gdf.crs = crs

    return gdf


def deymet_byloc(lon, lat, start=None, end=None, years=None, variables=None, pet=False):
    """Get daily climate data from Daymet for a single point.

    Parameters
    ----------
    lon : float
        Longitude of the point of interest
    lat : float
        Latitude of the point of interest
    start : string or datetime
        Start date
    end : string or datetime
        End date
    years : list
        List of years
    variables : string or list
        List of variables to be downloaded. The acceptable variables are:
        tmin, tmax, prcp, srad, vp, swe, dayl
        Descriptions can be found in https://daymet.ornl.gov/overview
    pet : bool
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`_.
        The default is False

    Returns
    -------
    Pandas DataFrame
        Climate data for the requested location and variables
    """

    if not (14.5 < lat < 52.0) or not (-131.0 < lon < -53.0):
        msg = "The location is outside the Daymet dataset. The acceptable range is: "
        msg += "14.5 < lat < 52.0 and -131.0 < lon < -53.0"
        raise ValueError(msg)

    if years is None and start is not None and end is not None:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if start < pd.to_datetime("1980-01-01"):
            raise ValueError("Daymet database ranges from 1980 till present.")
    elif years is not None and start is None and end is None:
        years = years if isinstance(years, list) else [years]
    else:
        raise ValueError("Either years or start and end arguments should be provided.")

    vars_table = pd.read_html("https://daymet.ornl.gov/overview")[1]
    valid_variables = vars_table.Abbr.values

    if variables is not None:
        variables = variables if isinstance(variables, list) else [variables]

        invalid = [v for v in variables if v not in valid_variables]
        if len(invalid) > 0:
            msg = "These required variables are not in the dataset: "
            msg += ", ".join(x for x in invalid)
            msg += f'\nRequired variables are {", ".join(x for x in valid_variables)}'
            raise KeyError(msg)

        if pet:
            reqs = ["tmin", "tmax", "vp", "srad", "dayl"]
            variables = list(set(reqs) | set(variables))
    else:
        variables = valid_variables

    print(
        f"[LOC: ({lon:.2f}, {lat:.2f})] ".ljust(MARGINE)
        + "Downloading climate data from Daymet",
        end=" >>> ",
    )

    url = "https://daymet.ornl.gov/single-pixel/api/data"
    if years is None:
        dates = {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}
    else:
        dates = {"years": ",".join(str(x) for x in years)}

    payload = {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "vars": ",".join(x for x in variables),
        "format": "json",
        **dates,
    }

    session = utils.retry_requests()
    r = utils.get_url(session, url, payload)

    df = pd.DataFrame(r.json()["data"])
    df.index = pd.to_datetime(df.year * 1000.0 + df.yday, format="%Y%j")
    df.drop(["year", "yday"], axis=1, inplace=True)

    print("finished.")

    if pet:
        print(
            f"[LOC: ({lon:.2f}, {lat:.2f})] ".ljust(MARGINE) + "Computing PET",
            end=" >>> ",
        )

        df = utils.pet_fao(df, lon, lat)

        print("finished.")

    return df


def daymet_bygeom(
    geometry,
    start=None,
    end=None,
    years=None,
    variables=None,
    pet=False,
    resolution=None,
):
    """Gridded data from the Daymet database.

    The data is clipped using netCDF Subset Service.

    Parameters
    ----------
    geometry : Geometry
        The geometry for downloading clipping the data. For a box geometry,
        the order should be as follows:
        geom = box(west, southe, east, north)
    start : string or datetime
        Starting date
    end : string or datetime
        Ending date
    years : list
        List of years
    variables : string or list
        List of variables to be downloaded. The acceptable variables are:
        tmin, tmax, prcp, srad, vp, swe, dayl
        Descriptions can be found in https://daymet.ornl.gov/overview
    pet : bool
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`_.
        The default is False
    resolution : float
        The desired output resolution for the output in km,
        defaults to no resampling. The resampling is done using bilinear method

    Returns
    -------
    xarray.DataArray
        The climate data within the requested geometery.
    """

    from pandas.tseries.offsets import DateOffset
    from shapely.geometry import Polygon

    base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/"

    if years is None and start is not None and end is not None:
        start = pd.to_datetime(start) + DateOffset(hour=12)
        end = pd.to_datetime(end) + DateOffset(hour=12)
        if start < pd.to_datetime("1980-01-01"):
            raise ValueError("Daymet database ranges from 1980 till 2018.")
        dates = utils.daymet_dates(start, end)
    elif years is not None and start is None and end is None:
        years = years if isinstance(years, list) else [years]
        start_list, end_list = [], []
        for year in years:
            s = pd.to_datetime(f"{year}0101")
            start_list.append(s + DateOffset(hour=12))
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                end_list.append(pd.to_datetime(f"{year}1230") + DateOffset(hour=12))
            else:
                end_list.append(pd.to_datetime(f"{year}1231") + DateOffset(hour=12))
        dates = [(s, e) for s, e in zip(start_list, end_list)]
    else:
        raise ValueError("Either years or start and end arguments should be provided.")

    variables = variables if isinstance(variables, list) else [variables]

    vars_table = pd.read_html("https://daymet.ornl.gov/overview")[1]
    units = dict(zip(vars_table["Abbr"], vars_table["Units"]))
    valid_variables = vars_table.Abbr.values

    if variables is not None:
        variables = variables if isinstance(variables, list) else [variables]

        invalid = [v for v in variables if v not in valid_variables]
        if len(invalid) > 0:
            msg = "These required variables are not in the dataset: "
            msg += ", ".join(x for x in invalid)
            msg += f'\nRequired variables are {", ".join(x for x in valid_variables)}'
            raise KeyError(msg)

        if pet:
            reqs = ["tmin", "tmax", "vp", "srad", "dayl"]
            variables = list(set(reqs) | set(variables))
    else:
        variables = valid_variables

    if not isinstance(geometry, Polygon):
        raise TypeError("The geometry argument should be of Shapely's Polygon type.")

    print(
        f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(MARGINE)
        + "Downloading climate data from Daymet",
        end=" >>> ",
    )

    west, south, east, north = np.round(geometry.bounds, 6)
    urls = []
    for s, e in dates:
        for v in variables:
            urls.append(
                base_url
                + "&".join(
                    [
                        f"{s.year}/daymet_v3_{v}_{s.year}_na.nc4?var=lat",
                        "var=lon",
                        f"var={v}",
                        f"north={north}",
                        f"west={west}",
                        f"east={east}",
                        f"south={south}",
                        "disableProjSubset=on",
                        "horizStride=1",
                        f'time_start={s.strftime("%Y-%m-%dT%H:%M:%SZ")}',
                        f'time_end={e.strftime("%Y-%m-%dT%H:%M:%SZ")}',
                        "timeStride=1",
                        "accept=netcdf",
                    ]
                )
            )
    session = utils.retry_requests()

    r = utils.get_url(session, urls[0])
    data = xr.open_dataset(r.content)

    for url in urls[1:]:
        r = utils.get_url(session, url)
        data = xr.merge([data, xr.open_dataset(r.content)])

    for k, v in units.items():
        if k in variables:
            data[k].attrs["units"] = v

    data = data.drop_vars(["lambert_conformal_conic"])
    data.attrs[
        "crs"
    ] = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +ellps=WGS84"

    x_res, y_res = float(data.x.diff("x").min()), float(data.y.diff("y").min())
    x_origin = data.x.values[0] - x_res / 2.0  # PixelAsArea Convention
    y_origin = data.y.values[0] - y_res / 2.0  # PixelAsArea Convention

    transform = (x_res, 0, x_origin, 0, y_res, y_origin)

    x_end = x_origin + data.dims.get("x") * x_res
    y_end = y_origin + data.dims.get("y") * y_res
    x_options = np.array([x_origin, x_end])
    y_options = np.array([y_origin, y_end])

    data.attrs["transform"] = transform
    data.attrs["res"] = (x_res, y_res)
    data.attrs["bounds"] = (
        x_options.min(),
        y_options.min(),
        x_options.max(),
        y_options.max(),
    )

    print("finished.")

    if pet:
        print(
            f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                MARGINE
            )
            + "Computing PET",
            end=" >>> ",
        )
        data = utils.pet_fao_gridded(data)
        print("finished.")

    data = utils.clip_daymet(data, geometry)

    if resolution is not None:
        res_x = resolution if data.x[0] < data.x[-1] else -resolution
        new_x = np.arange(data.x[0], data.x[-1] + res_x, res_x)

        res_y = resolution if data.y[0] < data.y[-1] else -resolution
        new_y = np.arange(data.y[0], data.y[-1] + res_y, res_y)
        data = data.interp(x=new_x, y=new_y, method="linear")

    return data


class NLDI:
    """Access to the Hydro Network-Linked Data Index (NLDI) service."""

    def __init__(self, station_id):
        """Intialize NLCD.

        Parameters
        ----------
        station_id : string
            USGS station ID, defaults to None.
        navigation : string, optional
            Navigation option for delineating the watershed. Options are:
            upstreamMain, upstreamTributaries, downstreamMain, downstreamDiversions
            Defaults to upstreamTributaries.
        distance : int, optional
            Distance in km for finding USGS stations along the flowlines
            based on the navigation option.
            Defaults to None that finds all the stations.
        """

        self.station_id = str(station_id)
        self.session = utils.retry_requests()

    @property
    def starting_comid(self):
        """Find starting ComID based on the USGS station."""
        return nhdplus_navigate(
            "nwissite", self.station_id, navigation=None
        ).comid.tolist()[0]

    @property
    def tributaries(self):
        """Get upstream tributaries of the watershed."""
        return nhdplus_navigate("nwissite", self.station_id)

    @property
    def main(self):
        """Get upstream main channel of the watershed."""
        return nhdplus_navigate("nwissite", self.station_id, navigation="upstreamMain")

    @property
    def pour_points(self):
        """Get upstream tributaries of the watershed."""
        return nhdplus_navigate("nwissite", self.station_id, dataSource="huc12pp")

    @property
    def comids(self):
        """Find ComIDs of all the flowlines."""
        return self.tributaries.comid.tolist()

    @property
    def basin(self):
        """Delineate the basin."""
        return nwis_basin(self.station_id)

    @property
    def flowlines(self):
        """Get NHDPlus V2 flowlines for the entire watershed"""
        return nhdplus_byid("nhdflowline_network", self.comids)


def nhdplus_navigate(
    feature,
    featureids,
    navigation="upstreamTributaries",
    distance=None,
    dataSource="flowline",
):
    """Get flowlines or USGS stations based on ComID(s) from NHDPlus V2.

    Parameters
    ----------
    feature : string
        The requested feature. The valid features are ``nwissite`` and ``comid``.
    featureids : string or list
        The ID(s) of the requested feature.
    navigation : string, optional
        The direction for navigating the NHDPlus database. The valid options are:
        None, ``upstreamMain``, ``upstreamTributaries``,``downstreamMain``,
        ``downstreamDiversions``. Defaults to upstreamTributaries.
    distance : float, optional
        The distance to limit the navigation in km. Defaults to None (limitless).
    dataSource : string, optional
        The data source to be navigated. Acceptable options are ``flowline`` for flowlines,
        ``nwissite`` for USGS stations and ``huc12pp`` for HUC12 pour points.
        Defaults to None.

    Returns
    -------
    GeoDataFrame
    """
    valid_features = ["comid", "nwissite"]
    if feature not in valid_features:
        msg = "The acceptable feature options are:"
        msg += f" {', '.join(x for x in valid_features)}"
        raise ValueError(msg)

    valid_dataSource = ["flowline", "nwissite", "huc12pp"]
    if dataSource not in valid_dataSource:
        msg = "The acceptable dataSource options are:"
        msg += f"{', '.join(x for x in valid_dataSource)}"
        raise ValueError(msg)

    if not isinstance(featureids, list):
        featureids = [featureids]

    if feature == "nwissite":
        featureids = ["USGS-" + str(f) for f in featureids]

    if len(featureids) == 0:
        raise ValueError("The featureID list is empty!")

    ds = "" if dataSource == "flowline" else f"/{dataSource}"
    dis = "" if distance is None else f"?distance={distance}"

    nav_options = {
        "upstreamMain": "UM",
        "upstreamTributaries": "UT",
        "downstreamMain": "DM",
        "downstreamDiversions": "DD",
    }
    if navigation is not None and navigation not in list(nav_options.keys()):
        msg = "The acceptable navigation options are:"
        msg += f" {', '.join(x for x in list(nav_options.keys()))}"
        raise ValueError(msg)
    elif navigation is None:
        nav = ""
    else:
        nav = f"navigate/{nav_options[navigation]}{ds}{dis}"

    base_url = f"https://labs.waterdata.usgs.gov/api/nldi/linked-data/{feature}"
    crs = "epsg:4326"
    session = utils.retry_requests()

    def get_url(fid):
        url = f"{base_url}/{fid}/{nav}"

        r = utils.get_url(session, url)
        return gpd.GeoDataFrame.from_features(r.json(), crs=crs)

    gdf = gpd.GeoDataFrame(pd.concat(get_url(fid) for fid in featureids))
    comid = "nhdplus_comid" if dataSource == "flowline" else "comid"
    gdf = gdf.rename(columns={comid: "comid"})
    gdf = gdf[["comid", "geometry"]]
    gdf["comid"] = gdf.comid.astype("int64")
    gdf.crs = crs

    return gdf


def nhdplus_bybox(feature, bbox):
    """Get NHDPlus flowline database within a bounding box.

    Parameters
    ----------
    feature : string
        The NHDPlus feature to be downloaded. Valid features are:
        ``nhdarea``, ``nhdwaterbody``, ``catchmentsp``, and ``nhdflowline_network``
    bbox : list
        The bounding box for the region of interest, defaults to None. The list
        should provide the corners in this order:
        [west, south, east, north]

    Returns
    -------
    GeoDataFrame
    """

    valid_features = ["nhdarea", "nhdwaterbody", "catchmentsp", "nhdflowline_network"]
    if feature not in valid_features:
        msg = f"The provided feature, {feature}, is not valid."
        msg += f" Valid features are {', '.join(x for x in valid_features)}"
        raise ValueError(msg)

    service = utils.Geoserver("https://cida.usgs.gov/nwc/geoserver", "wfs", feature)
    r = service.getfeature_bybox(bbox)

    crs = "epsg:4326"
    gdf = gpd.GeoDataFrame.from_features(r.json(), crs=crs)
    gdf.crs = crs
    if gdf.shape[0] == 0:
        raise KeyError(
            f"No feature was found in bbox({', '.join(str(round(x, 3)) for x in bbox)})"
        )

    return gdf


def nhdplus_byid(feature, featureids):
    """Get flowlines or catchments from NHDPlus V2 based on ComIDs.

    Parameters
    ----------
    feature : string
        The requested feature. The valid features are:
        ``catchmentsp`` and ``nhdflowline_network``
    featureids : string or list
        The ID(s) of the requested feature.

    Returns
    -------
    GeoDataFrame
    """
    valid_features = ["catchmentsp", "nhdflowline_network"]
    if feature not in valid_features:
        msg = f"The provided feature, {feature}, is not valid."
        msg += f"Valid features are {', '.join(x for x in valid_features)}"
        raise ValueError(msg)

    service = utils.Geoserver("https://cida.usgs.gov/nwc/geoserver", "wfs", feature)
    propertyname = "featureid" if feature == "catchmentsp" else "comid"
    r = service.getfeature_byid(propertyname, featureids)

    crs = "epsg:4326"
    gdf = gpd.GeoDataFrame.from_features(r.json(), crs=crs)
    gdf.crs = crs
    if gdf.shape[0] == 0:
        raise KeyError("No feature was found with the provided IDs")
    return gdf


def ssebopeta_bygeom(geometry, start=None, end=None, years=None, resolution=None):
    """Gridded data from the SSEBop database.

    Note
    ----
    Since there's still no web service available for subsetting, the data first
    needs to be downloads for the requested period then the data is masked by the
    region interest locally. Therefore, it's not as fast as other functions and
    the bottleneck could be download speed.

    Parameters
    ----------
    geometry : Geometry
        The geometry for downloading clipping the data. For a box geometry,
        the order should be as follows:
        geom = box(minx, miny, maxx, maxy)
    start : string or datetime
        Starting date
    end : string or datetime
        Ending date
    years : list
        List of years
    resolution : float
        The desired output resolution for the output in decimal degree,
        defaults to no resampling. The resampling is done using bilinear method

    Returns
    -------
    xarray.DataArray
        The actual ET for the requested region.
    """

    from shapely.geometry import Polygon
    import socket
    from unittest.mock import patch
    import zipfile
    import io

    if not isinstance(geometry, Polygon):
        raise TypeError("Geometry should be of type Shapely Polygon.")

    if years is None and start is not None and end is not None:
        if pd.to_datetime(start) < pd.to_datetime("2000-01-01"):
            raise ValueError("SSEBop database ranges from 2000 till 2018.")
    elif years is not None and start is None and end is None:
        years = years if isinstance(years, list) else [years]
        dates = [pd.date_range(f"{year}0101", f"{year}1231") for year in years]
        for d in dates:
            if d[0] < pd.to_datetime("2000-01-01"):
                raise ValueError("SSEBop database ranges from 2000 till 2018.")
    else:
        raise ValueError("Either years or start and end arguments should be provided.")

    base_url = (
        "https://edcintl.cr.usgs.gov/downloads/sciweb1/"
        + "shared/uswem/web/conus/eta/modis_eta/daily/downloads"
    )
    f_list = [
        (d, f"{base_url}/det{d.strftime('%Y%j')}.modisSSEBopETactual.zip")
        for d in pd.date_range(start, end)
    ]

    print(
        f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(MARGINE)
        + f"Downloading the data from SSEBop",
        end=" >>> ",
    )

    orig_getaddrinfo = socket.getaddrinfo
    session = utils.retry_requests()

    def getaddrinfoIPv4(host, port, family=0, type=0, proto=0, flags=0):
        return orig_getaddrinfo(
            host=host,
            port=port,
            family=socket.AF_INET,
            type=type,
            proto=proto,
            flags=flags,
        )

    # disable IPv6 to speedup the download
    with patch("socket.getaddrinfo", side_effect=getaddrinfoIPv4):
        # find the mask using the first dataset
        dt, url = f_list[0]

        r = utils.get_url(session, url)

        z = zipfile.ZipFile(io.BytesIO(r.content))
        with rasterio.MemoryFile() as memfile:
            memfile.write(z.read(z.filelist[0].filename))
            with memfile.open() as src:
                ras_msk, _ = rasterio.mask.mask(src, [geometry])
                nodata = src.nodata
                with xr.open_rasterio(src) as ds:
                    ds.data = ras_msk
                    msk = ds < nodata if nodata > 0.0 else ds > nodata
                    ds = ds.where(msk, drop=True)
                    ds = ds.expand_dims(dict(time=[dt]))
                    ds = ds.squeeze("band", drop=True)
                    ds.name = "eta"
                    data = ds * 1e-3

        # apply the mask to the rest of the data and merge
        for dt, url in f_list[1:]:
            r = utils.get_url(session, url)

            z = zipfile.ZipFile(io.BytesIO(r.content))
            with rasterio.MemoryFile() as memfile:
                memfile.write(z.read(z.filelist[0].filename))
                with memfile.open() as src:
                    with xr.open_rasterio(src) as ds:
                        ds = ds.where(msk, drop=True)
                        ds = ds.expand_dims(dict(time=[dt]))
                        ds = ds.squeeze("band", drop=True)
                        ds.name = "eta"
                        data = xr.merge([data, ds * 1e-3])

    data["eta"].attrs["units"] = "mm/day"

    if resolution is not None:
        fac = resolution * 3600.0 / 30.0  # from degree to 1 km
        new_x = np.linspace(data.x[0], data.x[-1], int(data.dims["x"] / fac))
        new_y = np.linspace(data.y[0], data.y[-1], int(data.dims["y"] / fac))
        data = data.interp(x=new_x, y=new_y, method="linear")

    print("finished.")
    return data


def ssebopeta_byloc(lon, lat, start=None, end=None, years=None):
    """Gridded data from the SSEBop database.

    The data is clipped using netCDF Subset Service.

    Parameters
    ----------
    geom : list
        The bounding box for downloading the data. The order should be
        as follows:
        geom = [west, south, east, north]
    start : string or datetime
        Starting date
    end : string or datetime
        Ending date
    years : list
        List of years

    Returns
    -------
    xarray.DataArray
        The actual ET for the requested region.
    """

    from shapely.geometry import box

    # form a geometry with a size less than the grid size (1 km)
    ext = 0.0005
    geometry = box(lon - ext, lat - ext, lon + ext, lat + ext)

    if years is None and start is not None and end is not None:
        ds = ssebopeta_bygeom(geometry, start=start, end=end)
    elif years is not None and start is None and end is None:
        ds = ssebopeta_bygeom(geometry, years=years)
    else:
        raise ValueError("Either years or start and end arguments should be provided.")

    data = ds.to_dataframe().reset_index()[["time", "et"]]
    data.columns = [["time", "et (mm/day)"]]
    data.set_index("time", inplace=True)
    return data


def NLCD(
    geometry,
    years=None,
    data_dir="/tmp",
    width=2000,
    resolution=None,
    statistics=False,
):
    """Get data from NLCD 2016 database.

    Download land use, land cover data from NLCD2016 database within
    a given geometry in epsg:4326.

    Note
    ----
        NLCD data has a resolution of 1 arc-sec (~30 m).

        The following references have been used:
            * https://github.com/jzmiller1/nlcd
            * https://geopython.github.io/OWSLib/
            * https://www.mrlc.gov/data-services-page
            * https://www.arcgis.com/home/item.html?id=624863a9c2484741a9e2cc1ec9c95bce
            * https://github.com/ozak/georasters
            * https://automating-gis-processes.github.io/CSC18/index.html
            * https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend

    Parameters
    ----------
    geometry : Shapely Polygon
        The geometry for extracting the data.
    years : dict, optional
        The years for NLCD data as a dictionary, defaults to {'impervious': 2016, 'cover': 2016, 'canopy': 2016}.
    data_dir : string or Path, optional
        The directory for storing the output ``geotiff`` files, defaults to /tmp/
    width : int, optional
        Width of the output image in pixels, defaults to 2000 pixels.
    resolution : float
        The desired output resolution for the output in decimal degree,
        defaults to no resampling. The resampling is done using bilinear method
        for impervious and canopy data, and majority for the cover.
    statistics : bool
        Whether to perform simple statistics on the data, default to False

    Returns
    -------
     xarray.DataArray (optional), tuple (optional)
         The data as a single DataArray and or the statistics in form of
         three dicts (imprevious, canopy, cover) in a tuple

    """
    from owslib.wms import WebMapService
    import rasterstats
    from shapely.geometry import Polygon
    import os
    from rasterio.enums import Resampling

    if not isinstance(geometry, Polygon):
        raise TypeError("Geometry should be of type Shapely Polygon.")

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        try:
            os.makedirs(data_dir)
        except OSError:
            print(
                f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                    MARGINE
                )
                + f"Input directory cannot be created: {data_dir}"
            )

    nlcd_meta = utils.nlcd_helper()

    avail_years = {
        "impervious": nlcd_meta["impervious_years"],
        "cover": nlcd_meta["cover_years"],
        "canopy": nlcd_meta["canopy_years"],
    }

    if years is None:
        years = {"impervious": 2016, "cover": 2016, "canopy": 2016}
    if isinstance(years, dict):
        years = years
    else:
        raise TypeError(
            f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                MARGINE
            )
            + "Years should be of type dict."
        )

    for service in list(years.keys()):
        if years[service] not in avail_years[service]:
            msg = (
                f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                    MARGINE
                )
                + f"{service.capitalize()} data for {years[service]} is not in the databse."
                + "Avaible years are:"
                + f"{' '.join(str(x) for x in avail_years[service])}"
            )
            raise ValueError(msg)

    url = "https://www.mrlc.gov/geoserver/mrlc_download/wms?service=WMS,request=GetCapabilities"
    fpaths = [
        Path(data_dir, f"{d}_{years[d]}.geotiff").exists() for d in list(years.keys())
    ]
    if not all(fpaths) or str(data_dir) == "/tmp":
        print(
            f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                MARGINE
            )
            + "Connecting to MRLC Web Map Service",
            end=" >>> ",
        )
        wms = WebMapService(url, version="1.3.0")
        print("connected.")

    layers = [
        ("canopy", f'NLCD_{years["canopy"]}_Tree_Canopy_L48'),
        ("cover", f'NLCD_{years["cover"]}_Land_Cover_Science_product_L48'),
        ("impervious", f'NLCD_{years["impervious"]}_Impervious_L48'),
    ]

    params = {}
    nodata = 199
    for data_type, layer in layers:
        data_path = Path(data_dir, f"{data_type}_{years[data_type]}.geotiff")
        if Path(data_path).exists() and str(data_dir) != "/tmp":
            print(
                f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                    MARGINE
                )
                + f"Using existing {data_type} data file: {data_path}"
            )
        else:
            bbox = geometry.bounds

            geod = pyproj.Geod(ellps="WGS84")
            west, south, east, north = bbox
            _, _, bbox_w = geod.inv(west, south, east, south)
            _, _, bbox_h = geod.inv(west, south, west, north)
            height = int(abs(bbox_h) / abs(bbox_w) * width)

            print(
                f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                    MARGINE
                )
                + f"Downloading {data_type} data from NLCD {years[data_type]} database",
                end=" >>> ",
            )

            img = wms.getmap(
                layers=[layer],
                srs="epsg:4326",
                bbox=bbox,
                size=(width, height),
                format="image/geotiff",
            )

            with rasterio.MemoryFile() as memfile:
                memfile.write(img.read())
                with memfile.open() as src:
                    if resolution is not None:
                        # degree to arc-sec since res is 1 arc-sec
                        if data_type == "cover":
                            resampling = Resampling.mode
                        else:
                            resampling = Resampling.bilinear
                        fac = resolution * 3600.0
                        data = src.read(
                            out_shape=(
                                src.count,
                                int(src.width / fac),
                                int(src.height / fac),
                            ),
                            resampling=resampling,
                        )
                        transform = src.transform * src.transform.scale(
                            (src.width / data.shape[1]), (src.height / data.shape[2])
                        )
                        meta = src.meta
                        meta.update(
                            {
                                "driver": "GTiff",
                                "width": data.shape[1],
                                "height": data.shape[2],
                                "transform": transform,
                            }
                        )

                        with rasterio.open("/tmp/resampled.tif", "w", **meta) as dest:
                            dest.write(data)

                        with rasterio.open("/tmp/resampled.tif", "r") as dest:
                            ras_msk, transform = rasterio.mask.mask(
                                dest, [geometry], nodata=nodata
                            )
                            meta = src.meta
                    else:
                        ras_msk, transform = rasterio.mask.mask(
                            src, [geometry], nodata=nodata
                        )
                        meta = src.meta

                    meta.update(
                        {
                            "driver": "GTiff",
                            "width": ras_msk.shape[1],
                            "height": ras_msk.shape[2],
                            "transform": transform,
                            "nodata": nodata,
                        }
                    )

                    with rasterio.open(data_path, "w", **meta) as dest:
                        dest.write(ras_msk)
            print("finished.")

        if statistics and data_type != "cover":
            params[data_type] = rasterstats.zonal_stats(
                geometry, data_path, category_map=nlcd_meta["classes"],
            )[0]

    data_path = Path(data_dir, f"cover_{years['cover']}.geotiff")
    with xr.open_rasterio(data_path) as ds:
        ds = ds.squeeze("band", drop=True)
        ds = ds.where((ds > 0) & (ds < nodata), drop=True)
        ds.name = "cover"
        attrs = ds.attrs
        ds.attrs["units"] = "classes"

    for data_type in ["canopy", "impervious"]:
        data_path = Path(data_dir, f"{data_type}_{years[data_type]}.geotiff")
        with xr.open_rasterio(data_path) as rs:
            rs = rs.squeeze("band", drop=True)
            rs = rs.where(rs < nodata, drop=True)
            rs.name = data_type
            rs.attrs["units"] = "%"
            ds = xr.merge([ds, rs])

    ds.attrs = attrs

    if statistics:
        cover_arr = ds.cover.values
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
            np.array(
                [np.count_nonzero(cover_arr // 10 == c) for c in range(10) if c != 6]
            )
            / total_pix
            * 100.0
        )

        category_percentage = dict(zip(list(nlcd_meta["categories"].keys()), cat_list))

        stats = {
            "impervious": params["impervious"],
            "canopy": params["canopy"],
            "cover": {"classes": class_percentage, "categories": category_percentage},
        }

    if statistics:
        return ds, stats
    else:
        return ds


def dem_bygeom(geometry, demtype="SRTMGL1", resolution=None, output=None):
    """Get DEM data from `OpenTopography <https://opentopography.org/>`_ service.

    Parameters
    ----------
    geometry : Geometry
        A shapely Polygon.
    demtype : string, optional
        The type of DEM to be downloaded, default to SRTMGL1 for 30 m resolution.
        Available options are 'SRTMGL3' for SRTM GL3 (3 arc-sec or ~90m) and 'SRTMGL1' for
        SRTM GL1 (1 arc-sec or ~30m).
    resolution : float, optional
        The desired output resolution for the output in decimal degree,
        defaults to no resampling. The resampling is done using cubic convolution method
    output : string or Path, optional
        The path to save the data as raster, defaults to None.

    Returns
    -------
    xarray.DataArray
        DEM in meters.
    """

    import rasterio
    import rasterio.mask
    from shapely.geometry import Polygon
    from rasterio.enums import Resampling
    import os

    if not isinstance(geometry, Polygon):
        raise TypeError("Geometry should be of type Shapely Polygon.")

    bbox = dict(
        zip(["west", "south", "east", "north"], [round(i, 6) for i in geometry.bounds])
    )

    url = "https://portal.opentopography.org/otr/getdem"
    payload = {"demtype": demtype, "outputFormat": "GTiff", **bbox}

    print(
        f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(MARGINE)
        + f"Downloading DEM data from OpenTopography",
        end=" >>> ",
    )

    session = utils.retry_requests()
    r = utils.get_url(session, url, payload)

    with rasterio.MemoryFile() as memfile:
        memfile.write(r.content)
        with memfile.open() as src:
            if resolution is not None:
                fac = resolution * 3600.0  # degree to arc-sec since res is 1 arc-sec
                data = src.read(
                    out_shape=(src.count, int(src.width / fac), int(src.height / fac)),
                    resampling=Resampling.bilinear,
                )
                transform = src.transform * src.transform.scale(
                    (src.width / data.shape[1]), (src.height / data.shape[2])
                )
                meta = src.meta
                meta.update(
                    {
                        "driver": "GTiff",
                        "width": data.shape[1],
                        "height": data.shape[2],
                        "transform": transform,
                    }
                )

                with rasterio.open("/tmp/resampled.tif", "w", **meta) as tmp:
                    tmp.write(data)

                with rasterio.open("/tmp/resampled.tif", "r") as tmp:
                    ras_msk, transform = rasterio.mask.mask(tmp, [geometry])
                    nodata = tmp.nodata
                    dest = "/tmp/resampled.tif"
                    meta = tmp.meta
                    meta.update(
                        {
                            "driver": "GTiff",
                            "width": ras_msk.shape[1],
                            "height": ras_msk.shape[2],
                            "transform": transform,
                        }
                    )
            else:
                ras_msk, transform = rasterio.mask.mask(src, [geometry])
                nodata = src.nodata
                dest = src
                meta = src.meta
                meta.update(
                    {
                        "driver": "GTiff",
                        "width": ras_msk.shape[1],
                        "height": ras_msk.shape[2],
                        "transform": transform,
                    }
                )

            if output is not None:
                data_dir = Path(output).parent
                if not data_dir.is_dir():
                    try:
                        os.makedirs(data_dir)
                    except OSError:
                        print(
                            f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                                MARGINE
                            )
                            + f"Input directory cannot be created: {data_dir}"
                        )
                with rasterio.open(output, "w", **meta) as f:
                    f.write(ras_msk)

            with xr.open_rasterio(dest) as ds:
                ds.data = ras_msk
                msk = ds < nodata if nodata > 0.0 else ds > nodata
                ds = ds.where(msk, drop=True)
                ds = ds.squeeze("band", drop=True)
                ds.name = "elevation"
                ds.attrs["units"] = "meters"

    print("finished.")

    return ds
