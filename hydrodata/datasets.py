#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accessing data from the supported databases through their APIs."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from hydrodata import utils
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

import xarray as xr

MARGINE = 15


def nwis(station_id, start, end):
    """Get daily streamflow observation data from USGS.

    Parameters
    ----------
    station_id : string
        The gage ID  of the USGS station
    start : string or datetime
        Start date
    end : string or datetime
        End date

    Returns
    -------
        qobs : dataframe
        Streamflow data observations in cubic meter per second (cms)
    """

    station_id = str(station_id)
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end = pd.to_datetime(end).strftime("%Y-%m-%d")

    url = "https://waterservices.usgs.gov/nwis/dv"
    payload = {
        "format": "json",
        "sites": station_id,
        "startDT": start,
        "endDT": end,
        "parameterCd": "00060",
        "siteStatus": "all",
    }
    err = pd.read_html("https://waterservices.usgs.gov/rest/DV-Service.html")[0]

    print(
        f"[ID: {station_id}] ".ljust(MARGINE)
        + "Downloading stream flow data from NWIS",
        end=" >>> ",
    )

    session = utils.retry_requests()
    try:
        r = session.get(url, params=payload)
    except HTTPError:
        print(
            f"[ID: {station_id}] ".ljust(MARGINE)
            + f"{err[err['HTTP Error Code'] == r.status_code].Explanation.values[0]}"
        )
        raise
    except ConnectionError or Timeout or RequestException:
        raise

    try:
        ts = r.json()["value"]["timeSeries"][0]["values"][0]["value"]
    except IndexError:
        msg = (
            f"[ID: {station_id}] ".ljust(MARGINE)
            + "The requested data is not available in the station."
            + f"Check out https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no={station_id}"
        )
        raise IndexError(msg)
    df = pd.DataFrame.from_dict(ts, orient="columns")
    try:
        df["dateTime"] = pd.to_datetime(df["dateTime"], format="%Y-%m-%dT%H:%M:%S")
    except KeyError:
        msg = (
            f"[ID: {station_id}] ".ljust(MARGINE)
            + "The data is not available in the requested date range."
            + f"Check out https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no={station_id}"
        )
        raise KeyError("")
    df.set_index("dateTime", inplace=True)
    qobs = df.value.astype("float64") * 0.028316846592  # Convert cfs to cms
    print("finished.")
    return qobs


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
        payload = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "vars": ",".join(x for x in variables),
            "format": "json",
        }
    else:
        payload = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "years": ",".join(str(x) for x in years),
            "vars": ",".join(x for x in variables),
            "format": "json",
        }

    session = utils.retry_requests()
    try:
        r = session.get(url, params=payload)
    except HTTPError or ConnectionError or Timeout or RequestException:
        raise

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
    geometry, start=None, end=None, years=None, variables=None, pet=False
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

    Returns
    -------
    xarray.DataArray
        The climate data within the requested geometery.
    """

    from pandas.tseries.offsets import DateOffset
    from shapely.geometry import Polygon, Point

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
    ds_list = []
    for url in urls:
        try:
            r = session.get(url)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise
        ds_list.append(xr.open_dataset(r.content))

    data = xr.merge(ds_list)
    for k, v in units.items():
        if k in variables:
            data[k].attrs["units"] = v

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

    def _within(x, y, g):
        return np.array([Point(i, j).within(g) for i, j in np.nditer((x, y))]).reshape(
            x.shape
        )

    def within(da, shape):
        return xr.apply_ufunc(_within, da.lon, da.lat, kwargs={"g": shape})

    data = data.where(within(data, geometry), drop=True)
    return data


class NLDI:
    """Access to the Hydro Network-Linked Data Index (NLDI) service."""

    def __init__(self, station_id, navigation="upstreamTributaries", distance=None):
        """Intialize NLCD.

        Parameters
        ----------
        station_id : string
            USGS station ID
        navigation : string
            Navigation option for delineating the watershed. Options are:
            upstreamMain, upstreamTributaries, downstreamMain, downstreamDiversions
            The default is upstreamTributaries.
        distance : int
            Distance in km for finding USGS stations along the flowlines
            based on the navigation option.
            Default is None that finds all the stations.
        """

        self.station_id = station_id
        self.distance = None if distance is None else str(int(distance))

        self.nav_options = {
            "upstreamMain": "UM",
            "upstreamTributaries": "UT",
            "downstreamMain": "DM",
            "downstreamDiversions": "DD",
        }
        if navigation not in list(self.nav_options.keys()):
            msg = "The acceptable navigation options are:"
            msg += f"{', '.join(x for x in list(self.nav_options.keys()))}"
            raise ValueError(f"The acceptable navigation options are:")
        else:
            self.navigation = self.nav_options[navigation]

        self.base_url = "https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite"
        self.session = utils.retry_requests()

    @property
    def comid(self):
        """Find starting ComID based on the USGS station."""
        url = self.base_url + f"/USGS-{self.station_id}"
        try:
            r = self.session.get(url)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        return gpd.GeoDataFrame.from_features(r.json()).values[0]

    @property
    def basin(self):
        """Delineate the basin."""
        url = self.base_url + f"/USGS-{self.station_id}/basin"
        try:
            r = self.session.get(url)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.crs = "EPSG:4326"
        return gdf.geometry

    def get_river_network(self, navigation=None):
        """Get the river network geometry from NHDPlus V2."""
        navigation = (
            self.navigation if navigation is None else self.nav_options[navigation]
        )

        rnav = [k for k, v in self.nav_options.items() if v == navigation]
        rnav = "stream ".join(rnav[0].split("stream")).lower()
        print(
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + f"Downloading {rnav} from NLDI",
            end=" >>> ",
        )
        url = self.base_url + f"/USGS-{self.station_id}/navigate/{navigation}"
        try:
            r = self.session.get(url)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.columns = ["geometry", "comid"]
        gdf.set_index("comid", inplace=True)
        print("finished.")
        return gdf

    def get_stations(self, navigation=None, distance=None):
        """Find the USGS stations along the river network."""
        navigation = (
            self.navigation if navigation is None else self.nav_options[navigation]
        )
        distance = self.distance if distance is None else str(int(distance))

        rnav = [k for k, v in self.nav_options.items() if v == navigation]
        rnav = "stream ".join(rnav[0].split("stream")).lower()
        if distance is None:
            st = "all the stations"
        else:
            st = f"stations within {distance} km of"
        print(
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + f"Downloading {st} {rnav} from NLDI",
            end=" >>> ",
        )
        if distance is None:
            url = (
                self.base_url
                + f"/USGS-{self.station_id}/navigate/{navigation}/nwissite"
            )
        else:
            url = (
                self.base_url
                + f"/USGS-{self.station_id}/navigate/{navigation}/nwissite?distance={distance}"
            )
        try:
            r = self.session.get(url)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.set_index("comid", inplace=True)
        print("finsihed.")
        return gdf

    def get_nhdplus_byid(self, comids, layer="nhdflowline_network"):
        """Get NHDPlus flowline database based on ComIDs"""
        id_name = dict(catchmentsp="featureid", nhdflowline_network="comid")
        if layer not in list(id_name.keys()):
            raise ValueError(
                f"Acceptable values for layer are {', '.join(x for x in list(id_name.keys()))}"
            )
        print(
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + f"Downloading flowlines by ComIDs from NLDI",
            end=" >>> ",
        )

        url = "https://cida.usgs.gov/nwc/geoserver/nhdplus/ows"

        filter_1 = "".join(
            [
                '<?xml version="1.0"?>',
                '<wfs:GetFeature xmlns:wfs="http://www.opengis.net/wfs" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:gml="http://www.opengis.net/gml" service="WFS" version="1.1.0" outputFormat="application/json" xsi:schemaLocation="http://www.opengis.net/wfs http://schemas.opengis.net/wfs/1.1.0/wfs.xsd">',
                '<wfs:Query xmlns:feature="http://gov.usgs.cida/nhdplus" typeName="feature:',
                layer,
                '" srsName="EPSG:4326">',
                '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">',
                "<ogc:Or>",
                "<ogc:PropertyIsEqualTo>",
                "<ogc:PropertyName>",
                id_name[layer],
                "</ogc:PropertyName>",
                "<ogc:Literal>",
            ]
        )

        filter_2 = "".join(
            [
                "</ogc:Literal>",
                "</ogc:PropertyIsEqualTo>",
                "<ogc:PropertyIsEqualTo>",
                "<ogc:PropertyName>",
                id_name[layer],
                "</ogc:PropertyName>",
                "<ogc:Literal>",
            ]
        )

        filter_3 = "".join(
            [
                "</ogc:Literal>",
                "</ogc:PropertyIsEqualTo>",
                "</ogc:Or>",
                "</ogc:Filter>",
                "</wfs:Query>",
                "</wfs:GetFeature>",
            ]
        )

        filter_xml = "".join([filter_1, filter_2.join(comids), filter_3])
        try:
            r = self.session.post(url, data=filter_xml)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.set_index("comid", inplace=True)
        print("finished.")
        return gdf

    def get_nhdplus_bybox(self, geom=None, layer="nhdflowline_network"):
        """Get NHDPlus flowline database within a bounding box."""
        valid_layers = ["nhdarea", "nhdwaterbody", "catchmentsp", "nhdflowline_network"]
        if layer not in valid_layers:
            msg = f"The provided layer, {layer}, is not valid."
            msg += f"Valid layers are {', '.join(x for x in valid_layers)}"
            raise ValueError(msg)

        if geom is None:
            geom = self.basin.bounds.values[0]

        if any((i > 180.0) or (i < -180.0) for i in geom):
            raise ValueError("geom should be provided in lat/lon coordinates.")

        print(
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + f"Downloading flowlines by BBOX from NLDI",
            end=" >>> ",
        )

        url = "https://cida.usgs.gov/nwc/geoserver/nhdplus/ows"
        filter_xml = "".join(
            [
                '<?xml version="1.0"?>',
                '<wfs:GetFeature xmlns:wfs="http://www.opengis.net/wfs" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:gml="http://www.opengis.net/gml" service="WFS" version="1.1.0" outputFormat="application/json" xsi:schemaLocation="http://www.opengis.net/wfs http://schemas.opengis.net/wfs/1.1.0/wfs.xsd">',
                '<wfs:Query xmlns:feature="http://gov.usgs.cida/nhdplus" typeName="feature:',
                layer,
                '" srsName="EPSG:4326">',
                '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">',
                "<ogc:geom>",
                "<ogc:PropertyName>the_geom</ogc:PropertyName>",
                "<gml:Envelope>",
                "<gml:lowerCorner>",
                str(geom[1]),
                " ",
                str(geom[0]),
                "</gml:lowerCorner>",
                "<gml:upperCorner>",
                str(geom[3]),
                " ",
                str(geom[2]),
                "</gml:upperCorner>",
                "</gml:Envelope>",
                "</ogc:geom>",
                "</ogc:Filter>",
                "</wfs:Query>",
                "</wfs:GetFeature>",
            ]
        )
        try:
            r = self.session.post(url, data=filter_xml)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.set_index("comid", inplace=True)
        print("finished.")
        return gdf


def ssebopeta_bygeom(geometry, start=None, end=None, years=None):
    """Gridded data from the SSEBop database.

    The data is clipped using netCDF Subset Service.

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

    base_url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared//uswem/web/conus/eta/modis_eta/daily/downloads"
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

        try:
            r = session.get(url)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

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
            try:
                r = session.get(url)
            except HTTPError or ConnectionError or Timeout or RequestException:
                raise

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

    # form a geometry with a size less than a pixel (1 km)
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


def NLCD(geometry, years=None, data_dir="/tmp", width=2000):
    """Get data from NLCD 2016 database.

    Download land use, land cover data from NLCD2016 database within
    a given geometry with epsg:4326 projection.

    Note
    ----
        NLCD data has a 30 m resolution.

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

    Returns
    -------
     tuple of three :obj: `dict` (imprevious, canopy, cover)
         Some statistics of the extracted data.
    """
    from owslib.wms import WebMapService
    import rasterstats
    from shapely.geometry import Polygon
    import os

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

    nlcd_meta = dict(
        impervious_years=[2016, 2011, 2006, 2001],
        canopy_years=[2016, 2011],
        cover_years=[2016, 2013, 2011, 2008, 2006, 2004, 2001],
        legends={
            "0": "Unclassified",
            "11": "Open Water",
            "12": "Perennial Ice/Snow",
            "21": "Developed, Open Space",
            "22": "Developed, Low Intensity",
            "23": "Developed, Medium Intensity",
            "24": "Developed High Intensity",
            "31": "Barren Land (Rock/Sand/Clay)",
            "41": "Deciduous Forest",
            "42": "Evergreen Forest",
            "43": "Mixed Forest",
            "51": "Dwarf Scrub",
            "52": "Shrub/Scrub",
            "71": "Grassland/Herbaceous",
            "72": "Sedge/Herbaceous",
            "73": "Lichens",
            "74": "Moss",
            "81": "Pasture/Hay",
            "82": "Cultivated Crops",
            "90": "Woody Wetlands",
            "95": "Emergent Herbaceous Wetlands",
        },
        categories={
            "Unclassified": ("0"),
            "Water": ("11", "12"),
            "Developed": ("21", "22", "23", "24"),
            "Barren": ("31",),
            "Forest": ("41", "42", "43"),
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
            "52": 0.1000,
            "71": 0.0350,
            "81": 0.0325,
            "82": 0.0375,
            "90": 0.1200,
            "95": 0.0700,
        },
    )

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
    if not all(fpaths):
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
    for data_type, layer in layers:
        data_path = Path(data_dir, f"{data_type}_{years[data_type]}.geotiff")
        if Path(data_path).exists():
            print(
                f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                    MARGINE
                )
                + f"Using existing {data_type} data file: {data_path}"
            )
        else:
            bbox = geometry.bounds
            height = int(np.abs(bbox[1] - bbox[3]) / np.abs(bbox[0] - bbox[2]) * width)
            print(
                f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                    MARGINE
                )
                + f"Downloading {data_type} data from NLCD {years[data_type]} database",
                end=" >>> ",
            )

            try:
                img = wms.getmap(
                    layers=[layer],
                    srs="epsg:4326",
                    bbox=bbox,
                    size=(width, height),
                    format="image/geotiff",
                )
            except ConnectionError:
                raise (
                    f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(
                        MARGINE
                    )
                    + "Data could not be reached.."
                )

            with rasterio.MemoryFile() as memfile:
                memfile.write(img.read())
                with memfile.open() as src:
                    out_image, out_transform = rasterio.mask.mask(
                        src, [geometry], crop=True
                    )
                    out_meta = src.meta
                    out_meta.update(
                        {
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                        }
                    )
            with rasterio.open(data_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print("finished.")

        categorical = True if data_type == "cover" else False
        params[data_type] = rasterstats.zonal_stats(
            geometry,
            data_path,
            categorical=categorical,
            category_map=nlcd_meta["legends"],
        )[0]

    cover = rasterio.open(Path(data_dir, f"cover_{years['cover']}.geotiff"))
    total_pix = cover.shape[0] * cover.shape[1]
    cover_arr = cover.read()
    class_percentage = dict(
        zip(
            list(nlcd_meta["legends"].values()),
            [
                cover_arr[cover_arr == int(cat)].shape[0] / total_pix * 100.0
                for cat in list(nlcd_meta["legends"].keys())
            ],
        )
    )
    masks = [
        [leg in cat for leg in list(nlcd_meta["legends"].keys())]
        for cat in list(nlcd_meta["categories"].values())
    ]
    cat_list = [np.array(list(class_percentage.values()))[msk].sum() for msk in masks]
    category_percentage = dict(zip(list(nlcd_meta["categories"].keys()), cat_list))

    stats = {
        "impervious": params["impervious"],
        "canopy": params["canopy"],
        "cover": {"classes": class_percentage, "categories": category_percentage},
    }
    return stats


def dem_bygeom(geometry):
    """Get DEM data from `OpenTopography <https://opentopography.org/>`_ service.

    The DEM is extracted from SRTM1 (30-m resolution) database.

    Parameters
    ----------
    geometry : Geometry
        A shapely Polygon.

    Returns
    -------
    xarray.DataArray
        DEM in meters.
    """

    import rasterio
    import rasterio.mask
    from shapely.geometry import Polygon

    if not isinstance(geometry, Polygon):
        raise TypeError("Geometry should be of type Shapely Polygon.")

    west, south, east, north = geometry.bounds
    url = "http://opentopo.sdsc.edu/otr/getdem?"
    payload = dict(
        demtype="SRTMGL1",
        west=west,
        south=south,
        east=east,
        north=north,
        outputFormat="GTiff",
    )
    session = utils.retry_requests()
    try:
        r = session.get(url, params=payload)
    except ConnectionError or Timeout or RequestException:
        raise

    with rasterio.MemoryFile() as memfile:
        memfile.write(r.content)
        with memfile.open() as src:
            ras_msk, _ = rasterio.mask.mask(src, [geometry])
            nodata = src.nodata
            with xr.open_rasterio(src) as ds:
                ds.data = ras_msk
                msk = ds < nodata if nodata > 0.0 else ds > nodata
                ds = ds.where(msk, drop=True)
                ds = ds.squeeze("band", drop=True)
                ds.name = "elevation"

    return ds
