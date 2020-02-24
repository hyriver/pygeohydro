# -*- coding: utf-8 -*-
"""Accessing data from the supported databases through their APIs."""

from hydrodata import utils
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import xarray as xr
from pathlib import Path
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException


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

    ts = r.json()["value"]["timeSeries"][0]["values"][0]["value"]
    df = pd.DataFrame.from_dict(ts, orient="columns")
    df["dateTime"] = pd.to_datetime(df["dateTime"], format="%Y-%m-%dT%H:%M:%S")
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
        df : Pandas DataFrame
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
        geom = box(minx, miny, maxx, maxy)
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
    data : xarray dataset
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
    data : xarray dataset
        The actual ET for the requested region.
    """

    import rasterio
    import rasterio.mask

    print(
        f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(MARGINE)
        + "Staging actual ET data from SSEBop:"
    )

    fnames = utils.stage_eta(start=start, end=end)

    f_list = [
        (f, pd.to_datetime("".join(filter(str.isdigit, f.stem)), format="%Y%j"))
        for f in fnames
    ]

    print(
        f"[CNT: ({geometry.centroid.x:.2f}, {geometry.centroid.y:.2f})] ".ljust(MARGINE)
        + "Proccessing the data for the requested region",
        end=" >>> ",
    )

    # find the mask using the first dataset
    fname = f_list[0][0]
    with rasterio.open(f"zip://{fname}!{fname.stem}.tif") as src:
        ras_msk, _ = rasterio.mask.mask(src, [geometry])

    with xr.open_rasterio(f"zip://{fname}!{fname.stem}.tif") as ds:
        ds.data = ras_msk
        msk = ds < 9999

    ds = xr.open_rasterio(f"zip://{fname}!{fname.stem}.tif")
    ds = ds.where(msk, drop=True)
    ds = ds.expand_dims(dict(time=[f_list[0][1]]))
    ds = ds.squeeze("band", drop=True)
    ds.name = "et"
    data = ds * 1e-3

    # apply the mask to the rest of the data and merge
    for fname, dt in f_list:
        ds = xr.open_rasterio(f"zip://{fname}!{fname.stem}.tif")
        ds = ds.where(msk, drop=True)
        ds = ds.expand_dims(dict(time=[dt]))
        ds = ds.squeeze("band", drop=True)
        ds.name = "et"
        data = xr.merge([data, ds * 1e-3])

    data["et"].attrs["units"] = "mm/day"

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
    data : Pandas DataFrame
        The actual ET for the requested region.
    """

    from shapely.geometry import box

    # form a geometry with a size less than a pixel (1 km)
    ext = 0.0005
    geometry = box(lon - ext, lat - ext, lon + ext, lat + ext)

    if years is None and start is not None and end is not None:
        ds = ssebopeta_bygeom(geom, start=start, end=end)
    elif years is not None and start is None and end is None:
        ds = ssebopeta_bygeom(geom, years=years)
    else:
        raise ValueError("Either years or start and end arguments should be provided.")

    data = ds.to_dataframe().reset_index()[["time", "et"]]
    data.columns = [["time", "et (mm/day)"]]
    data.set_index("time", inplace=True)
    return data
