# -*- coding: utf-8 -*-
"""Accessing data from the supported databases through their APIs."""

from hydrodata import utils
import pandas as pd
import geopandas as gpd
import json
import xarray as xr
from pathlib import Path
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException


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

    print(f"[ID: {station_id}] Downloading stream flow data from USGS >>>")
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

    session = utils.retry_requests()
    try:
        r = session.get(url, params=payload)
    except HTTPError:
        print(
            f"[ID: {station_id}] {err[err['HTTP Error Code'] == r.status_code].Explanation.values[0]}"
        )
        raise
    except ConnectionError or Timeout or RequestException:
        raise

    ts = r.json()["value"]["timeSeries"][0]["values"][0]["value"]
    df = pd.DataFrame.from_dict(ts, orient="columns")
    df["dateTime"] = pd.to_datetime(df["dateTime"], format="%Y-%m-%dT%H:%M:%S")
    df.set_index("dateTime", inplace=True)
    qobs = df.value.astype("float64") * 0.028316846592  # Convert cfs to cms
    return qobs


def deymet_singlepixel(lon, lat, start, end, pet=False, pheonolgy=False):
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
    pet : bool
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`_.
        The default is False
    pheonolgy : bool
        Whether to consider pheology in computing PET using growing season index.
        `Thompson et al., 2011 <https://doi.org/10.1029/2010WR009797>`_.
        The default is False.
        
    Returns
    -------
        qobs : dataframe
        Streamflow data observations in cubic meter per second (cms)
    """
    import eto

    if not (14.5 < lat < 52.0) or not (-131.0 < lon < -53.0):
        msg = "The location is outside the Daymet dataset. The acceptable range is: "
        msg += "14.5 < lat < 52.0 and -131.0 < lon < -53.0"
        raise ValueError(msg)

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if start < pd.to_datetime("1980-01-01"):
        raise ValueError("Daymet database ranges from 1980 till present.")

    url = "https://daymet.ornl.gov/single-pixel/api/data"
    payload = {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
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

    if pet:
        data = df[["tmax (deg c)", "tmin (deg c)", "vp (Pa)"]].copy()
        data.columns = ["T_max", "T_min", "e_a"]
        data["T_mean"] = data[["T_max", "T_min"]].mean(axis=1)
        data["R_s"] = df['srad (W/m^2)'] * df['dayl (s)'] * 1e-6  # to MJ/m2
        data["e_a"] *= 1e-3  # to kPa

        et = eto.ETo()
        freq = "D"
        elevation = utils.get_elevation(lon, lat)
        et.param_est(
            data[["R_s", "T_max", "T_min", "e_a"]],
            freq,
            elevation,
            lat,
            lon,
        )
        data["pet"] = et.eto_fao()

        if pheonolgy:
            tmax, tmin = 10.0, -5.0
            trng = 1.0 / (tmax - tmin)

            def gsi(row):
                if row.T_mean < tmin:
                    return 0
                elif row.T_mean > tmax:
                    return row.pet
                else:
                    return (row.T_mean - tmin) * trng * row.pet

            data["pet"] = data.apply(gsi, axis=1)
            
        df["pet (mm)"] = data["pet"]
    return df


def daymet_gridded(start_date, end_date, variables, bbox, tabel=False):
    """Gridded data from the Daymet database.
    
    The data is clipped using netCDF Subset Service.
    Parameters
    ----------
    start_date : string or datetime
        Starting date
    end_date : string or datetime
        Ending date
    variables : string or list
        List of variables to be downloaded. The acceptable variables are:
        tmin, tmax, prcp, srad, vp, swe, dayl
        Descriptions can be found in https://daymet.ornl.gov/overview
    bbox : list
        The bounding box for downloading the data. The order should be
        as follows:
        bbox = [west, south, east, north]
    tabel : bool
        If True, additionally a dataframe is returned that includes
        description of all the Daymet variables and their units.
    
    Returns
    -------
    data : xarray dataset
        The output dataset.
    var_tabel : Pandas dataframe
        A dataframe that includes description of all the Daymet variables. Only if
        tabel is set to True. The default is False.
    """
    from pandas.tseries.offsets import DateOffset
    import numpy as np

    base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/"

    start_date = pd.to_datetime(start_date) + DateOffset(hour=12)
    end_date = pd.to_datetime(end_date) + DateOffset(hour=12)

    variables = variables if isinstance(variables, list) else [variables]

    vars_table = pd.read_html("https://daymet.ornl.gov/overview")[1]
    valid_variables = vars_table.Abbr.values

    invalid_idx = [i for i, v in enumerate(variables) if v not in valid_variables]
    if len(invalid_idx) > 0:
        invalids = [variables[i] for i in invalid_idx]
        msg = "These variables are not valid:"
        msg += ", ".join(x for x in invalids)
        msg += f'\nValid variables are {", ".join(x for x in valid_variables)}'
        raise ValueError(msg)

    dates = utils.daymet_dates(start_date, end_date)

    west, south, east, north = np.round(bbox, 6)
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
    if tabel:
        return data, vars_table
    else:
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
        return gdf.geometry

    def get_river_network(self, navigation=None):
        """Get the river network geometry from NHDPlus V2."""
        navigation = (
            self.navigation if navigation is None else self.nav_options[navigation]
        )
        url = self.base_url + f"/USGS-{self.station_id}/navigate/{navigation}"
        try:
            r = self.session.get(url)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.columns = ["geometry", "comid"]
        gdf.set_index("comid", inplace=True)
        return gdf

    def get_stations(self, navigation=None, distance=None):
        """Find the USGS stations along the river network."""
        navigation = (
            self.navigation if navigation is None else self.nav_options[navigation]
        )
        distance = self.distance if distance is None else str(int(distance))

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
        return gdf

    def get_nhdplus_byid(self, comids, layer="nhdflowline_network"):
        """Get NHDPlus flowline database based on ComIDs"""
        id_name = dict(catchmentsp="featureid", nhdflowline_network="comid")
        if layer not in list(id_name.keys()):
            raise ValueError(
                f"Acceptable values for layer are {', '.join(x for x in list(id_name.keys()))}"
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
        return gdf

    def get_nhdplus_bybox(self, bbox=None, layer="nhdflowline_network"):
        """Get NHDPlus flowline database within a bounding box."""
        valid_layers = ["nhdarea", "nhdwaterbody", "catchmentsp", "nhdflowline_network"]
        if layer not in valid_layers:
            msg = f"The provided layer, {layer}, is not valid."
            msg += f"Valid layers are {', '.join(x for x in valid_layers)}"
            raise ValueError(msg)

        if bbox is None:
            bbox = self.basin.bounds.values[0]

        if any((i > 180.0) or (i < -180.0) for i in bbox):
            raise ValueError("bbox should be provided in lat/lon coordinates.")

        url = "https://cida.usgs.gov/nwc/geoserver/nhdplus/ows"
        filter_xml = "".join(
            [
                '<?xml version="1.0"?>',
                '<wfs:GetFeature xmlns:wfs="http://www.opengis.net/wfs" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:gml="http://www.opengis.net/gml" service="WFS" version="1.1.0" outputFormat="application/json" xsi:schemaLocation="http://www.opengis.net/wfs http://schemas.opengis.net/wfs/1.1.0/wfs.xsd">',
                '<wfs:Query xmlns:feature="http://gov.usgs.cida/nhdplus" typeName="feature:',
                layer,
                '" srsName="EPSG:4326">',
                '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">',
                "<ogc:BBOX>",
                "<ogc:PropertyName>the_geom</ogc:PropertyName>",
                "<gml:Envelope>",
                "<gml:lowerCorner>",
                str(bbox[1]),
                " ",
                str(bbox[0]),
                "</gml:lowerCorner>",
                "<gml:upperCorner>",
                str(bbox[3]),
                " ",
                str(bbox[2]),
                "</gml:upperCorner>",
                "</gml:Envelope>",
                "</ogc:BBOX>",
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
        return gdf


def streamstats(lon, lat, data_dir=None):
    """Get watershed geometr and characteristics from StreamStats service.

    The USGS StreamStats API is built around watersheds as organizational
    units. Watersheds in the 50 U.S. states can be found using lat/lon
    lookups, along with information about the watershed including its HUC code
    and a GeoJSON representation of the polygon of a watershed. Basin
    characteristics can also be extracted from watersheds. Additionally,
    the data including parameters and geometry will be saved as a json file.
    
    The original code is taken from:
    https://github.com/earthlab/streamstats
    
    Parameters
    ----------
    lon : float
        Longitude of point in decimal degrees.
    lat : float
        Latitude of point in decimal degrees.
    data_dir : string or Path
        The directory for storing the data. If not provided, the data will not
        be saved on disk.
        
    Returns
    -------
    parameters : dict
        A dictionary of watershed parameters except for NLCD related which
        can be retieved using the `nlcd` function separately.
    geometry : Polygon
        A Shapely polygon containing the watershed geometry.
    """
    url = "https://streamstats.usgs.gov/streamstatsservices/watershed.geojson"
    payload = {
        "rcode": utils.get_state(lon, lat),
        "xlocation": lon,
        "ylocation": lat,
        "crs": 4326,
        "includeparameters": True,
        "includeflowtypes": False,
        "includefeatures": True,
        "simplify": False,
    }

    try:
        session = utils.retry_requests()
        r = session.get(url, params=payload)
    except HTTPError or ConnectionError or Timeout or RequestException:
        raise

    data = r.json()

    parameters = data["parameters"]

    watershed_point = data["featurecollection"][0]["feature"]
    huc = watershed_point["features"][0]["properties"]["HUCID"]
    parameters.insert(
        0,
        {
            "ID": 0,
            "name": "HUC number",
            "description": "Hudrologic Unit Code of the watershed",
            "code": "HUC",
            "unit": "string",
            "value": huc,
        },
    )
    try:
        for dictionary in data["featurecollection"]:
            print(dictionary)
            gdf = gpd.GeoDataFrame.from_features(dictionary["feature"])
    except LookupError:
        raise LookupError(f"Could not find 'globalwatershed' in the data.")

    geometry = gdf.geometry.values[0]

    if data_dir is not None:
        wshed_file = Path(data_dir, "watershed.json")
        with open(wshed_file, "w") as fp:
            json.dump(data, fp)

    return parameters, geometry
