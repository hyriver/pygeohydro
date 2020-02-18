# -*- coding: utf-8 -*-
"""Accessing data from the supported databases using their APIs."""

from hydrodata import utils
import requests
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path


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
    """
    station_id = str(station_id)
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end = pd.to_datetime(end).strftime("%Y-%m-%d")

    print(f"[ID: {station_id}] Downloading stream flow data from USGS >>>")
    url = "https://waterservices.usgs.gov/nwis/dv"
    payload = {
        "format" : "json",
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
    except requests.exceptions.HTTPError:
        print(f"[ID: {station_id}] {err[err['HTTP Error Code'] == r.status_code].Explanation.values[0]}")
        raise
    except requests.exceptions.ConnectionError or requests.exceptions.Timeout or requests.exceptions.RequestException:
        raise

    ts = r.json()["value"]["timeSeries"][0]["values"][0]["value"]
    df = pd.DataFrame.from_dict(ts, orient="columns")
    df["dateTime"] = pd.to_datetime(df["dateTime"], format="%Y-%m-%dT%H:%M:%S")
    df.set_index("dateTime", inplace=True)
    qobs = df.value.astype("float64") * 0.028316846592 # Convert cfs to cms
    return qobs


class NLDI():
    """Access to the Hydro Network-Linked Data Index (NLDI) service."""

    def __init__(self, station_id, navigation="upstreamTributaries", distance=None):
        self.station_id = station_id
        self.distance = None if distance is None else str(int(distance))
        
        self.nav_options = {"upstreamMain": "UM",
                            "upstreamTributaries": "UT",
                            "downstreamMain": "DM",
                            "downstreamDiversions": "DD"}
        if navigation not in list(self.nav_options.keys()):
            msg = "The acceptable navigation options are:"
            msg += f"{', '.join(x for x in list(self.nav_options.keys()))}"
            raise ValueError(f"The acceptable navigation options are:")
        else:
            self.navigation = self.nav_options[navigation]

        self.base_url = "https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite"
        self.session = utils.retry_requests() 
    
    @property
    def comids(self):
        url = self.base_url + f"/USGS-{self.station_id}"
        r = self.session.get(url)
        return gpd.GeoDataFrame.from_features(r.json())

    @property
    def basin(self):
        url = self.base_url + f"/USGS-{self.station_id}/basin"
        r = self.session.get(url)
        return gpd.GeoDataFrame.from_features(r.json())

    def get_river_network(self, navigation=None):
        navigation = self.navigation if navigation is None else self.nav_options[navigation]
        url = self.base_url + f"/USGS-{self.station_id}/navigate/{navigation}"
        r = self.session.get(url)
        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.columns = ['geometry', 'comid']
        gdf.set_index('comid', inplace=True)
        return gdf
        
    def get_stations(self, navigation=None, distance=None):
        navigation = self.navigation if navigation is None else self.nav_options[navigation]
        distance = self.distance if distance is None else str(int(distance))

        if distance is None:
            url = self.base_url + f"/USGS-{self.station_id}/navigate/{navigation}/nwissite"
        else:
            url = self.base_url + f"/USGS-{self.station_id}/navigate/{navigation}/nwissite?distance={distance}"
        r = self.session.get(url)
        gdf = gpd.GeoDataFrame.from_features(r.json())
        gdf.set_index('comid', inplace=True)
        return gdf
    
    def get_nhdplus_byid(self, comids, layer):
        id_name = dict(catchmentsp = "featureid", nhdflowline_network = "comid")
        if layer not in list(id_name.keys()):
            raise ValueError(f"Acceptable values for layer are {', '.join(x for x in list(id_name.keys()))}")

        url = "https://cida.usgs.gov/nwc/geoserver/nhdplus/ows"

        filter_1 = ''.join(['<?xml version="1.0"?>',
                         '<wfs:GetFeature xmlns:wfs="http://www.opengis.net/wfs" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:gml="http://www.opengis.net/gml" service="WFS" version="1.1.0" outputFormat="application/json" xsi:schemaLocation="http://www.opengis.net/wfs http://schemas.opengis.net/wfs/1.1.0/wfs.xsd">',
                         '<wfs:Query xmlns:feature="http://gov.usgs.cida/nhdplus" typeName="feature:',
                         layer, '" srsName="EPSG:4326">',
                         '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">',
                         '<ogc:Or>',
                         '<ogc:PropertyIsEqualTo>',
                         '<ogc:PropertyName>',
                         id_name[layer],
                         '</ogc:PropertyName>',
                         '<ogc:Literal>'])

        filter_2 = ''.join(['</ogc:Literal>',
                         '</ogc:PropertyIsEqualTo>',
                         '<ogc:PropertyIsEqualTo>',
                         '<ogc:PropertyName>',
                         id_name[layer],
                         '</ogc:PropertyName>',
                         '<ogc:Literal>'])

        filter_3 = ''.join(['</ogc:Literal>',
                         '</ogc:PropertyIsEqualTo>',
                         '</ogc:Or>',
                         '</ogc:Filter>',
                         '</wfs:Query>',
                         '</wfs:GetFeature>'])

        filter_xml = ''.join([filter_1, filter_2.join(comids), filter_3])

        session = utils.retry_requests()
        r = session.post(url, data=filter_xml)
        gdf = gpd.GeoDataFrame.from_features(r.json())
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
        'rcode': utils.get_state(lon, lat),
        'xlocation': lon,
        'ylocation': lat,
        'crs': 4326,
        'includeparameters': True,
        'includeflowtypes': False,
        'includefeatures': True,
        'simplify': False
    }

    try:
        session = utils.retry_requests()
        r = session.get(url, params=payload)
    except requests.exceptions.HTTPError or requests.exceptions.ConnectionError or requests.exceptions.Timeout or requests.exceptions.RequestException:
        raise

    data = r.json()

    parameters = data['parameters']
    # Remove the NLCD elements since they are handled by the get_nlcd function
    parameters = [i for j, i in enumerate(parameters) if j not in [8, 10, 11, 12, 13, 23]]

    watershed_point = data['featurecollection'][0]['feature']
    huc = watershed_point['features'][0]['properties']['HUCID']
    parameters.insert(0, {'ID': 0,
                          'name': 'HUC number',
                          'description': 'Hudrologic Unit Code of the watershed',
                          'code': 'HUC',
                          'unit': 'string',
                          'value': huc})
    try:
        for dictionary in data['featurecollection']:
            print(dictionary)
            gdf = gpd.GeoDataFrame.from_features(dictionary['feature'])
    except LookupError:
        raise LookupError(f"Could not find 'globalwatershed' in the data.")

    geometry = gdf.geometry.values[0]

    if data_dir is not None:
        wshed_file = Path(data_dir, "watershed.json")
        with open(wshed_file, "w") as fp:
            json.dump(data, fp)

    return parameters, geometry