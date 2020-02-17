# -*- coding: utf-8 -*-
"""Get watershed geometry and characteristics using StreamStats service.

The original code is taken from:
https://github.com/earthlab/streamstats
"""

from hydrodata import utils
import requests


class Watershed():
    """Watershed covering a spatial region, with associated information.

    The USGS StreamStats API is built around watersheds as organizational
    units. Watersheds in the 50 U.S. states can be found using lat/lon
    lookups, along with information about the watershed including its HUC code
    and a GeoJSON representation of the polygon of a watershed. Basin
    characteristics can also be extracted from watersheds.
    
    Parameters
    ----------
    lon : float
        Longitude of point in decimal degrees.
    lat : float
        Latitude of point in decimal degrees.
    """
    def __init__(self, lon, lat):
        """Initialize a Watershed object."""
        self.lon, self.lat = lon, lat
        self.state = get_state(lon, lat)

        self.data = self._delineate()
        self.workspace = self.data['workspaceID']
        parameters = self.data['parameters']
        
        # Remove the NLCD elements since they are handled by the get_nlcd function
        self.parameters = [i for j, i in enumerate(parameters) if j not in [8, 10, 11, 12, 13, 23]]

    def __repr__(self):
        """Get the string representation of a watershed."""
        huc = self.huc
        huc_message = f"Watershed object with HUC{len(huc)}: {huc}"
        coord_message = f"containing lon/lat: ({self.lon}, {self.lat})"
        return ', '.join((huc_message, coord_message))

    def _delineate(self):
        """Find the watershed that contains a point.

        Implements a Delineate Watershed by Location query from
        https://streamstats.usgs.gov/docs/streamstatsservices/#/

        Returns
        -------
        data : dict
            Watershed data
        """
        url = "https://streamstats.usgs.gov/streamstatsservices/watershed.geojson"
        payload = {
            'rcode': self.state,
            'xlocation': self.lon,
            'ylocation': self.lat,
            'crs': 4326,
            'includeparameters': True,
            'includeflowtypes': False,
            'includefeatures': True,
            'simplify': False
        }

        try:
            session = utils.retry_requests()
            r = session.get(url, params=payload)
            r.raise_for_status()
        except requests.exceptions.HTTPError or requests.exceptions.ConnectionError or requests.exceptions.Timeout or requests.exceptions.RequestException:
            raise
        return r.json()

    @property
    def huc(self):
        """Find the Hydrologic Unit Code (HUC) of the watershed."""
        watershed_point = self.data['featurecollection'][0]['feature']
        huc = watershed_point['features'][0]['properties']['HUCID']
        return huc

    @property
    def boundary(self):
        """Return the full watershed GeoJSON as a dictionary."""
        for dictionary in self.data['featurecollection']:
            if dictionary.get('name', '') == 'globalwatershed':
                return dictionary['feature']
        raise LookupError('Could not find "globalwatershed" in the feature'
                          'collection.')

    @property
    def characteristics(self):
        """List the available watershed characteristics.

        Details about these characteristics can be found in the StreamStats
        https://streamstatsags.cr.usgs.gov/ss_defs/basin_char_defs.aspx
        """
        chars = dict((p['code'], p['name']) for p in self.parameters)
        return chars

    def get_characteristic(self, code=None):
        """Retrieve a specified watershed characteristic code.

        Valid codes can be seen as keys in the dictionary returned
        by the characteristics() method.

        Parameters
        ----------
        code : string
            Watershed characteristic code to extract.
        

        Returns
        -------
        characteristic_values : dict
            A dictionary containing specified characteristic's data and metadata
        """
        keys = list(self.characteristics.keys())
        if code not in keys:
            raise ValueError(f"code must be a valid key: {', '.join(keys)}")
        characteristic_index = keys.index(code)
        characteristic_values = self.parameters[characteristic_index]
        return characteristic_values

    
def get_state(lon, lat):
    """Get the state code from US Censue database"""
    import geocoder
    
    try:
        g = geocoder.uscensus([lat, lon], method='reverse')
        return g.geojson['features'][0]['properties']['raw']['States'][0]['STUSAB']
    except KeyError:
        raise KeyError('The location should be inside the US')