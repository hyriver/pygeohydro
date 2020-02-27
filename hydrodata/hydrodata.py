#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The main module for generating an instance of hydrodata.

It can be used as follows:
    >>> from hydrodata import Station
    >>> frankford = Station('2010-01-01', '2015-12-31', station_id='01467087')

For more information refer to the Usage section of the document.
"""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from hydrodata import utils
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

MARGINE = 15


class Station:
    """Download data from the databases.

    Download climate and streamflow observation data from Daymet and USGS,
    respectively. The data is saved to an NetCDF file. Either coords or station_id
    argument should be specified.
    """

    def __init__(
        self, start, end, station_id=None, coords=None, data_dir="data", width=2000,
    ):
        """Initialize the instance.

        Parameters
        ----------
        start : string or datetime
            The starting date of the time period.
        end : string or datetime
            The end of the time period.
        station_id : string, optional
            USGS station ID, defaults to None
        coords : tuple, optional
            Longitude and latitude of the point of interest, defaults to None
        data_dir : string or Path, optional
            Path to the location of climate data, defaults to 'data'
        width : int, optional
            Width of the geotiff image for LULC in pixels, defaults to 2000 px.
        """

        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.width = width

        self.session = utils.retry_requests()

        if station_id is None and coords is not None:
            self.coords = coords
            self.get_id()
        elif coords is None and station_id is not None:
            self.station_id = str(station_id)
            self.get_coords()
        else:
            raise RuntimeError(
                "Either coordinates or station ID" + " should be specified."
            )

        self.lon, self.lat = self.coords

        self.data_dir = Path(data_dir, self.station_id)

        if not self.data_dir.is_dir():
            try:
                os.makedirs(self.data_dir)
            except OSError:
                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + f"Input directory cannot be created: {self.data_dir}"
                )

        self.get_watershed()

        print(self.__repr__())

    def __repr__(self):
        """Print the characteristics of the watershed."""
        msg = f"[ID: {self.station_id}] ".ljust(MARGINE) + f"Watershed: {self.name}\n"
        msg += "".ljust(MARGINE) + f"Coordinates: ({self.lon:.3f}, {self.lat:.3f})\n"
        msg += (
            "".ljust(MARGINE) + f"Altitude: {self.altitude:.0f} m above {self.datum}\n"
        )
        msg += "".ljust(MARGINE) + f"Drainage area: {self.drainage_area:.0f} sqkm."
        return msg

    def get_coords(self):
        """Get coordinates of the station from station ID."""
        url = "https://waterservices.usgs.gov/nwis/site"
        payload = {"format": "rdb", "sites": self.station_id, "hasDataTypeCd": "dv"}
        try:
            r = self.session.get(url, params=payload)
        except HTTPError or ConnectionError or Timeout or RequestException:
            raise

        r_text = r.text.split("\n")
        r_list = [l.split("\t") for l in r_text if "#" not in l]
        station = dict(zip(r_list[0], r_list[2]))

        self.coords = (float(station["dec_long_va"]), float(station["dec_lat_va"]))
        self.altitude = float(station["alt_va"]) * 0.3048  # convert ft to meter
        self.datum = station["alt_datum_cd"]
        self.name = station["station_nm"]

    def get_id(self):
        """Get station ID based on the specified coordinates."""
        import shapely.ops as ops
        import shapely.geometry as geom

        bbox = "".join(
            [
                f"{self.coords[0] - 0.5:.6f},",
                f"{self.coords[1] - 0.5:.6f},",
                f"{self.coords[0] + 0.5:.6f},",
                f"{self.coords[1] + 0.5:.6f}",
            ]
        )
        url = "https://waterservices.usgs.gov/nwis/site"
        payload = {"format": "rdb", "bBox": bbox, "hasDataTypeCd": "dv"}
        try:
            r = self.session.get(url, params=payload)
        except HTTPError:
            raise HTTPError(
                f"No USGS station found within a 50 km radius of ({self.coords[0]}, {self.coords[1]})."
            )

        r_text = r.text.split("\n")
        r_list = [l.split("\t") for l in r_text if "#" not in l]
        r_dict = [dict(zip(r_list[0], st)) for st in r_list[2:]]

        df = pd.DataFrame.from_dict(r_dict).dropna()
        df = df.drop(df[df.alt_va == ""].index)
        df[["dec_lat_va", "dec_long_va", "alt_va"]] = df[
            ["dec_lat_va", "dec_long_va", "alt_va"]
        ].astype("float")
        df = df[df.site_no.apply(len) == 8]

        point = geom.Point(self.coords)
        pts = dict(
            [
                [row.site_no, geom.Point(row.dec_long_va, row.dec_lat_va)]
                for row in df[["site_no", "dec_lat_va", "dec_long_va"]].itertuples()
            ]
        )
        gdf = gpd.GeoSeries(pts)
        nearest = ops.nearest_points(point, gdf.unary_union)[1]
        idx = gdf[gdf.geom_equals(nearest)].index[0]
        station = df[df.site_no == idx]

        self.station_id = station.site_no.values[0]
        self.coords = (station.dec_long_va.values[0], station.dec_lat_va.values[0])
        self.altitude = (
            float(station["alt_va"].values[0]) * 0.3048
        )  # convert ft to meter
        self.datum = station["alt_datum_cd"].values[0]
        self.name = station.station_nm.values[0]

    def get_watershed(self):
        """Download the watershed geometry from the NLDI service."""
        from hydrodata.datasets import NLDI

        geom_file = self.data_dir.joinpath("geometry.gpkg")

        self.watershed = NLDI(station_id=self.station_id)
        self.comid = self.watershed.comid
        self.tributaries = self.watershed.get_river_network(
            navigation="upstreamTributaries"
        )
        self.main_channel = self.watershed.get_river_network(navigation="upstreamMain")

        # drainage area in sq. km
        self.flowlines = self.watershed.get_nhdplus_byid(
            comids=self.tributaries.index.values
        )
        self.drainage_area = self.flowlines.areasqkm.sum()

        if geom_file.exists():
            gdf = gpd.read_file(geom_file)
            self.geometry = gdf.geometry.values[0]
            return

        print(
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + "Downloading watershed geometry using NLDI service >>>"
        )
        geom = self.watershed.basin
        self.geometry = geom.values[0]
        geom.to_file(geom_file)

        print(
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + f"The watershed geometry saved to {geom_file}."
        )
