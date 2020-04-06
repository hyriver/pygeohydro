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
import hydrodata.datasets as hds
import numpy as np
import pandas as pd
from hydrodata import utils

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

        info = hds.nwis_siteinfo(ids=self.station_id, expanded=True)
        try:
            self.drainage_area = (
                info["contrib_drain_area_va"].astype("float64").values[0] * 2.5899
            )
        except ValueError:
            self.drainage_area = (
                info["drain_area_va"].astype("float64").values[0] * 2.5899
            )
        except ValueError:
            self.drainage_area = self.watershed.flowlines.areasqkm.sum()

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
        siteinfo = hds.nwis_siteinfo(ids=self.station_id)
        st = siteinfo[siteinfo.stat_cd == "00003"]
        st_begin = st.begin_date.values[0]
        st_end = st.end_date.values[0]
        if self.start < st_begin or self.end > st_end:
            msg = (
                f"[ID: {self.station_id}] ".ljust(MARGINE)
                + "Daily Mean data unavailable for the specified time period."
                + " The data is available from "
                + f"{np.datetime_as_string(st_begin, 'D')} to "
                + f"{np.datetime_as_string(st_end, 'D')}."
            )
            raise ValueError(msg)

        self.coords = (
            st["dec_long_va"].astype("float64").values[0],
            st["dec_lat_va"].astype("float64").values[0],
        )
        self.altitude = (
            st["alt_va"].astype("float64").values[0] * 0.3048
        )  # convert ft to meter
        self.datum = st["alt_datum_cd"].values[0]
        self.name = st.station_nm.values[0]

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
        df = hds.nwis_siteinfo(bbox=bbox)
        df = df[df.state_cd == "0003"]
        if len(df) < 1:
            msg = (
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "No USGS station were found within a 50-km radius with daily mean streamflow."
            )
            raise ValueError(msg)

        point = geom.Point(self.coords)
        pts = dict(
            [
                [sid, geom.Point(lon, lat)]
                for sid, lon, lat in df[
                    ["site_no", "dec_lat_va", "dec_long_va"]
                ].itertuples(name=None)
            ]
        )
        gdf = gpd.GeoSeries(pts)
        nearest = ops.nearest_points(point, gdf.unary_union)[1]
        idx = gdf[gdf.geom_equals(nearest)].index[0]
        station = df[df.site_no == idx]
        st_begin = station.begin_date.values[0]
        st_end = station.end_date.values[0]
        if self.start < st_begin or self.end > st_end:
            msg = (
                f"[ID: {self.station_id}] ".ljust(MARGINE)
                + "Daily Mean data unavailable for the specified time period."
                + " The data is available from "
                + f"{np.datetime_as_string(st_begin, 'D')} to "
                + f"{np.datetime_as_string(st_end, 'D')}."
            )
            raise ValueError(msg)

        self.station_id = station.site_no.values[0]
        self.coords = (
            station.dec_long_va.astype("float64").values[0],
            station.dec_lat_va.astype("float64").values[0],
        )
        self.altitude = (
            station["alt_va"].astype("float64").values[0] * 0.3048
        )  # convert ft to meter
        self.datum = station["alt_datum_cd"].values[0]
        self.name = station.station_nm.values[0]

    def get_watershed(self):
        """Download the watershed geometry from the NLDI service."""

        geom_file = self.data_dir.joinpath("geometry.gpkg")

        if geom_file.exists():
            print(
                f"[ID: {self.station_id}] ".ljust(MARGINE)
                + f"Using existing watershed geometry: {geom_file}"
            )
            self.basin = gpd.read_file(geom_file)
        else:
            print(
                f"[ID: {self.station_id}] ".ljust(MARGINE)
                + "Downloading watershed geometry using NLDI service >>>"
            )
            self.basin = hds.NLDI.basin(self.station_id)
            self.basin.to_file(geom_file)
            print(
                f"[ID: {self.station_id}] ".ljust(MARGINE)
                + f"The watershed geometry saved to {geom_file}."
            )

        self.geometry = self.basin.geometry.values[0]
