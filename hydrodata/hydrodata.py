#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The main module for generating an instance of hydrodata.

It can be used as follows:
    >>> from hydrodata import Station
    >>> wshed = Station('2010-01-01', '2015-12-31', station_id='01467087')

For more information refer to the Usage section of the document.
"""

import os
from pathlib import Path
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd

import hydrodata.datasets as hds
from hydrodata import utils

MARGINE = 15


class Station:
    """Download data from the databases.

    Download climate and streamflow observation data from Daymet and USGS,
    respectively. The data is saved to an NetCDF file. Either coords or station_id
    argument should be specified.
    """

    def __init__(
        self,
        start,
        end,
        station_id=None,
        coords=None,
        srad=0.5,
        data_dir="data",
        verbose=False,
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
        srad : float, optional
            Search radius in degrees for finding the closest station
            when coords is given, default to 0.5 degrees
        data_dir : string or Path, optional
            Path to the location of climate data, defaults to 'data'
        verbose : bool
            Whether to show messages
        """

        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.verbose = verbose

        self.session = utils.retry_requests()

        if station_id is None and coords is not None:
            self.coords = coords
            self.srad = srad
            self.get_id()
        elif coords is None and station_id is not None:
            self.station_id = str(station_id)
            self.get_coords()
        else:
            msg = (
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "Either coordinates or station ID should be specified."
            )
            raise ValueError(msg)

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
                info["contrib_drain_area_va"].astype("float64").to_numpy()[0] * 2.5899
            )
        except ValueError:
            self.drainage_area = (
                info["drain_area_va"].astype("float64").to_numpy()[0] * 2.5899
            )
        except ValueError:
            self.drainage_area = self.watershed.flowlines.areasqkm.sum()

        self.hcdn = info.hcdn_2009.to_numpy()[0]
        if self.verbose:
            print(self.__repr__())

    def __repr__(self):
        """Print the characteristics of the watershed."""
        return (
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + f"Watershed: {self.name}\n"
            + "".ljust(MARGINE)
            + f"Coordinates: ({self.lon:.3f}, {self.lat:.3f})\n"
            + "".ljust(MARGINE)
            + f"Altitude: {self.altitude:.0f} m above {self.datum}\n"
            + "".ljust(MARGINE)
            + f"Drainage area: {self.drainage_area:.0f} sqkm\n"
            + "".ljust(MARGINE)
            + f"HCDN 2009: {self.hcdn}."
        )

    def get_coords(self):
        """Get coordinates of the station from station ID."""
        siteinfo = hds.nwis_siteinfo(ids=self.station_id)
        st = siteinfo[siteinfo.stat_cd == "00003"]
        st_begin = st.begin_date.to_numpy()[0]
        st_end = st.end_date.to_numpy()[0]
        if self.start < st_begin or self.end > st_end:
            warn(
                f"[ID: {self.station_id}] ".ljust(MARGINE)
                + "Daily Mean data unavailable for the specified time period."
                + " The data is available from "
                + f"{np.datetime_as_string(st_begin, 'D')} to "
                + f"{np.datetime_as_string(st_end, 'D')}."
            )

        self.coords = (
            st["dec_long_va"].astype("float64").to_numpy()[0],
            st["dec_lat_va"].astype("float64").to_numpy()[0],
        )
        self.altitude = (
            st["alt_va"].astype("float64").to_numpy()[0] * 0.3048
        )  # convert ft to meter
        self.datum = st["alt_datum_cd"].to_numpy()[0]
        self.name = st.station_nm.to_numpy()[0]

    def get_id(self):
        """Get station ID based on the specified coordinates."""
        import shapely.geometry as geom

        bbox = [
            self.coords[0] - self.srad,
            self.coords[1] - self.srad,
            self.coords[0] + self.srad,
            self.coords[1] + self.srad,
        ]

        sites = hds.nwis_siteinfo(bbox=bbox)
        sites = sites[sites.stat_cd == "00003"]
        if len(sites) < 1:
            msg = (
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "No USGS station were found within a "
                + f"{int(self.srad * 111 / 10) * 10}-km radius "
                + f"of ({self.coords[0]}, {self.coords[1]}) with daily mean streamflow."
            )
            raise ValueError(msg)

        point = geom.Point(self.coords)
        pts = dict(
            [
                [sid, geom.Point(lon, lat)]
                for sid, lon, lat in sites[
                    ["site_no", "dec_long_va", "dec_lat_va"]
                ].itertuples(name=None, index=False)
            ]
        )

        stations = gpd.GeoSeries(pts)
        distance = stations.apply(lambda x: x.distance(point)).sort_values()

        station_id = None
        for sid, dis in distance.iteritems():
            station = sites[sites.site_no == sid]
            st_begin = station.begin_date.to_numpy()[0]
            st_end = station.end_date.to_numpy()[0]

            if self.start < st_begin or self.end > st_end:
                continue
            else:
                station_id = sid
                break

        if station_id is None:
            msg = (
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "No USGS station were found within a "
                + f"{int(self.srad * 111 / 10) * 10}-km radius "
                + f"with daily mean streamflow from {self.start} to {self.end}"
                + ".\nUse utils.interactive_map(bbox) function to explore "
                + "the available stations in the area."
            )
            raise ValueError(msg)
        else:
            self.station_id = station_id

        self.coords = (
            station.dec_long_va.astype("float64").to_numpy()[0],
            station.dec_lat_va.astype("float64").to_numpy()[0],
        )
        self.altitude = (
            station["alt_va"].astype("float64").to_numpy()[0] * 0.3048
        )  # convert ft to meter
        self.datum = station["alt_datum_cd"].to_numpy()[0]
        self.name = station.station_nm.to_numpy()[0]

    def get_watershed(self):
        """Download the watershed geometry from the NLDI service."""

        geom_file = self.data_dir.joinpath("geometry.gpkg")

        if geom_file.exists():
            if self.verbose:
                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + f"Using existing watershed geometry: {geom_file}"
                )
            self.basin = gpd.read_file(geom_file)
        else:
            if self.verbose:
                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + "Downloading watershed geometry using NLDI service >>>"
                )
            self.basin = hds.NLDI.basin(self.station_id)
            self.basin.to_file(geom_file)

            if self.verbose:
                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + f"The watershed geometry saved to {geom_file}."
                )

        self.geometry = self.basin.geometry.to_numpy()[0]
