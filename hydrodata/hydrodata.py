#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The main module for generating an instance of hydrodata.

It can be used as follows:
    >>> from hydrodata import Station
    >>> frankford = Station('2010-01-01', '2015-12-31', station_id='01467087')

For more information refer to the Usage section of the document.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from numba import njit, prange
from hydrodata import utils
import xarray as xr
import json
import os
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException


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
        data_dir="data",
        rain_snow=False,
        tcr=0.0,
        phenology=False,
        width=2000,
    ):
        """Initialize the instance.

        :param start: The starting date of the time period.
        :type start: string or datetime
        :param end: The end of the time period.
        :type end: string or datetime
        :param station_id: USGS station ID, defaults to None
        :type station_id: string, optional
        :param coords: Longitude and latitude of the point of interest, defaults to None
        :type coords: tuple, optional
        :param data_dir: Path to the location of climate data, defaults to 'data'
        :type data_dir: string or Path, optional
        :param rain_snow: Wethere to separate snow from precipitation, defaults to False.
        :type rain_snow: bool, optional
        :param tcr: Critical temperetature for separating snow from precipitation, defaults to 0 (deg C).
        :type tcr: float, optional
        :param width: Width of the geotiff image for LULC in pixels, defaults to 2000 px.
        :type width: int
        """

        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.rain_snow = rain_snow
        self.tcr = tcr
        self.phenology = phenology
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
                    + f"Input directory cannot be created: {d}"
                )

        self.get_watershed()

        print(self.__repr__())

    def __repr__(self):
        msg = f"[ID: {self.station_id}] ".ljust(MARGINE) + f"Watershed: {self.name}\n"
        msg += "".ljust(MARGINE) + f"Coordinates: ({self.lon:.3f}, {self.lat:.3f})\n"
        msg += (
            "".ljust(MARGINE) + f"Altitude: {self.altitude:.0f} m above {self.datum}\n"
        )
        msg += "".ljust(MARGINE) + f"Drainage area: {self.drainage_area:.0f} sqkm."
        return msg

    def get_coords(self):
        """Get coordinates of the station from station ID."""

        # Get altitude of the station
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
        except requests.HTTPError:
            raise requests.HTTPError(
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

    def get_nlcd(self, years=None):
        """Get data from NLCD 2016 database.

        Download land use, land cover, canopy and impervious data from NLCD2016
        database clipped by a given Polygon geometry with epsg:4326 projection.
        Note: NLCD data has a 30 m resolution.

        :param years: The years for NLCD data as a dictionary, defaults to {'impervious': 2016, 'cover': 2016, 'canopy': 2016}.
        :type years: dict, optional

        :returns: Stats of watershed imperviousness, canpoy and cover.
        :rtype: tuple
        """
        from owslib.wms import WebMapService
        import rasterstats
        import fiona
        import rasterio
        import rasterio.mask
        from hydrodata.nlcd_helper import NLCD

        nlcd = NLCD()
        avail_years = {
            "impervious": nlcd.impervious_years,
            "cover": nlcd.cover_years,
            "canopy": nlcd.canopy_years,
        }
        if years is None:
            self.lulc_years = {"impervious": 2016, "cover": 2016, "canopy": 2016}
        else:
            self.lulc_years = years

        def mrlc_url(service):
            if self.lulc_years[service] not in avail_years[service]:
                msg = (
                    f"{service.capitalize()} data for {self.lulc_years[service]} is not in the databse."
                    + "Avaible years are:"
                    + f"{' '.join(str(x) for x in avail_years[service])}"
                )
                raise ValueError(msg)

            if service == "impervious":
                link = f"NLCD_{self.lulc_years[service]}_Impervious_L48"
            elif service == "canopy":
                link = f"NLCD_{self.lulc_years[service]}_Tree_Canopy_L48"
            elif service == "cover":
                link = f"NLCD_{self.lulc_years[service]}_Land_Cover_Science_product_L48"
            return (
                "https://www.mrlc.gov/geoserver/mrlc_download/"
                + link
                + "/wms?service=WMS&request=GetCapabilities"
            )

        urls = {
            "impervious": mrlc_url("impervious"),
            "cover": mrlc_url("cover"),
            "canopy": mrlc_url("canopy"),
        }

        params = {}
        for data_type, url in urls.items():
            data = Path(
                self.data_dir, f"{data_type}_{self.lulc_years[data_type]}.geotiff"
            )
            if Path(data).exists():
                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + f"Using existing {data_type} data file: {data}"
                )
            else:
                bbox = self.geometry.bounds
                height = int(
                    np.abs(bbox[1] - bbox[3]) / np.abs(bbox[0] - bbox[2]) * self.width
                )
                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + f"Downloading {data_type} data from NLCD {self.lulc_years[data_type]} database >>>"
                )
                wms = WebMapService(url, version="1.3.0")
                try:
                    img = wms.getmap(
                        layers=list(wms.contents),
                        srs="epsg:4326",
                        bbox=bbox,
                        size=(self.width, height),
                        format="image/geotiff",
                        transparent=True,
                    )
                except ConnectionError:
                    raise ("Data is not availble on the server.")

                with open(data, "wb") as out:
                    with rasterio.MemoryFile() as memfile:
                        memfile.write(img.read())
                        with memfile.open() as src:
                            out_image, out_transform = rasterio.mask.mask(
                                src, [self.geometry], crop=True
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

                with rasterio.open(data, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + f"{data_type.capitalize()} data was saved to {data}"
                )

            categorical = True if data_type == "cover" else False
            params[data_type] = rasterstats.zonal_stats(
                self.geometry, data, categorical=categorical, category_map=nlcd.legends
            )[0]

        self.impervious = params["impervious"]
        self.canopy = params["canopy"]
        self.cover = params["cover"]


@njit(parallel=True)
def separate_snow(prcp, tmean, tcr=0.0):
    """Separate snow and rain based on a critical temperature.

    The separation is based on a critical temperature (C) with the default
    value of 0 degree C.
    """
    nt = prcp.shape[0]
    pr = np.zeros(nt, np.float64)
    ps = np.zeros(nt, np.float64)
    for t in prange(nt):
        if tmean[t] > tcr:
            pr[t] = prcp[t]
            ps[t] = 0.0
        else:
            pr[t] = 0.0
            ps[t] = prcp[t]
    return pr, ps


def cover_stats(fpath):
    """Compute percentages of the land cover classes and categories.
    
    :param fpath: Path to the cover ``.geotiff`` file.
    :type fpath: string or Path
    :returns: Percentage of NLCD's cover classes and categories
    :rtype: dict
    """
    import rasterio
    from hydrodata.nlcd_helper import NLCD
    import numpy as np

    nlcd = NLCD()

    cover = rasterio.open(fpath)
    total_pix = cover.shape[0] * cover.shape[1]
    cover_arr = cover.read()
    class_percentage = dict(
        zip(
            list(nlcd.legends.values()),
            [
                cover_arr[cover_arr == int(cat)].shape[0] / total_pix * 100.0
                for cat in list(nlcd.legends.keys())
            ],
        )
    )
    masks = [
        [leg in cat for leg in list(nlcd.legends.keys())]
        for cat in list(nlcd.categories.values())
    ]
    cat_list = [np.array(list(class_percentage.values()))[msk].sum() for msk in masks]
    category_percentage = dict(zip(list(nlcd.categories.keys()), cat_list))
    return class_percentage, category_percentage
