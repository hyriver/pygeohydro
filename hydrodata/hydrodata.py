#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The main module for generating an instance of hydrodata.

It can be used as follows:
    >>> from hydrodata import Dataloader
    >>> frankford = Dataloader('2010-01-01', '2015-12-31', station_id='01467087')

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


class Dataloader:
    """Download data from the databases.

    Download climate and streamflow observation data from Daymet and USGS,
    respectively. The data is saved to an HDF5 file. Either coords or station_id
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

        Parameters
        ----------
        start : string or datetime
            The starting date of the time period.
        end : string or datetime
            The end of the time period.
        station_id : string
            USGS station ID
        coords : tuple
            A tuple including longitude and latitude of the point of interest.
        data_dir : string
            Path to the location of climate data. The naming
            convention is data_dir/{watershed name}_climate.nc
        rain_snow : bool
            Wethere to separate snow from precipitation. if True, ``tcr``
            variable can be provided.
        tcr : float
            Critical temperetature for separating snow from precipitation.
            The default value is 0 degree C.
        phenology : bool
            consider phenology for computing PET based on
            Thompson et al., 2011 (https://doi.org/10.1029/2010WR009797)
        width : float
            Width of the geotiff image for LULC in pixels.
            Default is 2000 px. The height is computed automatically
            from the domain's aspect ratio.

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
                print(f"[ID: {self.station_id}] Input directory cannot be created: {d}")

        self.get_watershed()

        # drainage area in sq. km
        self.drainage_area = self.get_characteristic("DRNAREA")["value"] * 2.59

        print(f"[ID: {self.station_id}] {self.__repr__()}")

    def __repr__(self):
        return f"The station is located in the {self.wshed_name} watershed"

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
        self.wshed_name = station["station_nm"]

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
        self.wshed_name = station.station_nm.values[0]

    def get_watershed(self):
        """Download the watershed geometry from the NLDI service."""
        from hydrodata.datasets import NLDI

        geom_file = self.data_dir.joinpath("geometry.gpkg")

        self.watershed = NLDI(station_id=self.station_id)
        self.comid = self.watershed.comid

        if geom_file.exists():
            gdf = gpd.read_file(geom_file)
            self.geometry = gdf.geometry.values[0]
            return

        print(
            f"[ID: {self.station_id}] Downloading watershed geometry using NLDI service >>>"
        )
        geom = self.watershed.basin
        self.geometry = geom.values[0]
        geom.to_file(geom_file)

        print(f"[ID: {self.station_id}] The watershed geometry saved to {geom_file}.")

    def get_climate(self):
        """Get climate data from the Daymet database.

        The function first downloads climate data then computes potential
        evapotranspiration using ETo python package. Then downloads streamflow
        data from USGS database and saves the data as an HDF5 file and return
        it as a Pandas dataframe. The naming convention for the HDF5 file is
        <station_id>_<start>_<end>.nc. If the files already exits on the disk
        it is read and returned as a Pandas dataframe.
        """
        import json
        import daymetpy
        import eto

        fname = (
            "_".join([self.start.strftime("%Y%m%d"), self.end.strftime("%Y%m%d")])
            + ".nc"
        )
        self.clm_file = self.data_dir.joinpath(fname)

        if self.clm_file.exists():
            print(
                f"[ID: {self.station_id}] Using existing climate data file: {self.clm_file}"
            )
            self.climate = xr.open_dataset(self.clm_file)
            return

        print(
            f"[ID: {self.station_id}] Downloading climate data from the Daymet database >>>"
        )
        climate = daymetpy.daymet_timeseries(
            lon=round(self.lon, 6),
            lat=round(self.lat, 6),
            start_year=self.start.year,
            end_year=self.end.year,
        )
        climate.drop("year", inplace=True, axis=1)
        climate = climate[self.start : self.end]
        climate["tmean"] = climate[["tmin", "tmax"]].mean(axis=1)

        print(
            f"[ID: {self.station_id}] Computing potential evapotranspiration (PET) using FAO method"
        )
        df = climate[["tmax", "tmin", "vp"]].copy()
        df.columns = ["T_max", "T_min", "e_a"]
        df["R_s"] = climate.srad * climate.dayl * 1e-6  # to MJ/m2
        df["e_a"] *= 1e-3  # to kPa

        et1 = eto.ETo()
        freq = "D"
        et1.param_est(
            df[["R_s", "T_max", "T_min", "e_a"]],
            freq,
            self.altitude,
            self.lat,
            self.lon,
        )
        climate["pet"] = et1.eto_fao()

        # Multiply PET by growing season index, GSI, for phenology
        # (Thompson et al., 2011) 
        # https://doi.org/10.1029/2010WR009797
        if self.phenology:
            print(f"[ID: {self.station_id}] Considering phenology in PET")
            tmax, tmin = 10.0, -5.0
            trng = 1.0 / (tmax - tmin)

            def gsi(row):
                if row.tmean < tmin:
                    return 0
                elif row.tmean > tmax:
                    return row.pet
                else:
                    return (row.tmean - tmin) * trng * row.pet

            climate["pet"] = climate.apply(gsi, axis=1)

        print(
            f"[ID: {self.station_id}] Downloading stream flow data from USGS database >>>"
        )
        url = "https://waterservices.usgs.gov/nwis/dv/?format=json"
        payload = {
            "sites": self.station_id,
            "startDT": self.start.strftime("%Y-%m-%d"),
            "endDT": self.end.strftime("%Y-%m-%d"),
            "parameterCd": "00060",
            "siteStatus": "all",
        }
        err = pd.read_html("https://waterservices.usgs.gov/rest/DV-Service.html")[0]

        try:
            r = self.session.get(url, params=payload)
        except HTTPError:
            print(
                f"[ID: {self.station_id}] {err[err['HTTP Error Code'] == r.status_code].Explanation.values[0]}"
            )
            raise
        except ConnectionError or Timeout or RequestException:
            raise

        df = json.loads(r.text)
        df = df["value"]["timeSeries"][0]["values"][0]["value"]
        df = pd.DataFrame.from_dict(df, orient="columns")
        df["dateTime"] = pd.to_datetime(df["dateTime"], format="%Y-%m-%dT%H:%M:%S")
        df.set_index("dateTime", inplace=True)
        # Convert cfs to cms
        climate["qobs"] = df.value.astype("float64") * 0.028316846592

        climate = climate[["prcp", "tmin", "tmax", "tmean", "pet", "qobs"]]

        if self.rain_snow:
            print(f"[ID: {self.station_id}] Separating snow from precipitation")
            rain, snow = separate_snow(
                climate["prcp"].values, climate["tmean"].values, self.tcr
            )

            climate["rain"], climate["snow"] = rain, snow

        climate = climate[self.start : self.end].dropna()

        df = xr.Dataset.from_dataframe(climate)

        df.prcp.attrs["units"] = "mm/day"
        df[["tmin", "tmax", "tmean"]].attrs["units"] = "C"
        df.pet.attrs["units"] = "mm"
        df.qobs.attrs["units"] = "cms"
        if self.rain_snow:
            df[["rain", "snow"]].attrs["units"] = "mm/day"

        df.attrs["alt_meter"] = self.altitude
        df.attrs["datum"] = self.datum
        df.attrs["name"] = self.wshed_name
        df.attrs["id"] = self.station_id
        df.attrs["lon"] = self.lon
        df.attrs["lat"] = self.lat

        df.to_netcdf(self.clm_file)
        self.climate = df
        print(
            f"[ID: {self.station_id}] " + f"Climate data was saved to {self.clm_file}"
        )

    def get_nlcd(self, years=None):
        """Get data from NLCD 2016 database.

        Download land use, land cover, canopy and impervious data from NLCD2016
        database clipped by a given Polygon geometry with epsg:4326 projection.
        Note: NLCD data has a 30 m resolution.

        Parameters
        ----------
        years : dict
            The years for NLCD data as a dictionary.
            The default is {'impervious': 2016, 'cover': 2016, 'canopy': 2016}.

        Returns
        -------
        impervious : dict
            A dictionary containing min, max, mean and count of the
            imperviousness of the watershed
        canpoy : dict
            A dictionary containing min, max, mean and count of the canpoy
            of the watershed
        cover : dataframe
            A dataframe containing watershed's land coverage percentage.
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
                    f"[ID: {self.station_id}] Using existing {data_type} data file: {data}"
                )
            else:
                bbox = self.geometry.bounds
                height = int(
                    np.abs(bbox[1] - bbox[3]) / np.abs(bbox[0] - bbox[2]) * self.width
                )
                print(
                    f"[ID: {self.station_id}] Downloading {data_type} data from NLCD {self.lulc_years[data_type]} database >>>"
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
                    f"[ID: {self.station_id}] "
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
    
    Parameters
    ----------
    fpath : string or Path
        Path to the cover ``.geotiff`` file.
    
    Returns
    -------
    class_percentage : dict
        Percentage of all the 20 NLCD's cover classes
    category_percentage : dict
        Percentage of all the 9 NLCD's cover categories
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


def plot(
    Q_dict, prcp, drainage_area, title, figsize=(13, 12), threshold=1e-3, output=None
):
    """Plot hydrological signatures with precipitation as the second axis.

    Plots includes daily, monthly and annual hydrograph as well as
    regime curve (monthly mean) and flow duration curve. The input
    discharges are converted from cms to mm/day based on the watershed
    area.

    Parameters
    ----------
    daily_dict : dict or dataframe
        A series containing daily discharges in m$^3$/s.
        A series or a dictionary of series can be passed where its keys
        are the labels and its values are the series.
    prcp : Series
        A series containing daily precipitation in mm/day
    drainage_area : float
        Watershed drainage area in km$^2$
    title : string
        Plot's supertitle
    figsize : tuple
        Width and height of the plot in inches. The default is (8, 10)
    threshold : float
        The threshold for cutting off the discharge for the flow duration
        curve to deal with log 0 issue. The default is 1e-3.
    output : string
        Path to save the plot as png. The default is `None` which means
        the plot is not saved to a file.
    """
    from hydrodata.plotter import plot

    plot(
        Q_dict,
        prcp,
        drainage_area,
        title,
        figsize=figsize,
        threshold=threshold,
        output=output,
    )
    return


def plot_discharge(
    Q_dict,
    drainage_area,
    title="Streaflow data for the watersheds",
    figsize=(13, 12),
    threshold=1e-3,
    output=None,
):
    """Plot hydrological signatures without precipitation.

    The plots include daily, monthly and annual hydrograph as well as
    regime curve (monthly mean) and flow duration curve.

    Parameters
    ----------
    daily_dict : dict or series
        A series containing daily discharges in m$^3$/s.
        A series or a dictionary of series can be passed where its keys
        are the labels and its values are the series.
    drainage_area : float
        Watershed drainage area in km$^2$
    title : string
        Plot's supertitle.
    figsize : tuple
        Width and height of the plot in inches. The default is (8, 10)
    threshold : float
        The threshold for cutting off the discharge for the flow duration
        curve to deal with log 0 issue. The default is 1e-3.
    output : string
        Path to save the plot as png. The default is `None` which means
        which means the plot is not saved to a file.
    """
    from hydrodata.plotter import plot_discharge

    plot_discharge(
        Q_dict,
        drainage_area,
        title,
        figsize=figsize,
        threshold=threshold,
        output=output,
    )
    return


def read_climate(fpath):
    import h5py

    fpath = Path(fpath)
    with h5py.File(fpath, "r") as f:
        climate = pd.DataFrame(f["c"])
    climate.columns = [
        "prcp (mm/day)",
        "tmin (C)",
        "tmax (C)",
        "tmean (C)",
        "pet (mm)",
        "qobs (cms)",
    ]
    # daymet doesn't account for leap years.
    # It removes Dec 31 when leap year.
    index = pd.date_range(self.start, self.end)
    nl = index[~index.is_leap_year]
    lp = index[
        (index.is_leap_year) & (~index.strftime("%Y-%m-%d").str.endswith("12-31"))
    ]
    index = index[(index.isin(nl)) | (index.isin(lp))]
    self.climate.index = index
