"""Accessing data from the supported databases through their APIs."""

import io
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from shapely.geometry import Polygon

from hydrodata import connection, helpers, services, utils
from hydrodata.connection import RetrySession
from hydrodata.exceptions import (
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingInputs,
    ZeroMatched,
)
from hydrodata.services import WFS

MARGINE = 15


def nwis_streamflow(station_ids: Union[List[str], str], dates: Tuple[str, str]) -> pd.DataFrame:
    """Get daily streamflow observations from USGS.

    Parameters
    ----------
    station_ids : str, list
        The gage ID(s)  of the USGS station.
    dates : tuple
        Start and end dates as a tuple (start, end).

    Returns
    -------
    pandas.DataFrame
        Streamflow data observations in cubic meter per second (cms)
    """

    if isinstance(station_ids, str):
        station_ids = [station_ids]
    elif isinstance(station_ids, list):
        station_ids = [str(i) for i in station_ids]
    else:
        raise InvalidInputType("ids", "str or list")

    if isinstance(dates, tuple) and len(dates) == 2:
        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])
    else:
        raise InvalidInputType("dates", "tuple", "(start, end)")

    siteinfo = nwis_siteinfo(station_ids)
    check_dates = siteinfo.loc[
        (siteinfo.stat_cd == "00003") & (start < siteinfo.begin_date) & (end > siteinfo.end_date),
        "site_no",
    ].tolist()
    nas = [s for s in station_ids if s in check_dates]
    if len(nas) > 0:
        raise InvalidInputRange(
            "Daily Mean data unavailable for the specified time "
            + "period for the following stations:\n"
            + ", ".join(str(s) for s in nas)
        )

    url = "https://waterservices.usgs.gov/nwis/dv"
    payload = {
        "format": "json",
        "sites": ",".join(str(s) for s in station_ids),
        "startDT": start.strftime("%Y-%m-%d"),
        "endDT": end.strftime("%Y-%m-%d"),
        "parameterCd": "00060",
        "statCd": "00003",
        "siteStatus": "all",
    }

    r = RetrySession().post(url, payload)

    ts = r.json()["value"]["timeSeries"]
    r_ts = {t["sourceInfo"]["siteCode"][0]["value"]: t["values"][0]["value"] for t in ts}

    def to_df(col, dic):
        q = pd.DataFrame.from_records(dic, exclude=["qualifiers"], index=["dateTime"])
        q.index = pd.to_datetime(q.index)
        q.columns = [col]
        q[col] = q[col].astype("float64") * 0.028316846592  # Convert cfs to cms
        return q

    qobs = pd.concat([to_df(f"USGS-{s}", t) for s, t in r_ts.items()], axis=1)
    return qobs


def nwis_siteinfo(
    ids: Optional[Union[str, List[str]]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    expanded: bool = False,
) -> pd.DataFrame:
    """Get NWIS stations by a list of IDs or within a bounding box.

    Only stations that record(ed) daily streamflow data are returned.
    The following columns are included in the dataframe with expanded
    set to False:

    ==================  ==================================
    Name                Description
    ==================  ==================================
    site_no             Site identification number
    station_nm          Site name
    site_tp_cd          Site type
    dec_lat_va          Decimal latitude
    dec_long_va         Decimal longitude
    coord_acy_cd        Latitude-longitude accuracy
    dec_coord_datum_cd  Decimal Latitude-longitude datum
    alt_va              Altitude of Gage/land surface
    alt_acy_va          Altitude accuracy
    alt_datum_cd        Altitude datum
    huc_cd              Hydrologic unit code
    parm_cd             Parameter code
    stat_cd             Statistical code
    ts_id               Internal timeseries ID
    loc_web_ds          Additional measurement description
    medium_grp_cd       Medium group code
    parm_grp_cd         Parameter group code
    srs_id              SRS ID
    access_cd           Access code
    begin_date          Begin date
    end_date            End date
    count_nu            Record count
    hcdn_2009           Whether is in HCDN-2009 stations
    ==================  ==================================

    Parameters
    ----------
    ids : str or list
        Station ID(s)
    bbox : tuple
        Coordinates of bounding box in this order (west, south, east, north)
    expanded : bool, optional
        Whether to get expanded sit information for example drainage area.

    Returns
    -------
    pandas.DataFrame
        NWIS stations
    """
    if (bbox is None and ids is None) or (bbox is not None and ids is not None):
        raise MissingInputs("Either ids or bbox argument should be provided.")

    if ids is None:
        if isinstance(bbox, tuple) and len(bbox) == 4:
            query = {"bBox": ",".join(f"{b:.06f}" for b in bbox)}
        else:
            raise InvalidInputType("bbox", "tuple", "(west, south, east, north)")
    else:
        if isinstance(ids, str):
            query = {"sites": ids}
        elif isinstance(ids, list):
            query = {"sites": ",".join(str(i) for i in ids)}
        else:
            raise InvalidInputType("ids", "str or list")

    url = "https://waterservices.usgs.gov/nwis/site"

    outputType = {"siteOutput": "expanded"} if expanded else {"outputDataTypeCd": "dv"}

    payload = {
        **query,
        **outputType,
        "format": "rdb",
        "parameterCd": "00060",
        "siteStatus": "all",
        "hasDataTypeCd": "dv",
    }

    r = RetrySession().post(url, payload)

    r_text = r.text.split("\n")
    r_list = [txt.split("\t") for txt in r_text if "#" not in txt]
    r_dict = [dict(zip(r_list[0], st)) for st in r_list[2:]]

    sites = pd.DataFrame.from_dict(r_dict).dropna()
    sites = sites.drop(sites[sites.alt_va == ""].index)
    try:
        sites = sites[sites.parm_cd == "00060"]
        sites["begin_date"] = pd.to_datetime(sites["begin_date"])
        sites["end_date"] = pd.to_datetime(sites["end_date"])
    except AttributeError:
        pass

    int_cols = ["dec_lat_va", "dec_long_va", "alt_va"]
    sites[int_cols] = sites[int_cols].astype("float64")

    sites = sites[sites.site_no.apply(len) == 8]

    gii = WaterData("gagesii", "epsg:900913")
    hcdn = gii.getfeature_byid("staid", sites.site_no.tolist())
    hcdn_dict = hcdn[["staid", "hcdn_2009"]].set_index("staid").hcdn_2009.to_dict()
    sites["hcdn_2009"] = sites.site_no.apply(
        lambda x: len(hcdn_dict[x]) > 0 if x in hcdn_dict.keys() else False
    )

    return sites


class WaterData:
    """Access to `Water Data <https://labs.waterdata.usgs.gov/geoserver/web/wicket/bookmarkable/org.geoserver.web.demo.MapPreviewPage?2>`__ service.

    Parameters
    ----------
    layer : str
        A valid layer from the WaterData service. Valid layers are:
        ``nhdarea``, ``nhdwaterbody``, ``catchmentsp``, ``nhdflowline_network``
        ``gagesii``, ``huc08``, ``huc12``, ``huc12agg``, and ``huc12all``. Note that
        the layers' worksapce for the Water Data service is ``wmadata`` which will
        be added to the given ``layer`` argument if it is not provided.
    crs : str, optional
        The spatial reference system for requesting the data. Each layer support
        a limited number of CRSs, defaults to ``epsg:4269``.
    """

    def __init__(self, layer: str, crs: str = "epsg:4269") -> None:

        self.layer = layer if ":" in layer else f"wmadata:{layer}"
        self.crs = crs

        self.wfs = WFS(
            "https://labs.waterdata.usgs.gov/geoserver/wmadata/ows",
            layer=self.layer,
            outFormat="application/json",
            version="2.0.0",
            crs=self.crs,
        )

    def __repr__(self) -> str:
        """Print the services properties."""
        return (
            "Connected to the WFS service with the following properties:\n"
            + f"URL: {self.wfs.url}\n"
            + f"Version: {self.wfs.version}\n"
            + f"Layer: {self.wfs.layer}\n"
            + f"Output Format: {self.wfs.outFormat}\n"
            + f"Output CRS: {self.wfs.crs}"
        )

    def getfeature_bybox(
        self, bbox: Tuple[float, float, float, float], box_crs: str = "epsg:4326",
    ) -> gpd.GeoDataFrame:
        """Get NHDPlus flowlines database within a bounding box.

        Parameters
        ----------
        bbox : list
            The bounding box for the region of interest in WGS 83, defaults to None.
            The list should provide the corners in this order:
            [west, south, east, north]
        box_crs : str, optional
            The spatial reference system of the input bbox, defaults to
            epsg:4326.

        Returns
        -------
        geopandas.GeoDataFrame
            NHDPlus features
        """

        r = self.wfs.getfeature_bybox(bbox, box_crs=box_crs)
        features = utils.json_togeodf(r.json(), self.crs, self.crs)

        if features.shape[0] == 0:
            raise ZeroMatched(
                f"No feature was found in bbox({', '.join(str(round(x, 3)) for x in bbox)})"
            )

        return features

    def getfeature_byid(
        self, property_name: str, property_ids: Union[List[str], str],
    ) -> gpd.GeoDataFrame:
        """Get flowlines or catchments from NHDPlus V2 based on ComIDs.

        Parameters
        ----------
        property_name : str
            Property (column) name of the requested features in the database.
            You can use ``wfs.get_validnames()`` class function to get all
            the available column names for a specific layer.
        property_ids : str or list
            The ID(s) of the requested property name.

        Returns
        -------
        geopandas.GeoDataFrame
            NHDPlus features
        """

        r = self.wfs.get_validnames()
        valid_names = utils.json_togeodf(r.json(), self.crs, self.crs)

        if property_name not in valid_names:
            raise InvalidInputValue("property name", valid_names)

        r = self.wfs.getfeature_byid(property_name, property_ids, filter_spec="2.0")
        rjson = r.json()
        if rjson["numberMatched"] == 0:
            raise ZeroMatched(
                f"No feature was found in {self.layer} " + "layer that matches the given ID(s)."
            )

        return utils.json_togeodf(rjson, self.crs, self.crs)


class NLDI:
    """Access to the Hydro Network-Linked Data Index (NLDI) service."""

    def __init__(self) -> None:

        self.base_url = "https://labs.waterdata.usgs.gov/api/nldi/linked-data"
        self.session = RetrySession()
        r = self.session.get(self.base_url).json()
        self.valid_sources = [el for sub in utils.traverse_json(r, ["source"]) for el in sub]

    def getfeature_byid(
        self, fsource: str, fid: str, basin: bool = False, url_only: bool = False
    ) -> gpd.GeoDataFrame:
        """Get features of a single id

        Parameters
        ----------
        fsource : str
            The name of feature source. The valid sources are:
            comid, huc12pp, nwissite, wade, WQP
        fid : str
            The ID of the feature.
        basin : bool
            Whether to return the basin containing the feature.
        url_only : bool
            Whether to only return the generated url, defaults to False.
            It's intended for preparing urls for batch download.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features
        """

        if fsource not in self.valid_sources:
            raise InvalidInputValue("feature source", self.valid_sources)

        url = "/".join([self.base_url, fsource, fid])
        if basin:
            url += "/basin"

        if url_only:
            return url

        return utils.json_togeodf(self.session.get(url).json(), "epsg:4269", "epsg:4326")

    def navigate_byid(
        self,
        fsource: str,
        fid: str,
        navigation: str,
        source: Optional[str] = None,
        distance: Optional[int] = None,
        url_only: bool = False,
    ) -> gpd.GeoDataFrame:
        """Navigate the NHDPlus databse from a single feature id

        Parameters
        ----------
        fsource : str
            The name of feature source. The valid sources are:
            comid, huc12pp, nwissite, wade, WQP.
        fid : str
            The ID of the feature.
        navigation : str
            The navigation method.
        source : str, optional
            Return the data from another source after navigating
            the features using fsource, defaults to None.
        distance : int, optional
            Limit the search for navigation up to a distance, defaults to None.
        url_only : bool
            Whether to only return the generated url, defaults to False.
            It's intended for  preparing urls for batch download.

        Returns
        -------
        geopandas.GeoDataFrame
            NLDI indexed features
        """

        if fsource not in self.valid_sources:
            raise InvalidInputValue("feature source", self.valid_sources)

        url = "/".join([self.base_url, fsource, fid, "navigate"])

        valid_navigations = self.session.get(url).json()
        if navigation not in valid_navigations.keys():
            raise InvalidInputValue("navigation", valid_navigations.keys())

        url = valid_navigations[navigation]

        if source is not None:
            if source not in self.valid_sources:
                raise InvalidInputValue("source", self.valid_sources)
            url += f"/{source}"

        if distance is not None:
            url += f"?distance={int(distance)}"

        if url_only:
            return url

        return utils.json_togeodf(self.session.get(url).json(), "epsg:4269", "epsg:4326")


class Daymet:
    """Base class for Daymet requests

    Parameters
    ----------
    dates : tuple, optional
        Start and end dates as a tuple, (start, end), default to None.
    years : int or list or tuple, optional
        List of year(s), default to None.
    variables : str or list or tuple, optional
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found in https://daymet.ornl.gov/overview.
        Defaults to None i.e., all the variables are downloaded.
    pet : bool, optional
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False
    """

    def __init__(
        self,
        dates: Optional[Tuple[str, str]] = None,
        years: Optional[Union[List[int], int]] = None,
        variables: Optional[Union[List[str], str]] = None,
        pet: bool = False,
    ) -> None:
        self.session = RetrySession()

        if (years is None and dates is None) or (years is not None and dates is not None):
            raise MissingInputs("Either years or dates arguments should be provided.")

        if years is None:
            if isinstance(dates, tuple) and len(dates) == 2:
                start = pd.to_datetime(dates[0])
                end = pd.to_datetime(dates[1])
            else:
                raise InvalidInputType("dates", "tuple", "(start, end)")

            if start < pd.to_datetime("1980-01-01"):
                raise InvalidInputRange("Daymet database ranges from 1980 to 2019.")

            self.date_dict = {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
            }
        elif dates is None:
            years = years if isinstance(years, (list, tuple)) else [years]
            self.date_dict = {"years": ",".join(str(x) for x in years)}

        vars_table = helpers.daymet_variables()

        self.units = dict(zip(vars_table["Abbr"], vars_table["Units"]))

        valid_variables = vars_table.Abbr.to_list()
        if variables is None:
            self.variables = valid_variables
        else:
            self.variables = variables if isinstance(variables, list) else [variables]

            if not all(v for v in variables if v not in valid_variables):
                raise InvalidInputValue("variables", valid_variables)

            if pet:
                reqs = ("tmin", "tmax", "vp", "srad", "dayl")
                self.variables = list(set(reqs) | set(variables))


def daymet_byloc(
    coords: Tuple[float, float],
    dates: Optional[Tuple[str, str]] = None,
    years: Optional[List[int]] = None,
    variables: Optional[Union[List[str], str]] = None,
    pet: bool = False,
) -> pd.DataFrame:
    """Get daily climate data from Daymet for a single point.

    Parameters
    ----------
    coords : tuple
        Longitude and latitude of the location of interest as a tuple (lon, lat)
    dates : tuple, optional
        Start and end dates as a tuple (start, end), default to None.
    years : int or list or tuple, optional
        List of year(s), default to None.
    variables : str or list or tuple, optional
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found in https://daymet.ornl.gov/overview.
        Defaults to None i.e., all the variables are downloaded.
    pet : bool, optional
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False

    Returns
    -------
    pandas.DataFrame
        Daily climate data for a location
    """
    daymet = Daymet(dates, years, variables, pet)

    if isinstance(coords, tuple) and len(coords) == 2:
        lon, lat = coords
    else:
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

    if not ((14.5 < lat < 52.0) or (-131.0 < lon < -53.0)):
        raise InvalidInputRange(
            "The location is outside the Daymet dataset. "
            + "The acceptable range is: "
            + "14.5 < lat < 52.0 and -131.0 < lon < -53.0"
        )

    url = "https://daymet.ornl.gov/single-pixel/api/data"

    payload = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "vars": ",".join(v for v in daymet.variables),
        "format": "json",
        **daymet.date_dict,
    }

    r = daymet.session.get(url, payload)

    clm = pd.DataFrame(r.json()["data"])
    clm.index = pd.to_datetime(clm.year * 1000.0 + clm.yday, format="%Y%j")
    clm = clm.drop(["year", "yday"], axis=1)

    if pet:
        clm = utils.pet_fao_byloc(clm, coords)
    return clm


def daymet_bygeom(
    geometry: Polygon,
    geo_crs: str = "epsg:4326",
    dates: Optional[Tuple[str, str]] = None,
    years: Optional[List[int]] = None,
    variables: Optional[List[str]] = None,
    pet: bool = False,
    fill_holes: bool = False,
    n_threads: int = 8,
) -> xr.Dataset:
    """Gridded data from the Daymet database as 1-km resolution.

    The data is clipped using NetCDF Subset Service.

    Parameters
    ----------
    geometry : shapely.geometry.Polygon
        The geometry of the region of interest.
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    dates : tuple, optional
        Start and end dates as a tuple (start, end), default to None.
    years : list
        List of years
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found in https://daymet.ornl.gov/overview
    pet : bool
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior, defaults to False.
    n_threads : int, optional
        Number of threads for simultaneous download, defaults to 8.

    Returns
    -------
    xarray.Dataset
        Daily climate data within a geometry
    """
    from pandas.tseries.offsets import DateOffset

    daymet = Daymet(dates, years, variables, pet)

    if years is None:
        start = pd.to_datetime(daymet.date_dict["start"]) + DateOffset(hour=12)
        end = pd.to_datetime(daymet.date_dict["end"]) + DateOffset(hour=12)
        dates_itr = utils.daymet_dates(start, end)

    else:
        start_list, end_list = [], []
        for year in daymet.date_dict["years"].split(","):
            s = pd.to_datetime(f"{year}0101")
            start_list.append(s + DateOffset(hour=12))
            if int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0):
                end_list.append(pd.to_datetime(f"{year}1230") + DateOffset(hour=12))
            else:
                end_list.append(pd.to_datetime(f"{year}1231") + DateOffset(hour=12))
        dates_itr = list(zip(start_list, end_list))

    if not isinstance(geometry, Polygon):
        raise InvalidInputType("geometry", "Shapely's Polygon")

    if fill_holes:
        geometry = Polygon(geometry.exterior)

    geometry = utils.match_crs(geometry, geo_crs, "epsg:4326")

    west, south, east, north = np.round(geometry.bounds, 6)
    base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/"
    urls = []

    for s, e in dates_itr:
        for v in daymet.variables:
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

    def getter(url):
        return xr.open_dataset(daymet.session.get(url).content)

    data = xr.merge(utils.threading(getter, urls, max_workers=n_threads))

    for k, v in daymet.units.items():
        if k in daymet.variables:
            data[k].attrs["units"] = v

    data = data.drop_vars(["lambert_conformal_conic"])
    data.attrs["crs"] = " ".join(
        ["+proj=lcc", "+lon_0=-100", "+lat_0=42.5", "+lat_1=25", "+lat_2=60", "+ellps=WGS84"]
    )

    x_res, y_res = data.x.diff("x").min().item(), data.y.diff("y").min().item()
    x_origin = data.x.values[0] - x_res / 2.0  # PixelAsArea Convention
    y_origin = data.y.values[0] - y_res / 2.0  # PixelAsArea Convention

    transform = (x_res, 0, x_origin, 0, y_res, y_origin)

    x_end = x_origin + data.dims["x"] * x_res
    y_end = y_origin + data.dims["y"] * y_res
    x_options = np.array([x_origin, x_end])
    y_options = np.array([y_origin, y_end])

    data.attrs["transform"] = transform
    data.attrs["res"] = (x_res, y_res)
    data.attrs["bounds"] = (
        x_options.min(),
        y_options.min(),
        x_options.max(),
        y_options.max(),
    )

    if pet:
        data = utils.pet_fao_gridded(data)

    mask, transform = utils.geom_mask(geometry, data.dims["x"], data.dims["y"], ds_crs=data.crs,)
    data = data.where(~xr.DataArray(mask, dims=("y", "x")), drop=True)
    return data


def ssebopeta_byloc(
    coords: Tuple[float, float],
    dates: Optional[Tuple[str, str]] = None,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Daily actual ET for a location from SSEBop database in mm/day.

    Parameters
    ----------
    coords : tuple
        Longitude and latitude of the location of interest as a tuple (lon, lat)
    dates : tuple, optional
        Start and end dates as a tuple (start, end), default to None.
    years : list or tuple, optional
        List of years, default to None.

    Returns
    -------
    pandas.DataFrame
        Daily actual ET for a location
    """
    if isinstance(coords, tuple) and len(coords) == 2:
        lon, lat = coords
    else:
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = dates
    else:
        raise InvalidInputType("dates", "tuple", "(start, end)")

    f_list = utils.get_ssebopeta_urls(start=start, end=end, years=years)
    session = RetrySession()

    with connection.onlyIPv4():

        def _ssebop(urls):
            dt, url = urls
            r = session.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))

            with rio.MemoryFile() as memfile:
                memfile.write(z.read(z.filelist[0].filename))
                with memfile.open() as src:
                    return {
                        "dt": dt,
                        "eta": [e[0] for e in src.sample([(lon, lat)])][0],
                    }

        elevations = utils.threading(_ssebop, f_list, max_workers=4)
    data = pd.DataFrame.from_records(elevations)
    data.columns = ["datetime", "eta (mm/day)"]
    data = data.set_index("datetime")
    return data * 1e-3


def ssebopeta_bygeom(
    geometry: Polygon,
    geo_crs: str = "epsg:4326",
    dates: Optional[Tuple[str, str]] = None,
    years: Optional[List[int]] = None,
    resolution: Optional[float] = None,
    fill_holes: bool = False,
) -> xr.DataArray:
    """Daily actual ET for a region from SSEBop database in mm/day at 1 km resolution.

    Notes
    -----
    Since there's still no web service available for subsetting SSEBop, the data first
    needs to be downloaded for the requested period then it is masked by the
    region of interest locally. Therefore, it's not as fast as other functions and
    the bottleneck could be the download speed.

    Parameters
    ----------
    geometry : shapely.geometry.Polygon
        The geometry for downloading clipping the data. For a tuple bbox,
        the order should be (west, south, east, north).
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    dates : tuple, optional
        Start and end dates as a tuple (start, end), default to None.
    years : list
        List of years
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior (Polygon type), defaults to False.

    Returns
    -------
    xarray.DataArray
        Daily actual ET within a geometry
    """

    if not isinstance(geometry, Polygon):
        raise InvalidInputType("geometry", "Shapely's Polygon")

    geometry = utils.match_crs(geometry, geo_crs, "epsg:4326")

    if fill_holes:
        geometry = Polygon(geometry.exterior)

    resolution = 1.0e3 / 6371000.0 * 3600.0 / np.pi * 180.0
    west, south, east, north = geometry.bounds

    width = int((east - west) * 3600 / resolution)
    height = int(abs(north - south) / abs(east - west) * width)

    mask, transform = utils.geom_mask(geometry, width, height)

    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = dates
    else:
        raise InvalidInputType("dates", "tuple", "(start, end)")

    f_list = utils.get_ssebopeta_urls(start=start, end=end, years=years)

    session = RetrySession()

    with connection.onlyIPv4():

        def _ssebop(url_stamped):
            dt, url = url_stamped
            r = session.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return (dt, z.read(z.filelist[0].filename))

        resp = utils.threading(_ssebop, f_list, max_workers=4,)

        data = utils.create_dataset(resp[0][1], mask, transform, width, height, "eta", None)
        data = data.expand_dims({"time": [resp[0][0]]})

        if len(resp) > 1:
            for dt, r in resp:
                ds = utils.create_dataset(r, mask, transform, width, height, "eta", None)
                ds = ds.expand_dims({"time": [dt]})
                data = xr.merge([data, ds])

    eta = data.eta.copy()
    eta = eta.where(eta < eta.nodatavals[0], drop=True)
    eta *= 1e-3
    eta.attrs.update({"units": "mm/day", "nodatavals": (np.nan,)})
    return eta


def nlcd(
    geometry: Polygon,
    geo_crs: str = "epsg:4326",
    years: Optional[Dict[str, int]] = None,
    width: Optional[int] = None,
    resolution: Optional[float] = None,
    file_path: Optional[Union[str, Path]] = None,
    fill_holes: bool = False,
    crs: str = "epsg:4326",
) -> xr.Dataset:
    """Get data from NLCD database (2016).

    Download land use, land cover data from NLCD (2016) database within
    a given geometry in epsg:4326.

    Notes
    -----
        NLCD data has a resolution of 1 arc-sec (~30 m).

    Parameters
    ----------
    geometry : shapely.geometry.Polygon
        The geometry for extracting the data.
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    years : dict, optional
        The years for NLCD data as a dictionary, defaults to
        {'impervious': 2016, 'cover': 2016, 'canopy': 2016}.
    width : int
        The width of the output image in pixels. The height is computed
        automatically from the geometry's bounding box aspect ratio. Either width
        or resolution should be provided.
    resolution : float
        The data resolution in arc-seconds. The width and height are computed in pixel
        based on the geometry bounds and the given resolution. Either width or
        resolution should be provided.
    file_path : dict, optional
        The path to save the downloaded images, defaults to None which will only return
        the data as ``xarray.Dataset`` and doesn't save the files. The argument should be
        a dict with keys as the variable name in the output dataframe and values as
        the path to save to the file.
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior, defaults to False.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.

    Returns
    -------
     xarray.DataArray
         NLCD within a geometry
    """
    nlcd_meta = helpers.nlcd_helper()

    names = ["impervious", "cover", "canopy"]
    avail_years = {n: nlcd_meta[f"{n}_years"] for n in names}

    if years is None:
        years = {"impervious": 2016, "cover": 2016, "canopy": 2016}
    if isinstance(years, dict):
        for service in years.keys():
            if years[service] not in avail_years[service]:
                raise InvalidInputValue(
                    f"{service.capitalize()} data for {years[service]}", avail_years[service],
                )
    else:
        raise InvalidInputType(
            "years", "dict", "{'impervious': 2016, 'cover': 2016, 'canopy': 2016}"
        )

    url = "https://www.mrlc.gov/geoserver/mrlc_download/wms"

    layers = {
        "canopy": f'NLCD_{years["canopy"]}_Tree_Canopy_L48',
        "cover": f'NLCD_{years["cover"]}_Land_Cover_Science_product_L48',
        "impervious": f'NLCD_{years["impervious"]}_Impervious_L48',
    }

    ds = services.wms_bygeom(
        url,
        layers,
        "image/geotiff",
        geometry,
        width=width,
        resolution=resolution,
        fill_holes=fill_holes,
        geo_crs=geo_crs,
        crs=crs,
    )
    ds.cover.attrs["units"] = "classes"
    ds.canopy.attrs["units"] = "%"
    ds.impervious.attrs["units"] = "%"
    return ds


class NationalMap:
    """Access to `3DEP <https://www.usgs.gov/core-science-systems/ngp/3dep>`__ service.

    The 3DEP service has multi-resolution sources so depending on the user
    provided resolution (or width) the data is resampled on server-side based
    on all the available data sources.

    Notes
    -----
    There are three functions available for getting DEM ("3DEPElevation:None" layer),
    slope ("3DEPElevation:Slope Degrees" layer), and
    aspect ("3DEPElevation:Aspect Degrees" layer). If other layers from the 3DEP
    service is desired use ``get_map`` function. The layer should be pass to this function
    as a dict where the key is the desired variable name to be used in the returned DataArray
    and the value should be a valid National Map layer e.g, {"elevation": "3DEPElevation:None"}.
    The following layers are available:
    - "3DEPElevation:Hillshade Gray"
    - "3DEPElevation:Aspect Degrees"
    - "3DEPElevation:Aspect Map"
    - "3DEPElevation:GreyHillshade_elevationFill"
    - "3DEPElevation:Hillshade Multidirectional"
    - "3DEPElevation:Slope Map"
    - "3DEPElevation:Slope Degrees"
    - "3DEPElevation:Hillshade Elevation Tinted"
    - "3DEPElevation:Height Ellipsoidal"
    - "3DEPElevation:Contour 25"
    - "3DEPElevation:Contour Smoothed 25"
    - "3DEPElevation:None" (for DEM)

    Parameters
    ----------
    geometry : shapely.geometry.Polygon
        A shapely Polygon in WGS 84 (epsg:4326).
    geo_crs : str, optional
        The spatial reference system of the input geometry, defaults to
        epsg:4326.
    width : int
        The width of the output image in pixels. The height is computed
        automatically from the geometry's bounding box aspect ratio. Either width
        or resolution should be provided.
    resolution : float
        The data resolution in arc-seconds. The width and height are computed in pixel
        based on the geometry bounds and the given resolution. Either width or
        resolution should be provided.
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior, defaults to False.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    fpath : str or Path
        Path to save the output as a ``tiff`` file, defaults to None.
    """

    def __init__(
        self,
        geometry: Polygon,
        geo_crs: str = "epsg:4326",
        width: Optional[int] = None,
        resolution: Optional[float] = None,
        fill_holes: bool = False,
        crs: str = "epsg:4326",
        fpath: Optional[Union[str, Path]] = None,
    ) -> None:
        self.url = (
            "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"
        )
        self.geometry = geometry
        self.width = width
        self.resolution = resolution
        self.fill_holes = fill_holes
        self.geo_crs = geo_crs
        self.crs = crs
        self.fpath = fpath

    def get_dem(self) -> xr.DataArray:
        """DEM as an ``xarray.DataArray`` in meters"""

        dem = self.get_map({"elevation": "3DEPElevation:None"})
        dem.attrs["units"] = "meters"
        return dem

    def get_aspect(self) -> xr.DataArray:
        """Aspect map as an ``xarray.DataArray`` in degrees"""

        aspect = self.get_map({"aspect": "3DEPElevation:Aspect Degrees"})
        aspect = aspect.where(aspect < aspect.nodatavals[0], drop=True)
        aspect.attrs["nodatavals"] = (np.nan,)
        aspect.attrs["units"] = "degrees"
        return aspect

    def get_slope(self, mpm: bool = False) -> xr.DataArray:
        """Slope from 3DEP service in degrees or meters/meters

        Parameters
        ----------
        mpm : bool, optional
            Whether to convert the slope to meters/meters from degrees, defaults to False

        Returns
        -------
        xarray.DataArray
            Slope within a geometry in degrees or meters/meters
        """

        slope = self.get_map({"slope": "3DEPElevation:Slope Degrees"})
        slope = slope.where(slope < slope.nodatavals[0], drop=True)
        slope.attrs["nodatavals"] = (np.nan,)
        if mpm:
            attrs = slope.attrs
            slope = np.tan(np.deg2rad(slope))
            slope.attrs = attrs
            slope.attrs["units"] = "meters/meters"
        else:
            slope.attrs["units"] = "degrees"
        return slope

    def get_map(self, layer: Dict[str, str]) -> Union[xr.DataArray, xr.Dataset]:
        """Get requested map using the national map's WMS service"""

        name = str(list(layer.keys())[0]).replace(" ", "_")
        _fpath: Optional[Dict[str, Optional[Union[str, Path]]]]
        _fpath = None if self.fpath is None else {name: self.fpath}
        return services.wms_bygeom(
            self.url,
            layer,
            "image/tiff",
            self.geometry,
            width=self.width,
            resolution=self.resolution,
            fill_holes=self.fill_holes,
            fpath=_fpath,
            geo_crs=self.geo_crs,
            crs=self.crs,
        )


class Station:
    """Download data from the databases.

    Download climate and streamflow observation data from Daymet and USGS,
    respectively. The data is saved to a NetCDF file. Either coords or station_id
    argument should be specified.

    Parameters
    ----------
    station_id : str, optional
        USGS station ID, defaults to None
    coords : tuple, optional
        Longitude and latitude of the point of interest, defaults to None
    dates : tuple, optional
        Limit the search for stations with daily mean discharge available within
        a date interval when coords is specified, defaults to None
    srad : float, optional
        Search radius in degrees for finding the closest station
        when coords is given, default to 0.5 degrees
    data_dir : str or Path, optional
        Path to the location of climate data, defaults to 'data'
    verbose : bool
        Whether to show messages
    """

    def __init__(
        self,
        station_id: Optional[str] = None,
        coords: Optional[Tuple[float, float]] = None,
        dates: Optional[Tuple[str, str]] = None,
        srad: float = 0.5,
        data_dir: Union[str, Path] = "data",
        verbose: bool = False,
    ) -> None:
        if dates is not None:
            start, end = dates
            self.start = pd.to_datetime(start)
            self.end = pd.to_datetime(end)

        self.verbose = verbose

        if station_id is None and coords is not None:
            self.coords = coords
            self.srad = srad
            self.get_id()
        elif coords is None and station_id is not None:
            self.station_id = str(station_id)
            self.get_coords()
        else:
            raise MissingInputs(
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "Either coordinates or station ID should be specified."
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

        self.nldi = NLDI()
        self.get_watershed()

        info = nwis_siteinfo(ids=self.station_id, expanded=True)
        try:
            self.areasqkm = info["contrib_drain_area_va"].astype("float64").to_numpy()[0] * 2.5899
        except ValueError:
            try:
                self.areasqkm = info["drain_area_va"].astype("float64").to_numpy()[0] * 2.5899
            except ValueError:
                self.areasqkm = self.flowlines().areasqkm.sum()

        self.hcdn = info.hcdn_2009.to_numpy()[0]

        if self.verbose:
            print(self.__repr__())

    def __repr__(self) -> str:
        """Print the characteristics of the watershed."""
        return (
            f"[ID: {self.station_id}] ".ljust(MARGINE)
            + f"Watershed: {self.name}\n"
            + "".ljust(MARGINE)
            + f"Coordinates: ({self.lon:.3f}, {self.lat:.3f})\n"
            + "".ljust(MARGINE)
            + f"Altitude: {self.altitude:.0f} m above {self.datum}\n"
            + "".ljust(MARGINE)
            + f"Drainage area: {self.areasqkm:.0f} sqkm\n"
            + "".ljust(MARGINE)
            + f"HCDN 2009: {self.hcdn}\n"
            + "".ljust(MARGINE)
            + "Data availlability:"
            + f"{np.datetime_as_string(self.st_begin, 'D')} to "
            + f"{np.datetime_as_string(self.st_end, 'D')}."
        )

    def get_coords(self) -> None:
        """Get coordinates of the station from station ID."""
        siteinfo = nwis_siteinfo(ids=self.station_id)
        st = siteinfo[siteinfo.stat_cd == "00003"]
        self.st_begin = st.begin_date.to_numpy()[0]
        self.st_end = st.end_date.to_numpy()[0]

        self.coords = (
            st["dec_long_va"].astype("float64").to_numpy()[0],
            st["dec_lat_va"].astype("float64").to_numpy()[0],
        )
        self.altitude = st["alt_va"].astype("float64").to_numpy()[0] * 0.3048  # convert ft to meter
        self.datum = st["alt_datum_cd"].to_numpy()[0]
        self.name = st.station_nm.to_numpy()[0]

    def get_id(self) -> None:
        """Get station ID based on the specified coordinates."""
        import shapely.geometry as geom

        bbox = (
            self.coords[0] - self.srad,
            self.coords[1] - self.srad,
            self.coords[0] + self.srad,
            self.coords[1] + self.srad,
        )

        sites = nwis_siteinfo(bbox=bbox)
        sites = sites[sites.stat_cd == "00003"]
        if len(sites) < 1:
            raise ZeroMatched(
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "No USGS station were found within a "
                + f"{int(self.srad * 111 / 10) * 10}-km radius "
                + f"of ({self.coords[0]}, {self.coords[1]}) with daily mean streamflow."
            )

        point = geom.Point(self.coords)
        pts = {
            sid: geom.Point(lon, lat)
            for sid, lon, lat in sites[["site_no", "dec_long_va", "dec_lat_va"]].itertuples(
                name=None, index=False
            )
        }

        stations = gpd.GeoSeries(pts)
        distance = stations.apply(lambda x: x.distance(point)).sort_values()

        if distance.shape[0] == 0:
            station_id = None
        elif self.start is None and self.end is None:
            station = sites[sites.site_no == distance.index[0]]
            self.st_begin = station.begin_date.to_numpy()[0]
            self.st_end = station.end_date.to_numpy()[0]
            station_id = station.site_no.to_numpy()[0]
        else:
            station_id = None
            for sid in distance.index:
                station = sites[sites.site_no == sid]
                self.st_begin = station.begin_date.to_numpy()[0]
                self.st_end = station.end_date.to_numpy()[0]

                if self.start < self.st_begin or self.end > self.st_end:
                    continue
                else:
                    station_id = sid
                    break

        if station_id is None:
            raise ZeroMatched(
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "No USGS station were found within a "
                + f"{int(self.srad * 111 / 10) * 10}-km radius.\n"
                + "Use ``utils.interactive_map(bbox)`` function to explore "
                + "the available stations within a bounding box."
            )

        self.station_id = station_id
        self.coords = (
            station.dec_long_va.astype("float64").to_numpy()[0],
            station.dec_lat_va.astype("float64").to_numpy()[0],
        )
        # convert ft to meter
        self.altitude = station["alt_va"].astype("float64").to_numpy()[0] * 0.3048
        self.datum = station["alt_datum_cd"].to_numpy()[0]
        self.name = station.station_nm.to_numpy()[0]

    def get_watershed(self) -> None:
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
            self.basin = self.nldi.getfeature_byid(
                "nwissite", f"USGS-{self.station_id}", basin=True
            )
            self.basin.to_file(geom_file)

            if self.verbose:
                print(
                    f"[ID: {self.station_id}] ".ljust(MARGINE)
                    + f"The watershed geometry saved to {geom_file}."
                )

        self.geometry = self.basin.geometry.to_numpy()[0]

    def comids(
        self, navigation: str = "upstreamTributaries", distance: Optional[int] = None
    ) -> list:
        """Find ComIDs of the flowlines within the watershed.

        Parameters
        ----------
        navigation : str, optional
            The direction for navigating the NHDPlus database. The valid options are:
            ``upstreamMain``, ``upstreamTributaries``, ``downstreamMain``,
            ``downstreamDiversions``. Defaults to ``upstreamTributaries``.
        distance : int, optional
            The distance to limit the navigation in km, defaults to None.

        Returns
        -------
        list
            ComID(s) within a geometry
        """
        comids = self.nldi.navigate_byid(
            "nwissite", f"USGS-{self.station_id}", navigation=navigation, distance=distance,
        )
        return comids.nhdplus_comid.tolist()

    def nwis_stations(
        self, navigation: str = "upstreamTributaries", distance: Optional[int] = None
    ) -> gpd.GeoDataFrame:
        """Get USGS stations within the watershed.

        Parameters
        ----------
        navigation : str, optional
            The direction for navigating the NHDPlus database. The valid options are:
            ``upstreamMain``, ``upstreamTributaries``, ``downstreamMain``,
            ``downstreamDiversions``. Defaults to ``upstreamTributaries``.
        distance : float, optional
            The distance to limit the navigation in km. Defaults to None (all stations).

        Returns
        -------
        geopandas.GeoDataFrame
            USGS stations within a geometry
        """
        return self.nldi.navigate_byid(
            "nwissite",
            f"USGS-{self.station_id}",
            navigation=navigation,
            source="nwissite",
            distance=distance,
        )

    def pour_points(
        self, navigation: str = "upstreamTributaries", distance: Optional[int] = None
    ) -> gpd.GeoDataFrame:
        """Get HUC12 pour point within the watershed.

        Parameters
        ----------
        navigation : str, optional
            The direction for navigating the NHDPlus database. The valid options are:
            ``upstreamMain``, ``upstreamTributaries``, ``downstreamMain``,
            ``downstreamDiversions``. Defaults to ``upstreamTributaries``.
        distance : float, optional
            The distance to limit the navigation in km. Defaults to None (all stations).

        Returns
        -------
        geopandas.GeoDataFrame
            Pour points within a geometry
        """
        return self.nldi.navigate_byid(
            "nwissite",
            f"USGS-{self.station_id}",
            navigation=navigation,
            source="huc12pp",
            distance=distance,
        )

    def catchments(
        self, navigation: str = "upstreamTributaries", distance: Optional[int] = None
    ) -> gpd.GeoDataFrame:
        """Get chatchments for the watershed from NHDPlus V2.

        Parameters
        ----------
        navigation : str, optional
            The direction for navigating the NHDPlus database. The valid options are:
            ``upstreamMain``, ``upstreamTributaries``, ``downstreamMain``,
            ``downstreamDiversions``. Defaults to ``upstreamTributaries``.
        distance : float, optional
            The distance to limit the navigation in km. Defaults to None (all stations).

        Returns
        -------
        geopandas.GeoDataFrame
            Catchments within a geometry
        """
        wd = WaterData("catchmentsp")
        return wd.getfeature_byid("featureid", self.comids(navigation, distance))

    def flowlines(
        self, navigation: str = "upstreamTributaries", distance: Optional[int] = None
    ) -> gpd.GeoDataFrame:
        """Get flowlines for the watershed from NHDPlus V2.

        Parameters
        ----------
        navigation : str, optional
            The direction for navigating the NHDPlus database. The valid options are:
            ``upstreamMain``, ``upstreamTributaries``, ``downstreamMain``,
            ``downstreamDiversions``. Defaults to ``upstreamTributaries``.
        distance : float, optional
            The distance to limit the navigation in km, defaults to None.

        Returns
        -------
        geopandas.GeoDataFrame
            Flowlines within a geometry
        """
        wd = WaterData("nhdflowline_network")
        return wd.getfeature_byid("comid", self.comids(navigation, distance))


def interactive_map(bbox: Tuple[float, float, float, float]) -> folium.Map:
    """An interactive map including all USGS stations within a bounding box.

    Only stations that record(ed) daily streamflow data are included.

    Parameters
    ----------
    bbox : tuple
        List of corners in this order (west, south, east, north)

    Returns
    -------
    folium.Map
        Interactive map within a bounding box.
    """

    if isinstance(bbox, tuple):
        if len(bbox) != 4:
            raise InvalidInputType("bbox", "tuple", "(west, south, east, north)")
    else:
        raise InvalidInputType("bbox", "tuple", "(west, south, east, north)")

    sites = nwis_siteinfo(bbox=bbox)
    sites["coords"] = [
        (lat, lon)
        for lat, lon in sites[["dec_lat_va", "dec_long_va"]].itertuples(name=None, index=False)
    ]
    sites["altitude"] = (
        sites["alt_va"].astype(str) + " ft above " + sites["alt_datum_cd"].astype(str)
    )

    drain_area = nwis_siteinfo(bbox=bbox, expanded=True)[
        ["site_no", "drain_area_va", "contrib_drain_area_va"]
    ]
    sites = sites.merge(drain_area, on="site_no").dropna()

    sites["drain_area_va"] = sites["drain_area_va"].astype(str) + " square miles"
    sites["contrib_drain_area_va"] = sites["contrib_drain_area_va"].astype(str) + " square miles"

    cols_old = [
        "site_no",
        "station_nm",
        "coords",
        "altitude",
        "huc_cd",
        "drain_area_va",
        "contrib_drain_area_va",
        "begin_date",
        "end_date",
        "hcdn_2009",
    ]

    cols_new = [
        "Site No.",
        "Station Name",
        "Coordinate",
        "Altitude",
        "HUC8",
        "Drainage Area",
        "Contributing Drainga Area",
        "Begin date",
        "End data",
        "HCDN 2009",
    ]
    sites = sites.rename(columns=dict(zip(cols_old, cols_new)))[cols_new]

    msgs = []
    for row in sites.itertuples(index=False):
        msg = ""
        for col in sites:
            msg += "".join(
                ["<strong>", col, "</strong> : ", f"{row[sites.columns.get_loc(col)]}<br>"]
            )
        msgs.append(msg[:-4])

    sites["msg"] = msgs

    west, south, east, north = bbox
    lon = (west + east) * 0.5
    lat = (south + north) * 0.5

    imap = folium.Map(location=(lat, lon), tiles="Stamen Terrain", zoom_start=12)

    for coords, msg in sites[["Coordinate", "msg"]].itertuples(name=None, index=False):
        folium.Marker(
            location=coords, popup=folium.Popup(msg, max_width=250), icon=folium.Icon()
        ).add_to(imap)

    return imap
