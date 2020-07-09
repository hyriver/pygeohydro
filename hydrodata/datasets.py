"""Accessing data from the supported databases through their APIs."""
import io
import zipfile
from dataclasses import dataclass  # for python 3.6 compatibility
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from rasterio import features as rio_features
from rasterio import warp as rio_warp
from shapely.geometry import Polygon

from hydrodata import helpers, services, utils
from hydrodata.connection import RetrySession
from hydrodata.exceptions import (
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingInputs,
    ZeroMatched,
)
from hydrodata.services import WFS, ServiceURL

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

    if not isinstance(station_ids, (str, list)):
        raise InvalidInputType("ids", "str or list")

    station_ids = station_ids if isinstance(station_ids, list) else [station_ids]

    if not isinstance(dates, tuple) or len(dates) != 2:
        raise InvalidInputType("dates", "tuple", "(start, end)")

    start = pd.to_datetime(dates[0])
    end = pd.to_datetime(dates[1])

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
            + ", ".join(nas)
        )

    payload = {
        "format": "json",
        "sites": ",".join(station_ids),
        "startDT": start.strftime("%Y-%m-%d"),
        "endDT": end.strftime("%Y-%m-%d"),
        "parameterCd": "00060",
        "statCd": "00003",
        "siteStatus": "all",
    }

    r = RetrySession().post(f"{ServiceURL().restful.nwis}/dv", payload)

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

    query = utils.generate_nwis_query(ids, bbox)
    outputType = {"siteOutput": "expanded"} if expanded else {"outputDataTypeCd": "dv"}

    payload = {
        **query,
        **outputType,
        "format": "rdb",
        "parameterCd": "00060",
        "siteStatus": "all",
        "hasDataTypeCd": "dv",
    }

    r = RetrySession().post(f"{ServiceURL().restful.nwis}/site", payload).text.split("\n")

    r_list = [txt.split("\t") for txt in r if "#" not in txt]
    r_dict = [dict(zip(r_list[0], st)) for st in r_list[2:]]

    sites = pd.DataFrame.from_dict(r_dict).dropna()
    sites = sites.drop(sites[sites.alt_va == ""].index)

    try:
        sites = sites[sites.parm_cd == "00060"]
        sites["begin_date"] = pd.to_datetime(sites["begin_date"])
        sites["end_date"] = pd.to_datetime(sites["end_date"])
    except AttributeError:
        pass

    float_cols = ["dec_lat_va", "dec_long_va", "alt_va"]
    sites[float_cols] = sites[float_cols].astype("float64")

    sites = sites[sites.site_no.apply(len) == 8]

    gii = WaterData("gagesii", "epsg:900913")
    hcdn = gii.getfeature_byid("staid", sites.site_no.tolist())
    hcdn_dict = hcdn[["staid", "hcdn_2009"]].set_index("staid").hcdn_2009.to_dict()
    sites["hcdn_2009"] = sites.site_no.apply(
        lambda x: len(hcdn_dict[x]) > 0 if x in hcdn_dict.keys() else False
    )

    return sites


class WaterData:
    """Access to `Water Data <https://labs.waterdata.usgs.gov/geoserver>`__ service.

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
            ServiceURL().wfs.waterdata,
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

        self.base_url = ServiceURL().restful.nldi
        self.session = RetrySession()
        r = self.session.get(self.base_url).json()
        self.valid_sources = [el for sub in utils.traverse_json(r, ["source"]) for el in sub]

    def getfeature_byid(
        self, fsource: str, fid: str, basin: bool = False, url_only: bool = False
    ) -> gpd.GeoDataFrame:
        """Get features of a single id.

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
        """Navigate the NHDPlus databse from a single feature id.

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
    """Base class for Daymet requests.

    Parameters
    ----------
    variables : str or list or tuple, optional
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
        Defaults to None i.e., all the variables are downloaded.
    pet : bool, optional
        Whether to compute evapotranspiration based on
        `UN-FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`__.
        The default is False
    """

    def __init__(
        self, variables: Optional[Union[List[str], str]] = None, pet: bool = False,
    ) -> None:

        self.session = RetrySession()

        vars_table = helpers.daymet_variables()

        self.units = dict(zip(vars_table["Abbr"], vars_table["Units"]))

        valid_variables = vars_table.Abbr.to_list()
        if variables is None:
            self.variables = valid_variables
        else:
            self.variables = variables if isinstance(variables, list) else [variables]

            if not set(self.variables).issubset(set(valid_variables)):
                raise InvalidInputValue("variables", valid_variables)

            if pet:
                reqs = ("tmin", "tmax", "vp", "srad", "dayl")
                self.variables = list(set(reqs) | set(variables))

    @staticmethod
    def dates_todict(dates: Tuple[str, str]) -> Dict[str, str]:
        """Set dates by start and end dates as a tuple, (start, end)"""

        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InvalidInputType("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])

        if start < pd.to_datetime("1980-01-01"):
            raise InvalidInputRange("Daymet database ranges from 1980 to 2019.")

        return {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        }

    @staticmethod
    def years_todict(years: Union[List[int], int]) -> Dict[str, str]:
        """Set date by list of year(s)"""

        years = years if isinstance(years, list) else [years]
        return {"years": ",".join(str(y) for y in years)}

    def dates_tolist(
        self, dates: Tuple[str, str]
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Correct dates for Daymet accounting for leap years.

        Daymet doesn't account for leap years and removes Dec 31 when
        it's leap year. This function returns all the dates in the
        Daymet database within the provided date range.
        """

        date_dict = self.dates_todict(dates)
        start = pd.to_datetime(date_dict["start"]) + pd.DateOffset(hour=12)
        end = pd.to_datetime(date_dict["end"]) + pd.DateOffset(hour=12)

        period = pd.date_range(start, end)
        nl = period[~period.is_leap_year]
        lp = period[(period.is_leap_year) & (~period.strftime("%Y-%m-%d").str.endswith("12-31"))]
        _period = period[(period.isin(nl)) | (period.isin(lp))]
        years = [_period[_period.year == y] for y in _period.year.unique()]
        return [(y[0], y[-1]) for y in years]

    def years_tolist(
        self, years: Union[List[int], int]
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Correct dates for Daymet accounting for leap years.

        Daymet doesn't account for leap years and removes Dec 31 when
        it's leap year. This function returns all the dates in the
        Daymet database for the provided years.
        """

        date_dict = self.years_todict(years)
        start_list, end_list = [], []
        for year in date_dict["years"].split(","):
            s = pd.to_datetime(f"{year}0101")
            start_list.append(s + pd.DateOffset(hour=12))
            if int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0):
                end_list.append(pd.to_datetime(f"{year}1230") + pd.DateOffset(hour=12))
            else:
                end_list.append(pd.to_datetime(f"{year}1231") + pd.DateOffset(hour=12))
        return list(zip(start_list, end_list))


def daymet_byloc(
    coords: Tuple[float, float],
    dates: Optional[Tuple[str, str]] = None,
    years: Optional[Union[List[int], int]] = None,
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
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
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

    daymet = Daymet(variables, pet)

    if (years is None and dates is None) or (years is not None and dates is not None):
        raise MissingInputs("Either years or dates arguments should be provided.")

    if dates is not None:
        date_dict = daymet.dates_todict(dates)
    else:
        date_dict = daymet.years_todict(years)  # type: ignore

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

    payload = {
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "vars": ",".join(daymet.variables),
        "format": "json",
        **date_dict,
    }

    r = daymet.session.get(ServiceURL().restful.daymet_point, payload)

    clm = pd.DataFrame(r.json()["data"])
    clm.index = pd.to_datetime(clm.year * 1000.0 + clm.yday, format="%Y%j")
    clm = clm.drop(["year", "yday"], axis=1)

    if pet:
        clm = utils.pet_fao_byloc(clm, coords)
    return clm


def daymet_bygeom(
    geometry: Union[Polygon, Tuple[float, float, float, float]],
    geo_crs: str = "epsg:4326",
    dates: Optional[Tuple[str, str]] = None,
    years: Optional[List[int]] = None,
    variables: Optional[List[str]] = None,
    pet: bool = False,
    fill_holes: bool = False,
    n_threads: int = 8,
) -> xr.Dataset:
    """Gridded data from the Daymet database at 1-km resolution.

    The data is clipped using NetCDF Subset Service.

    Parameters
    ----------
    geometry : shapely.geometry.Polygon or bbox
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
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
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

    daymet = Daymet(variables, pet)

    if (years is None and dates is None) or (years is not None and dates is not None):
        raise MissingInputs("Either years or dates arguments should be provided.")

    if dates is not None:
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(years)  # type: ignore

    if isinstance(geometry, Polygon):
        if fill_holes:
            geometry = Polygon(geometry.exterior)
        bounds = utils.match_crs(geometry.bounds, geo_crs, "epsg:4326")
    else:
        bounds = utils.match_crs(geometry, geo_crs, "epsg:4326")

    west, south, east, north = np.round(bounds, 6)
    base_url = ServiceURL().restful.daymet_grid
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
    crs = " ".join(
        ["+proj=lcc", "+lon_0=-100", "+lat_0=42.5", "+lat_1=25", "+lat_2=60", "+ellps=WGS84"]
    )
    data.attrs["crs"] = crs

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

    _geometry = utils.match_crs(geometry, geo_crs, crs)
    if isinstance(_geometry, Polygon):
        bounds = _geometry.bounds
    else:
        bounds = _geometry

    transform = rio_warp.calculate_default_transform(
        crs,
        crs,
        width=data.dims["x"],
        height=data.dims["y"],
        left=bounds[0],
        bottom=bounds[1],
        right=bounds[2],
        top=bounds[3],
    )[0]
    mask = rio_features.geometry_mask([_geometry], (data.dims["y"], data.dims["x"]), transform)
    data = data.where(xr.DataArray(~mask, dims=("y", "x")), drop=True)
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

    f_list = _get_ssebopeta_urls(start=start, end=end, years=years)
    session = RetrySession()

    with session.onlyIPv4():

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
    fill_holes: bool = False,
) -> xr.DataArray:
    """Daily actual ET for a region from SSEBop database in mm/day at 1 km
    resolution resolution.

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
        List of yearsS
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior (Polygon type), defaults to False.

    Returns
    -------
    xarray.DataArray
        Daily actual ET within a geometry in mm/day at 1 km resolution
    """

    geometry = utils.match_crs(geometry, geo_crs, "epsg:4326")

    if fill_holes:
        geometry = Polygon(geometry.exterior)

    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = dates
    else:
        raise InvalidInputType("dates", "tuple", "(start, end)")

    f_list = _get_ssebopeta_urls(start=start, end=end, years=years)

    session = RetrySession()

    with session.onlyIPv4():

        def _ssebop(url_stamped):
            dt, url = url_stamped
            r = session.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return (dt, z.read(z.filelist[0].filename))

        resp = utils.threading(_ssebop, f_list, max_workers=4,)

        data = utils.create_dataset(resp[0][1], geometry, "eta", None)
        data = data.expand_dims({"time": [resp[0][0]]})

        if len(resp) > 1:
            for dt, r in resp:
                ds = utils.create_dataset(r, geometry, "eta", None)
                ds = ds.expand_dims({"time": [dt]})
                data = xr.merge([data, ds])

    eta = data.eta.copy()
    eta = eta.where(eta < eta.nodatavals[0], drop=True)
    eta *= 1e-3
    eta.attrs.update({"units": "mm/day", "nodatavals": (np.nan,)})
    return eta


def _get_ssebopeta_urls(
    start: Optional[Union[pd.DatetimeIndex, str]] = None,
    end: Optional[Union[pd.DatetimeIndex, str]] = None,
    years: Optional[Union[int, List[int]]] = None,
) -> List[Tuple[pd.DatetimeIndex, str]]:
    """Get list of URLs for SSEBop dataset within a period."""
    if years is None and start is not None and end is not None:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if start < pd.to_datetime("2000-01-01") or end > pd.to_datetime("2018-12-31"):
            raise InvalidInputRange("SSEBop database ranges from 2000 to 2018.")
        dates = pd.date_range(start, end)
    elif years is not None and start is None and end is None:
        years = years if isinstance(years, list) else [years]
        seebop_yrs = np.arange(2000, 2019)

        if any(y not in seebop_yrs for y in years):
            raise InvalidInputRange("SSEBop database ranges from 2000 to 2018.")

        d_list = [pd.date_range(f"{y}0101", f"{y}1231") for y in years]
        dates = d_list[0] if len(d_list) == 1 else d_list[0].union_many(d_list[1:])
    else:
        raise MissingInputs("Either years or start and end arguments should be provided.")

    base_url = ServiceURL().http.ssebopeta
    f_list = [(d, f"{base_url}/det{d.strftime('%Y%j')}.modisSSEBopETactual.zip") for d in dates]

    return f_list


def nlcd(
    geometry: Polygon,
    resolution: float,
    years: Optional[Dict[str, int]] = None,
    geo_crs: str = "epsg:4326",
    crs: str = "epsg:4326",
    fpath: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
    fill_holes: bool = False,
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
    resolution : float
        The data resolution in meters. The width and height of the output are computed in pixel
        based on the geometry bounds and the given resolution.
    years : dict, optional
        The years for NLCD data as a dictionary, defaults to
        {'impervious': 2016, 'cover': 2016, 'canopy': 2016}.
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    fpath : dict, optional
        The path to save the downloaded images, defaults to None which will only return
        the data as ``xarray.Dataset`` and doesn't save the files. The argument should be
        a dict with keys as the variable name in the output dataframe and values as
        the path to save to the file.
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior, defaults to False.

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

    layers = {
        "canopy": f'NLCD_{years["canopy"]}_Tree_Canopy_L48',
        "cover": f'NLCD_{years["cover"]}_Land_Cover_Science_product_L48',
        "impervious": f'NLCD_{years["impervious"]}_Impervious_L48',
    }

    ds = services.wms_bygeom(
        ServiceURL().wms.mrlc,
        layers,
        "image/geotiff",
        geometry,
        resolution,
        geo_crs=geo_crs,
        crs=crs,
        fill_holes=fill_holes,
        fpath=fpath,
    )
    ds.cover.attrs["units"] = "classes"
    ds.canopy.attrs["units"] = "%"
    ds.impervious.attrs["units"] = "%"
    return ds


@dataclass
class NationalMap:
    """Access to `3DEP <https://www.usgs.gov/core-science-systems/ngp/3dep>`__
    service.

    The 3DEP service has multi-resolution sources so depending on the user
    provided resolution the data is resampled on server-side based
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
    resolution : float
        The data resolution in meters. The width and height of the output are computed in pixel
        based on the geometry bounds and the given resolution.
    geo_crs : str, optional
        The spatial reference system of the input geometry, defaults to
        epsg:4326.
    fill_holes : bool, optional
        Whether to fill the holes in the geometry's interior, defaults to False.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    fpath : str or Path
        Path to save the output as a ``tiff`` file, defaults to None.
    """

    geometry: Polygon
    resolution: float
    geo_crs: str = "epsg:4326"
    fill_holes: bool = False
    crs: str = "epsg:4326"
    fpath: Optional[Union[str, Path]] = None

    def get_dem(self) -> xr.DataArray:
        """DEM as an ``xarray.DataArray`` in meters."""

        dem = self.get_map({"elevation": "3DEPElevation:None"})
        dem.attrs["units"] = "meters"
        return dem

    def get_aspect(self) -> xr.DataArray:
        """Aspect map as an ``xarray.DataArray`` in degrees."""

        aspect = self.get_map({"aspect": "3DEPElevation:Aspect Degrees"})
        aspect = aspect.where(aspect < aspect.nodatavals[0], drop=True)
        aspect.attrs["nodatavals"] = (np.nan,)
        aspect.attrs["units"] = "degrees"
        return aspect

    def get_slope(self, mpm: bool = False) -> xr.DataArray:
        """Slope from 3DEP service in degrees or meters/meters.

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
        """Get requested map using the national map's WMS service."""

        name = str(list(layer.keys())[0]).replace(" ", "_")
        _fpath: Optional[Dict[str, Optional[Union[str, Path]]]]
        _fpath = None if self.fpath is None else {name: self.fpath}
        return services.wms_bygeom(
            ServiceURL().wms.nm_3dep,
            layer,
            "image/tiff",
            self.geometry,
            self.resolution,
            geo_crs=self.geo_crs,
            crs=self.crs,
            fill_holes=self.fill_holes,
            fpath=_fpath,
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

        self.dates = dates
        if dates is not None:
            self.start = pd.to_datetime(dates[0])
            self.end = pd.to_datetime(dates[1])

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
        utils.check_dir(self.data_dir.joinpath("dummy"))

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
        from shapely import geometry as geom

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
            raise ZeroMatched(
                f"[ID: {self.coords}] ".ljust(MARGINE)
                + "No USGS station were found within a "
                + f"{int(self.srad * 111 / 10) * 10}-km radius.\n"
                + "Use ``utils.interactive_map(bbox)`` function to explore "
                + "the available stations within a bounding box."
            )

        station = sites[sites.site_no == distance.index[0]]
        self.st_begin = station.begin_date.to_numpy()[0]
        self.st_end = station.end_date.to_numpy()[0]
        self.station_id = station.site_no.to_numpy()[0]

        if self.dates is not None:
            if self.start < self.st_begin or self.end > self.st_end:
                for sid in distance.index[1:]:
                    self.st_begin = station.begin_date.to_numpy()[0]
                    self.st_end = station.end_date.to_numpy()[0]

                    if self.start < self.st_begin or self.end > self.st_end:
                        continue
                    self.station_id = sid
                    break

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
