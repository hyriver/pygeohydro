"""Accessing data from the supported databases through their APIs."""
import io
import logging
import re
import sys
import zipfile
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import async_retriever as ar
import cytoolz as tlz
import folium
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
import rasterio as rio
import xarray as xr
from pygeoogc import WMS, RetrySession, ServiceURL
from pynhd import NLDI, AGRBase, WaterData
from shapely.geometry import MultiPolygon, Polygon

from . import helpers
from .exceptions import (
    DataNotAvailable,
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    ZeroMatched,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False

DEF_CRS = "epsg:4326"


class NID(AGRBase):
    """Retrieve data from the National Inventory of Dams web service."""

    def __init__(self):
        super().__init__("nid2019_u", "*", DEF_CRS)
        self.service = ServiceURL().restful.nid_2019
        rjson = ar.retrieve(
            [
                "/".join(
                    [
                        "https://gist.githubusercontent.com/cheginit",
                        "91af7f7427763057a18000c5309280dc/raw",
                        "d1b138e03e4ab98ba0e34c3226da8cb62a0e4703/nid_column_attrs.json",
                    ]
                )
            ],
            "json",
        )
        self.attrs = pd.DataFrame(rjson[0])


def ssebopeta_byloc(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
) -> pd.DataFrame:
    """Daily actual ET for a location from SSEBop database in mm/day.

    Parameters
    ----------
    coords : tuple
        Longitude and latitude of the location of interest as a tuple (lon, lat)
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].

    Returns
    -------
    pandas.DataFrame
        Daily actual ET for a location
    """
    if not (isinstance(coords, tuple) and len(coords) == 2):
        raise InvalidInputType("coords", "tuple", "(lon, lat)")

    lon, lat = coords

    f_list = _get_ssebopeta_urls(dates)
    session = RetrySession()

    with session.onlyipv4():

        def _ssebop(urls: Tuple[str, str]) -> Dict[str, Union[str, List[float]]]:
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

        eta = pd.DataFrame.from_records(_ssebop(f) for f in f_list)
    eta.columns = ["datetime", "eta (mm/day)"]
    eta = eta.sort_values("datetime").set_index("datetime") * 1e-3
    return eta


def ssebopeta_bygeom(
    geometry: Union[Polygon, Tuple[float, float, float, float]],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    geo_crs: str = DEF_CRS,
) -> xr.DataArray:
    """Get daily actual ET for a region from SSEBop database.

    Notes
    -----
    Since there's still no web service available for subsetting SSEBop, the data first
    needs to be downloaded for the requested period then it is masked by the
    region of interest locally. Therefore, it's not as fast as other functions and
    the bottleneck could be the download speed.

    Parameters
    ----------
    geometry : shapely.geometry.Polygon or tuple
        The geometry for downloading clipping the data. For a tuple bbox,
        the order should be (west, south, east, north).
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.

    Returns
    -------
    xarray.DataArray
        Daily actual ET within a geometry in mm/day at 1 km resolution
    """
    _geometry = geoutils.geo2polygon(geometry, geo_crs, DEF_CRS)

    f_list = _get_ssebopeta_urls(dates)

    session = RetrySession()

    with session.onlyipv4():

        def _ssebop(url_stamped: Tuple[str, str]) -> xr.DataArray:
            dt, url = url_stamped
            resp = session.get(url)
            zfile = zipfile.ZipFile(io.BytesIO(resp.content))
            content = zfile.read(zfile.filelist[0].filename)
            ds = geoutils.gtiff2xarray({"eta": content}, _geometry, DEF_CRS)
            return ds.expand_dims({"time": [dt]})

        data = xr.merge(_ssebop(f) for f in f_list).sortby("time")

    eta = data.eta.copy()
    eta *= 1e-3
    eta.attrs.update({"units": "mm/day", "nodatavals": (np.nan,)})
    return eta


def _get_ssebopeta_urls(
    dates: Union[Tuple[str, str], Union[int, List[int]]]
) -> List[Tuple[pd.DatetimeIndex, str]]:
    """Get list of URLs for SSEBop dataset within a period or years."""
    if not isinstance(dates, (tuple, list, int)):
        raise InvalidInputType(
            "dates", "tuple, list, or int", "(start, end), year, or [years, ...]"
        )

    if isinstance(dates, tuple):
        if len(dates) != 2:
            raise InvalidInputType("dates", "(start, end)")
        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])
        if start < pd.to_datetime("2000-01-01") or end > pd.to_datetime("2020-12-31"):
            raise InvalidInputRange("SSEBop", ("2000", "2020"))
        date_range = pd.date_range(start, end)
    else:
        years = dates if isinstance(dates, list) else [dates]
        seebop_yrs = np.arange(2000, 2020)

        if any(y not in seebop_yrs for y in years):
            raise InvalidInputRange("SSEBop", ("2000", "2020"))

        d_list = [pd.date_range(f"{y}0101", f"{y}1231") for y in years]
        date_range = d_list[0] if len(d_list) == 1 else d_list[0].union_many(d_list[1:])

    base_url = ServiceURL().http.ssebopeta
    f_list = [
        (d, f"{base_url}/det{d.strftime('%Y%j')}.modisSSEBopETactual.zip") for d in date_range
    ]

    return f_list


def nlcd(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    resolution: float,
    years: Optional[Mapping[str, Union[int, List[int]]]] = None,
    region: str = "L48",
    geo_crs: str = DEF_CRS,
    crs: str = DEF_CRS,
) -> xr.Dataset:
    """Get data from NLCD database (2016).

    Download land use/land cover data from NLCD (2016) database within
    a given geometry in epsg:4326.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or tuple of length 4
        The geometry or bounding box (west, south, east, north) for extracting the data.
    resolution : float
        The data resolution in meters. The width and height of the output are computed in pixel
        based on the geometry bounds and the given resolution.
    years : dict, optional
        The years for NLCD layers as a dictionary, defaults to
        ``{'impervious': [2019], 'cover': [2019], 'canopy': [2019], "descriptor": [2019]}``.
        Layers that are not in years are ignored, e.g., ``{'cover': [2016, 2019]}`` returns
        land cover data for 2016 and 2019.
    region : str, optional
        Region in the US, defaults to ``L48``. Valid values are L48 (for CONUS), HI (for Hawaii),
        AK (for Alaska), and PR (for Puerto Rico). Both lower and upper cases are acceptable.
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.

    Returns
    -------
     xarray.DataArray
         NLCD within a geometry
    """
    if resolution < 30:
        logger.warning("NLCD resolution is 30 m, so finer resolutions are not recommended.")

    default_years = {"impervious": [2019], "cover": [2019], "canopy": [2016], "descriptor": [2019]}
    years = default_years if years is None else years

    if not isinstance(years, dict):
        raise InvalidInputType("years", "dict", f"{default_years}")

    _years = tlz.valmap(lambda x: x if isinstance(x, list) else [x], years)

    layers = _nlcd_layers(_years, region)

    _geometry = geoutils.geo2polygon(geometry, geo_crs, crs)
    wms = WMS(ServiceURL().wms.mrlc, layers=layers, outformat="image/geotiff", crs=crs)
    r_dict = wms.getmap_bybox(
        _geometry.bounds, resolution, box_crs=crs, kwargs={"styles": "raster"}
    )

    ds = geoutils.gtiff2xarray(r_dict, _geometry, crs)

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    units = OrderedDict(
        (("impervious", "%"), ("cover", "classes"), ("canopy", "%"), ("descriptor", "classes"))
    )
    for lyr in layers:
        name = [n for n in units if n in lyr.lower()][-1]
        lyr_name = f"{name}_{lyr.split('_')[1]}"
        ds = ds.rename({lyr: lyr_name})
        ds[lyr_name].attrs["units"] = units[name]
    return ds


def _nlcd_layers(years: Mapping[str, List[int]], region: str) -> List[str]:
    """Get NLCD layers for the provided years dictionary."""
    valid_regions = ["L48", "HI", "PR", "AK"]
    region = region.upper()
    if region not in valid_regions:
        raise InvalidInputValue("region", valid_regions)

    nlcd_meta = helpers.nlcd_helper()

    names = ["impervious", "cover", "canopy", "descriptor"]
    avail_years = {n: nlcd_meta[f"{n}_years"] for n in names}

    if any(
        yr not in avail_years[lyr] or lyr not in names for lyr, yrs in years.items() for yr in yrs
    ):
        vals = [f"\n{lyr}: {', '.join(str(y) for y in yr)}" for lyr, yr in avail_years.items()]
        raise InvalidInputValue("years", vals)

    layer_map = {
        "canopy": lambda _: "Tree_Canopy",
        "cover": lambda _: "Land_Cover_Science_Product",
        "impervious": lambda _: "Impervious",
        "descriptor": lambda x: "Impervious_Descriptor" if x == "AK" else "Impervious_descriptor",
    }

    return [
        f"NLCD_{yr}_{layer_map[lyr](region)}_{region}" for lyr, yrs in years.items() for yr in yrs
    ]


def cover_statistics(ds: xr.Dataset) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
    """Percentages of the categorical NLCD cover data.

    Parameters
    ----------
    ds : xarray.Dataset
        Cover DataArray from a LULC Dataset from the ``nlcd`` function.

    Returns
    -------
    dict
        Statistics of NLCD cover data
    """
    nlcd_meta = helpers.nlcd_helper()
    cover_arr = ds.values
    total_pix = np.count_nonzero(~np.isnan(cover_arr))

    class_percentage = dict(
        zip(
            [v.split(" -")[0].strip() for v in nlcd_meta["classes"].values()],
            [
                cover_arr[cover_arr == int(cat)].shape[0] / total_pix * 100.0
                for cat in list(nlcd_meta["classes"].keys())
            ],
        )
    )

    cov = np.floor_divide(cover_arr[~np.isnan(cover_arr)], 10).astype("int")
    cat_list = (
        np.array([np.count_nonzero(cov == c) for c in range(10) if c != 6]) / total_pix * 100.0
    )

    category_percentage = dict(zip(list(nlcd_meta["categories"].keys()), cat_list))

    return {"classes": class_percentage, "categories": category_percentage}


class NWIS:
    """Access NWIS web service."""

    def __init__(self) -> None:
        self.url = ServiceURL().restful.nwis

    def get_info(
        self, queries: Union[Dict[str, str], List[Dict[str, str]]], expanded: bool = False
    ) -> pd.DataFrame:
        """Send multiple queries to USGS Site Web Service.

        Parameters
        ----------
        queries : dict or list of dict
            A single or a list of valid queries.
        expanded : bool, optional
            Whether to get expanded sit information for example drainage area, default to False.

        Returns
        -------
        pandas.DataFrame
            A typed dataframe containing the site information.
        """
        queries = [queries] if isinstance(queries, dict) else queries

        payloads = self._validate_usgs_queries(queries, False)
        sites = self.retrieve_rdb(f"{self.url}/site", payloads)

        float_cols = ["dec_lat_va", "dec_long_va", "alt_va", "alt_acy_va"]

        if expanded:
            payloads = self._validate_usgs_queries(queries, True)
            sites = sites.merge(
                self.retrieve_rdb(f"{self.url}/site", payloads),
                on="site_no",
                how="outer",
                suffixes=("", "_overlap"),
            )
            sites = sites.filter(regex="^(?!.*_overlap)")
            float_cols += ["drain_area_va", "contrib_drain_area_va"]

        sites[float_cols] = sites[float_cols].apply(pd.to_numeric, errors="coerce")

        try:
            sites["begin_date"] = pd.to_datetime(sites["begin_date"])
            sites["end_date"] = pd.to_datetime(sites["end_date"])
        except (AttributeError, KeyError):
            pass

        gii = WaterData("gagesii", DEF_CRS)

        def _get_hcdn(site_no: str) -> Optional[bool]:
            hcdn = gii.byid("staid", site_no)
            try:
                return len(hcdn.hcdn_2009.iloc[0]) > 0
            except (AttributeError, KeyError):
                return None

        sites["hcdn_2009"] = sites.site_no.apply(_get_hcdn)

        return sites

    def get_parameter_codes(self, keyword: str) -> pd.DataFrame:
        """Search for parameter codes by name or number.

        Notes
        -----
        NWIS guideline for keywords is as follows:

            By default an exact search is made. To make a partial search the term
            should be prefixed and suffixed with a % sign. The % sign matches zero
            or more characters at the location. For example, to find all with "discharge"
            enter %discharge% in the field. % will match any number of characters
            (including zero characters) at the location.

        Parameters
        ----------
        keyword : str
            Keyword to search for parameters by name of number.

        Returns
        -------
        pandas.DataFrame
            Matched parameter codes as a dataframe with their description.

        Examples
        --------
        >>> from pygeohydro import NWIS
        >>> nwis = NWIS()
        >>> codes = nwis.get_parameter_codes("%discharge%")
        >>> codes.loc[codes.parameter_cd == "00060", "parm_nm"][0]
        'Discharge, cubic feet per second'
        """
        url = "https://help.waterdata.usgs.gov/code/parameter_cd_nm_query"
        kwds = [{"parm_nm_cd": keyword, "fmt": "rdb"}]
        return self.retrieve_rdb(url, kwds)

    def get_streamflow(
        self,
        station_ids: Union[Sequence[str], str],
        dates: Tuple[str, str],
        mmd: bool = False,
    ) -> pd.DataFrame:
        """Get mean daily streamflow observations from USGS.

        Parameters
        ----------
        station_ids : str, list
            The gage ID(s)  of the USGS station.
        dates : tuple
            Start and end dates as a tuple (start, end).
        mmd : bool, optional
            Convert cms to mm/day based on the contributing drainage area of the stations.

        Returns
        -------
        pandas.DataFrame
            Streamflow data observations in cubic meter per second (cms). The stations that
            don't provide mean daily discharge in the target period will be dropped.
        """
        sids, start, end = self._check_inputs(station_ids, dates)
        queries = [
            {
                "parameterCd": "00060",
                "siteStatus": "all",
                "outputDataTypeCd": "dv",
                "sites": ",".join(s),
            }
            for s in tlz.partition_all(1500, sids)
        ]

        siteinfo = self.get_info(queries)
        check_dates = siteinfo.loc[
            ((siteinfo.stat_cd == "00003") & (start > siteinfo.end_date)),
            "site_no",
        ]
        sids = list(set(sids).difference({s for s in sids if s in check_dates}))
        if len(sids) == 0:
            raise DataNotAvailable("discharge")

        payloads = [
            {
                "format": "json",
                "sites": ",".join(s),
                "startDT": start.strftime("%Y-%m-%d"),
                "endDT": end.strftime("%Y-%m-%d"),
                "parameterCd": "00060",
                "statCd": "00003",
                "siteStatus": "all",
            }
            for s in tlz.partition_all(1500, sids)
        ]
        urls, kwds = zip(*((f"{self.url}/dv", {"params": p}) for p in payloads))
        qobs = self._to_dataframe(ar.retrieve(urls, "json", kwds))

        if qobs.shape[1] != len(station_ids):
            dropped = [s for s in station_ids if f"USGS-{s}" not in qobs]
            logger.warning(
                f"Dropped {len(dropped)} stations since they don't have daily mean discharge "
                + f"from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}."
            )

        if mmd:
            area = self._get_drainage_area(sids)
            ms2mmd = 1000.0 * 24.0 * 3600.0
            try:
                return qobs.apply(lambda x: x / area.loc[x.name.split("-")[-1]] * ms2mmd)
            except KeyError as ex:
                raise DataNotAvailable("drainage") from ex
        return qobs

    @staticmethod
    def _check_inputs(
        station_ids: Union[Sequence[str], str], dates: Tuple[str, str]
    ) -> Tuple[Sequence[str], pd.DatetimeIndex, pd.DatetimeIndex]:
        """Validate inputs."""
        if not isinstance(station_ids, (str, Sequence, Iterable)):
            raise InvalidInputType("ids", "str or list of str")

        sids = [station_ids] if isinstance(station_ids, str) else station_ids

        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InvalidInputType("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])
        return sids, start, end

    @staticmethod
    def _get_drainage_area(station_ids: List[str]) -> pd.DataFrame:
        nldi = NLDI()
        basins = nldi.get_basins(station_ids)
        if isinstance(basins, tuple):
            basins = basins[0]
        eck4 = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        return basins.to_crs(eck4).area

    @staticmethod
    def _to_dataframe(resp: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert json to dataframe."""
        r_ts = {
            t["sourceInfo"]["siteCode"][0]["value"]: t["values"][0]["value"]
            for r in resp
            for t in r["value"]["timeSeries"]
            if len(t["values"][0]["value"]) != 0
        }

        def to_df(col: str, dic: Dict[str, Any]) -> pd.DataFrame:
            discharge = pd.DataFrame.from_records(dic, exclude=["qualifiers"], index=["dateTime"])
            discharge.index = pd.to_datetime(discharge.index)
            discharge.columns = [col]
            return discharge

        qobs = pd.concat([to_df(f"USGS-{s}", t) for s, t in r_ts.items()], axis=1)
        # Convert cfs to cms
        return qobs.astype("float64") * 0.028316846592

    @staticmethod
    def retrieve_rdb(url: str, payloads: List[Dict[str, str]]) -> pd.DataFrame:
        """Retrieve and process requests with RDB format.

        Parameters
        ----------
        service : str
            Name of USGS REST service, valid values are ``site``, ``dv``, ``iv``,
            ``gwlevels``, and ``stat``. Please consult USGS documentation
            `here <https://waterservices.usgs.gov/rest>`__ for more information.
        payloads : list of dict
            List of target payloads.

        Returns
        -------
        pandas.DataFrame
            Requested features as a pandas's DataFrame.
        """
        urls, kwds = zip(*((url, {"params": {**p, "format": "rdb"}}) for p in payloads))
        resp = ar.retrieve(urls, "text", kwds)
        try:
            not_found = next(filter(lambda x: x[0] != "#", resp), None)
            if not_found is not None:
                msg = re.findall("<p>(.*?)</p>", not_found.text)[1].rsplit(">", 1)[1]
                logger.info(f"Server error message:\n{msg}")
        except StopIteration:
            pass
        data = [r.strip().split("\n") for r in resp if r[0] == "#"]
        data = [t.split("\t") for d in data for t in d if "#" not in t]
        if len(data) == 0:
            raise ZeroMatched

        rdb_df = pd.DataFrame.from_dict(dict(zip(data[0], d)) for d in data[2:])
        if "agency_cd" in rdb_df:
            rdb_df = rdb_df[~rdb_df.agency_cd.str.contains("agency_cd|5s")].copy()
        return rdb_df

    @staticmethod
    def _validate_usgs_queries(
        queries: List[Dict[str, str]], expanded: bool = False
    ) -> List[Dict[str, str]]:
        """Validate queries to be used with USGS Site Web Service.

        Parameters
        ----------
        queries : list of dict
            List of valid queries.
        expanded : bool, optional
            Get expanded sit information such as drainage area, default to False.

        Returns
        -------
        list of dict
            Validated queries with additional keys for format and site output (basic/expanded).
        """
        if not (isinstance(queries, (list, tuple)) and all(isinstance(q, dict) for q in queries)):
            raise InvalidInputType("query", "list of dict")

        valid_query_keys = [
            "site",
            "sites",
            "location",
            "stateCd",
            "stateCds",
            "huc",
            "hucs",
            "bBox",
            "countyCd",
            "countyCds",
            "format",
            "siteOutput",
            "site_output",
            "seriesCatalogOutput",
            "seriesCatalog",
            "outputDataTypeCd",
            "outputDataType",
            "startDt",
            "endDt",
            "period",
            "modifiedSince",
            "siteName",
            "siteNm",
            "stationName",
            "stationNm",
            "siteNameOperator",
            "siteNameMatch",
            "siteNmMatch",
            "stationNameMatch",
            "stationNmMatch",
            "siteStatus",
            "siteType",
            "siteTypes",
            "siteTypeCd",
            "siteTypeCds",
            "hasDataTypeCd",
            "hasDataType",
            "dataTypeCd",
            "dataType",
            "parameterCd",
            "variable",
            "parameterCds",
            "variables",
            "var",
            "vars",
            "parmCd",
            "agencyCd",
            "agencyCds",
            "altMin",
            "altMinVa",
            "altMax",
            "altMaxVa",
            "drainAreaMin",
            "drainAreaMinVa",
            "drainAreaMax",
            "drainAreaMaxVa",
            "aquiferCd",
            "localAquiferCd",
            "wellDepthMin",
            "wellDepthMinVa",
            "wellDepthMax",
            "wellDepthMaxVa",
            "holdDepthMin",
            "holdDepthMinVa",
            "holeDepthMax",
            "holeDepthMaxVa",
        ]

        not_valid = list(tlz.concat(set(q).difference(set(valid_query_keys)) for q in queries))
        if len(not_valid) > 0:
            raise InvalidInputValue(f"query keys ({', '.join(not_valid)})", valid_query_keys)

        _queries = queries.copy()
        if expanded:
            _ = [
                q.pop(k) for k in ["outputDataTypeCd", "outputDataType"] for q in _queries if k in q
            ]
            output_type = {"siteOutput": "expanded"}
        else:
            output_type = {"siteOutput": "basic"}

        return [{**query, **output_type, "format": "rdb"} for query in _queries]


def interactive_map(
    bbox: Tuple[float, float, float, float],
    crs: str = DEF_CRS,
    nwis_kwds: Optional[Dict[str, Any]] = None,
) -> folium.Map:
    """Generate an interactive map including all USGS stations within a bounding box.

    Parameters
    ----------
    bbox : tuple
        List of corners in this order (west, south, east, north)
    crs : str, optional
        CRS of the input bounding box, defaults to EPSG:4326.
    nwis_kwds : dict, optional
        Optional keywords to include in the NWIS request as a dictionary like so:
        ``{"hasDataTypeCd": "dv,iv", "outputDataTypeCd": "dv,iv", "parameterCd": "06000"}``.
        Default to None.

    Returns
    -------
    folium.Map
        Interactive map within a bounding box.

    Examples
    --------
    >>> import pygeohydro as gh
    >>> nwis_kwds = {"hasDataTypeCd": "dv,iv", "outputDataTypeCd": "dv,iv"}
    >>> m = gh.interactive_map((-69.77, 45.07, -69.31, 45.45), nwis_kwds=nwis_kwds)
    >>> n_stations = len(m.to_dict()["children"]) - 1
    >>> n_stations
    10
    """
    bbox = ogc.utils.match_crs(bbox, crs, DEF_CRS)
    ogc.utils.check_bbox(bbox)

    nwis = NWIS()
    query = {"bBox": ",".join(f"{b:.06f}" for b in bbox)}
    if isinstance(nwis_kwds, dict):
        query.update(nwis_kwds)

    sites = nwis.get_info(query, expanded=True)

    sites["coords"] = list(sites[["dec_long_va", "dec_lat_va"]].itertuples(name=None, index=False))
    sites["altitude"] = (
        sites["alt_va"].astype("str") + " ft above " + sites["alt_datum_cd"].astype("str")
    )

    sites["drain_area_va"] = sites["drain_area_va"].astype("str") + " square miles"
    sites["contrib_drain_area_va"] = sites["contrib_drain_area_va"].astype("str") + " square miles"

    cols_old = [
        "site_no",
        "station_nm",
        "coords",
        "altitude",
        "huc_cd",
        "drain_area_va",
        "contrib_drain_area_va",
        "hcdn_2009",
    ]

    cols_new = [
        "Site No.",
        "Station Name",
        "Coordinate",
        "Altitude",
        "HUC8",
        "Drainage Area",
        "Contributing Drainage Area",
        "HCDN 2009",
    ]

    sites = sites.groupby("site_no").agg(set).reset_index()
    sites = sites.rename(columns=dict(zip(cols_old, cols_new)))[cols_new]

    msgs = []
    base_url = "https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no="
    for row in sites.itertuples(index=False):
        site_no = row[sites.columns.get_loc(cols_new[0])]
        msg = f"<strong>{cols_new[0]}</strong>: {site_no}<br>"
        for col in cols_new[1:]:
            value = ", ".join(str(s) for s in row[sites.columns.get_loc(col)])
            msg += f"<strong>{col}</strong>: {value}<br>"
        msg += f'<a href="{base_url}{site_no}" target="_blank">More on USGS Website</a>'
        msgs.append(msg[:-4])

    sites["msg"] = msgs

    west, south, east, north = bbox
    lon = (west + east) * 0.5
    lat = (south + north) * 0.5

    imap = folium.Map(
        location=(lat, lon),
        tiles="Stamen Terrain",
        zoom_start=10,
    )

    for coords, msg in sites[["Coordinate", "msg"]].itertuples(name=None, index=False):
        folium.Marker(
            location=list(coords)[0][::-1],
            popup=folium.Popup(msg, max_width=250),
            icon=folium.Icon(),
        ).add_to(imap)

    return imap
