"""Accessing data from the supported databases through their APIs."""
import contextlib
import io
import re
import warnings
import zipfile
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from unittest.mock import patch

import async_retriever as ar
import cytoolz as tlz
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
import rasterio as rio
import xarray as xr
from pygeoogc import WMS, RetrySession, ServiceURL
from pygeoogc import ZeroMatched as ZeroMatchedOGC
from pygeoogc import utils as ogc_utils
from pynhd import NLDI, AGRBase, WaterData
from shapely.geometry import MultiPolygon, Polygon

from . import helpers
from .exceptions import (
    DataNotAvailable,
    InvalidInputType,
    InvalidInputValue,
    MissingCRS,
    ServiceUnavailable,
    ZeroMatched,
)
from .helpers import logger

DEF_CRS = "epsg:4326"
EXPIRE = -1


def ssebopeta_byloc(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
) -> pd.DataFrame:
    """Daily actual ET for a location from SSEBop database in mm/day.

    .. deprecated:: 0.11.5
        Use :func:`ssebopeta_bycoords` instead.

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
    msg = " ".join(
        [
            "This function is deprecated and will be remove in future versions.",
            "Please use `ssebopeta_bycoords` instead.",
            "For now, this function calls `ssebopeta_bycoords`.",
        ]
    )
    warnings.warn(msg, DeprecationWarning)
    return ssebopeta_bycoords(coords, dates)


def ssebopeta_bycoords(
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

    f_list = helpers.get_ssebopeta_urls(dates)
    session = RetrySession()

    with patch("socket.has_ipv6", False):

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

    f_list = helpers.get_ssebopeta_urls(dates)

    session = RetrySession()

    with patch("socket.has_ipv6", False):

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


class _NLCD:
    """Get data from NLCD database (2019).

    Parameters
    ----------
    years : dict, optional
        The years for NLCD layers as a dictionary, defaults to
        ``{'impervious': [2019], 'cover': [2019], 'canopy': [2019], "descriptor": [2019]}``.
        Layers that are not in years are ignored, e.g., ``{'cover': [2016, 2019]}`` returns
        land cover data for 2016 and 2019.
    region : str, optional
        Region in the US, defaults to ``L48``. Valid values are L48 (for CONUS), HI (for Hawaii),
        AK (for Alaska), and PR (for Puerto Rico). Both lower and upper cases are acceptable.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    validation : bool, optional
        Validate the input arguments from the WMS service, defaults to True. Set this
        to False if you are sure all the WMS settings such as layer and crs are correct
        to avoid sending extra requests.
    expire_after : int, optional
        Expiration time for response caching in seconds, defaults to -1 (never expire).
    disable_caching : bool, optional
        If ``True``, disable caching requests, defaults to False.
    """

    def __init__(
        self,
        years: Optional[Mapping[str, Union[int, List[int]]]] = None,
        region: str = "L48",
        crs: str = DEF_CRS,
        validation: bool = True,
        expire_after: float = EXPIRE,
        disable_caching: bool = False,
    ) -> None:
        default_years = {
            "impervious": [2019],
            "cover": [2019],
            "canopy": [2016],
            "descriptor": [2019],
        }
        years = default_years if years is None else years
        if not isinstance(years, dict):
            raise InvalidInputType("years", "dict", f"{default_years}")
        self.years = tlz.valmap(lambda x: x if isinstance(x, list) else [x], years)
        self.region = region
        self.crs = crs
        self.validation = validation
        self.expire_after = expire_after
        self.disable_caching = disable_caching
        self.layers = self.get_layers(self.region, self.years)
        self.units = OrderedDict(
            (("impervious", "%"), ("cover", "classes"), ("canopy", "%"), ("descriptor", "classes"))
        )
        self.types = OrderedDict(
            (("impervious", "f4"), ("cover", "u1"), ("canopy", "f4"), ("descriptor", "u1"))
        )
        self.nodata = OrderedDict(
            (("impervious", 0), ("cover", 127), ("canopy", 0), ("descriptor", 127))
        )

        self.wms = WMS(
            ServiceURL().wms.mrlc,
            layers=self.layers,
            outformat="image/geotiff",
            crs=self.crs,
            validation=self.validation,
            expire_after=self.expire_after,
            disable_caching=self.disable_caching,
        )

    def get_response(
        self, bounds: Tuple[float, float, float, float], resolution: float
    ) -> Dict[str, bytes]:
        """Get response from a url."""
        return self.wms.getmap_bybox(bounds, resolution, self.crs, kwargs={"styles": "raster"})

    def to_xarray(self, r_dict: Dict[str, bytes], geometry: Polygon) -> xr.DataArray:
        """Convert response to xarray.DataArray."""
        try:
            ds = geoutils.gtiff2xarray(r_dict, geometry, self.crs)
        except rio.RasterioIOError as ex:
            raise ServiceUnavailable(self.wms.url) from ex
        attrs = ds.attrs
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
        for lyr in self.layers:
            name = [n for n in self.units if n in lyr.lower()][-1]
            lyr_name = f"{name}_{lyr.split('_')[1]}"
            ds = ds.rename({lyr: lyr_name})
            ds[lyr_name].attrs["units"] = self.units[name]
            ds[lyr_name] = ds[lyr_name].astype(self.types[name])
            ds[lyr_name].attrs["nodatavals"] = (self.nodata[name],)
        ds.attrs = attrs
        return ds

    @staticmethod
    def get_layers(region: str, years: Dict[str, List[int]]) -> List[str]:
        """Get NLCD layers for the provided years dictionary."""
        valid_regions = ["L48", "HI", "PR", "AK"]
        region = region.upper()
        if region not in valid_regions:
            raise InvalidInputValue("region", valid_regions)

        nlcd_meta = helpers.nlcd_helper()

        names = ["impervious", "cover", "canopy", "descriptor"]
        avail_years = {n: nlcd_meta[f"{n}_years"] for n in names}

        if any(
            yr not in avail_years[lyr] or lyr not in names
            for lyr, yrs in years.items()
            for yr in yrs
        ):
            vals = [f"\n{lyr}: {', '.join(str(y) for y in yr)}" for lyr, yr in avail_years.items()]
            raise InvalidInputValue("years", vals)

        layer_map = {
            "canopy": lambda _: "Tree_Canopy",
            "cover": lambda _: "Land_Cover_Science_Product",
            "impervious": lambda _: "Impervious",
            "descriptor": lambda x: "Impervious_Descriptor"
            if x == "AK"
            else "Impervious_descriptor",
        }

        return [
            f"NLCD_{yr}_{layer_map[lyr](region)}_{region}"
            for lyr, yrs in years.items()
            for yr in yrs
        ]


def nlcd_bygeom(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float], gpd.GeoDataFrame],
    resolution: float,
    years: Optional[Mapping[str, Union[int, List[int]]]] = None,
    region: str = "L48",
    geo_crs: str = DEF_CRS,
    crs: str = DEF_CRS,
    validation: bool = True,
    expire_after: float = EXPIRE,
    disable_caching: bool = False,
) -> Union[xr.Dataset, Dict[int, xr.Dataset]]:
    """Get data from NLCD database (2019).

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, tuple of length 4, or GeoDataFrame
        The geometry or bounding box (west, south, east, north) for extracting the data.
        You can either pass a single geometry or a ``GeoDataFrame`` with a geometry column.
    resolution : float
        The data resolution in meters. The width and height of the output are computed in pixel
        based on the geometry bounds and the given resolution.
    years : dict, optional
        The years for NLCD layers as a dictionary, defaults to
        ``{'impervious': [2019], 'cover': [2019], 'canopy': [2019], "descriptor": [2019]}``.
        Layers that are not in years are ignored, e.g., ``{'cover': [2016, 2019]}`` returns
        land cover data for 2016 and 2019.
    region : str, optional
        Region in the US, defaults to ``L48``. Valid values are ``L48`` (for CONUS),
        ``HI`` (for Hawaii), ``AK`` (for Alaska), and ``PR`` (for Puerto Rico).
        Both lower and upper cases are acceptable.
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    validation : bool, optional
        Validate the input arguments from the WMS service, defaults to True. Set this
        to False if you are sure all the WMS settings such as layer and crs are correct
        to avoid sending extra requests.
    expire_after : int, optional
        Expiration time for response caching in seconds, defaults to -1 (never expire).
    disable_caching : bool, optional
        If ``True``, disable caching requests, defaults to False.

    Returns
    -------
    xarray.Dataset or dict of xarray.Dataset
        NLCD within a geometry or a dict of NLCD datasets within corresponding geometries.
        If dict, the keys are indices of the input ``GeoDataFrame``.
    """
    if resolution < 30:
        logger.warning("NLCD's resolution is 30 m, so finer resolutions are not recommended.")

    if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
        single_geom = False
        if geometry.crs is None:
            raise MissingCRS
        _geometry = geometry.to_crs(crs).geometry.to_dict()
    else:
        single_geom = True
        _geometry = {0: geoutils.geo2polygon(geometry, geo_crs, crs)}

    nlcd_wms = _NLCD(
        years=years,
        region=region,
        crs=crs,
        validation=validation,
        expire_after=expire_after,
        disable_caching=disable_caching,
    )

    ds = {
        i: nlcd_wms.to_xarray(nlcd_wms.get_response(g.bounds, resolution), g)
        for i, g in _geometry.items()
    }
    if single_geom:
        return ds[0]
    return ds


def nlcd_bycoords(
    coords: List[Tuple[float, float]],
    years: Optional[Mapping[str, Union[int, List[int]]]] = None,
    region: str = "L48",
    expire_after: float = EXPIRE,
    disable_caching: bool = False,
) -> gpd.GeoDataFrame:
    """Get data from NLCD database (2019).

    Parameters
    ----------
    coords : list of tuple
        List of coordinates in the form of (longitude, latitude).
    years : dict, optional
        The years for NLCD layers as a dictionary, defaults to
        ``{'impervious': [2019], 'cover': [2019], 'canopy': [2019], "descriptor": [2019]}``.
        Layers that are not in years are ignored, e.g., ``{'cover': [2016, 2019]}`` returns
        land cover data for 2016 and 2019.
    region : str, optional
        Region in the US, defaults to ``L48``. Valid values are ``L48`` (for CONUS),
        ``HI`` (for Hawaii), ``AK`` (for Alaska), and ``PR`` (for Puerto Rico).
        Both lower and upper cases are acceptable.
    expire_after : int, optional
        Expiration time for response caching in seconds, defaults to -1 (never expire).
    disable_caching : bool, optional
        If ``True``, disable caching requests, defaults to False.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with the NLCD data and the coordinates.
    """
    if not isinstance(coords, list) or any(len(c) != 2 for c in coords):
        raise InvalidInputType("coords", "list of (lon, lat)")

    nlcd_wms = _NLCD(
        years=years,
        region=region,
        crs=DEF_CRS,
        validation=False,
        expire_after=expire_after,
        disable_caching=disable_caching,
    )
    points = gpd.GeoSeries(gpd.points_from_xy(*zip(*coords)), crs=DEF_CRS)
    bounds = points.to_crs(points.estimate_utm_crs()).buffer(35, cap_style=3)
    bounds = bounds.to_crs(DEF_CRS)
    ds_list = [nlcd_wms.to_xarray(nlcd_wms.get_response(b.bounds, 30), b.bounds) for b in bounds]

    def get_value(da: xr.DataArray, x: float, y: float) -> Union[int, float]:
        nodata = da.attrs["nodatavals"][0]
        return (
            da.fillna(nodata).interp(x=[x], y=[y], method="nearest").astype(da.dtype).values[0][0]
        )

    values = {
        v: [get_value(ds[v], x, y) for ds, (x, y) in zip(ds_list, coords)] for v in ds_list[0]
    }
    return points.to_frame("geometry").merge(
        pd.DataFrame(values), left_index=True, right_index=True
    )


def nlcd(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    resolution: float,
    years: Optional[Mapping[str, Union[int, List[int]]]] = None,
    region: str = "L48",
    geo_crs: str = DEF_CRS,
    crs: str = DEF_CRS,
) -> xr.Dataset:
    """Get data from NLCD database (2019).

    .. deprecated:: 0.11.5
        Use :func:`nlcd_bygeom` or :func:`nlcd_bycoords`  instead.

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
        Region in the US, defaults to ``L48``. Valid values are ``L48`` (for CONUS),
        ``HI`` (for Hawaii), ``AK`` (for Alaska), and ``PR`` (for Puerto Rico).
        Both lower and upper cases are acceptable.
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
    msg = " ".join(
        [
            "This function is deprecated and will be remove in future versions.",
            "Please use `nlcd_bygeom` or `nlcd_bycoords` instead.",
            "For now, this function calls `nlcd_bygeom` to retain the original functionality.",
        ]
    )
    warnings.warn(msg, DeprecationWarning)
    return nlcd_bygeom(geometry, resolution, years, region, geo_crs, crs)


def cover_statistics(ds: xr.Dataset) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
    """Percentages of the categorical NLCD cover data.

    Parameters
    ----------
    ds : xarray.DataArray
        Cover DataArray from a LULC Dataset from the ``nlcd`` function.

    Returns
    -------
    dict
        Statistics of NLCD cover data
    """
    if not isinstance(ds, xr.DataArray):
        raise InvalidInputType("ds", "xarray.DataArray")

    nlcd_meta = helpers.nlcd_helper()
    val, freq = np.unique(ds, return_counts=True)
    zero_idx = np.argwhere(val == 0)
    val = np.delete(val, zero_idx).astype(str)
    freq = np.delete(freq, zero_idx)
    freq_dict = dict(zip(val, freq))
    total_count = freq.sum()

    if any(c not in nlcd_meta["classes"] for c in freq_dict):
        raise InvalidInputValue("ds values", list(nlcd_meta["classes"]))  # noqa: TC003

    class_percentage = {
        nlcd_meta["classes"][k].split(" -")[0].strip(): v / total_count * 100.0
        for k, v in freq_dict.items()
    }
    category_percentage = {
        k: sum(freq_dict[c] for c in v if c in freq_dict) / total_count * 100.0
        for k, v in nlcd_meta["categories"].items()
    }

    return {"classes": class_percentage, "categories": category_percentage}


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

    sites["drain_area_va"] = sites["drain_area_va"].astype("str") + " sqmi"
    sites["contrib_drain_area_va"] = sites["contrib_drain_area_va"].astype("str") + " sqmi"
    sites["drain_sqkm"] = sites["drain_sqkm"].astype("str") + " sqkm"
    for c in ["drain_area_va", "contrib_drain_area_va", "drain_sqkm"]:
        sites.loc[sites[c].str.contains("nan"), c] = "N/A"

    cols_old = [
        "site_no",
        "station_nm",
        "coords",
        "altitude",
        "huc_cd",
        "drain_area_va",
        "contrib_drain_area_va",
        "drain_sqkm",
        "hcdn_2009",
    ]

    cols_new = [
        "Site No.",
        "Station Name",
        "Coordinate",
        "Altitude",
        "HUC8",
        "Drainage Area (NWIS)",
        "Contributing Drainage Area (NWIS)",
        "Drainage Area (GagesII)",
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


class NID(AGRBase):
    """Retrieve data from the National Inventory of Dams web service.

    Parameters
    ----------
    version : str, optional
        The database version. Version 2 and 3 are available. Version 2 has
        NID 2019 data and version 3 includes more recent data as well. At the
        moment both services are experimental and might not always work. The
        default version is 2. More information can be found at https://damsdev.net.
    """

    def __init__(self, version: int = 2) -> None:
        if version not in (2, 3):
            raise InvalidInputValue("version", ["2", "3"])

        layer = "nid2019_u" if version == 2 else "dams"
        super().__init__(layer, "*", DEF_CRS)
        self.service = getattr(ServiceURL().restful, f"nid_{int(version)}")
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

        def _get_hcdn(site_no: str) -> Tuple[float, Optional[bool]]:
            try:
                gage = gii.byid("staid", site_no)
                return gage.drain_sqkm.iloc[0], len(gage.hcdn_2009.iloc[0]) > 0  # noqa: TC300
            except (AttributeError, KeyError, ZeroMatchedOGC):
                return np.nan, None

        sites["drain_sqkm"], sites["hcdn_2009"] = zip(*[_get_hcdn(n) for n in sites.site_no])

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
        freq: str = "dv",
        mmd: bool = False,
        to_xarray: bool = False,
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """Get mean daily streamflow observations from USGS.

        Parameters
        ----------
        station_ids : str, list
            The gage ID(s)  of the USGS station.
        dates : tuple
            Start and end dates as a tuple (start, end).
        freq : str, optional
            The frequency of the streamflow data, defaults to ``dv`` (daily values).
            Valid frequencies are ``dv`` (daily values), ``iv`` (instantaneous values).
            Note that for ``iv`` the time zone for the input dates is assumed to be UTC.
        mmd : bool, optional
            Convert cms to mm/day based on the contributing drainage area of the stations.
            Defaults to False.
        to_xarray : bool, optional
            Whether to return a xarray.Dataset. Defaults to False.

        Returns
        -------
        pandas.DataFrame or xarray.Dataset
            Streamflow data observations in cubic meter per second (cms). The stations that
            don't provide the requested discharge data in the target period will be dropped.
            Note that when frequency is set to ``iv`` the time zone is converted to UTC.
        """
        valid_freqs = ["dv", "iv"]
        if freq not in valid_freqs:
            raise InvalidInputValue("freq", valid_freqs)
        utc = True if freq == "iv" else None

        sids, start, end = self._check_inputs(station_ids, dates, utc)

        queries = [
            {
                "parameterCd": "00060",
                "siteStatus": "all",
                "outputDataTypeCd": freq,
                "sites": ",".join(s),
            }
            for s in tlz.partition_all(1500, sids)
        ]

        siteinfo = self.get_info(queries)
        sids = siteinfo.site_no.unique()
        if len(sids) == 0:
            raise DataNotAvailable("discharge")

        params = {
            "format": "json",
            "parameterCd": "00060",
            "siteStatus": "all",
        }
        if freq == "dv":
            params.update({"statCd": "00003"})
            cond = (
                (siteinfo.stat_cd == "00003")
                & (siteinfo.parm_cd == "00060")
                & (start.tz_localize(None) < siteinfo.end_date)
                & (end.tz_localize(None) > siteinfo.begin_date)
            )
        else:
            cond = (
                (siteinfo.parm_cd == "00060")
                & (start.tz_localize(None) < siteinfo.end_date)
                & (end.tz_localize(None) > siteinfo.begin_date)
            )

        time_fmt = "%Y-%m-%d" if utc is None else "%Y-%m-%dT%H:%M%z"
        startDT = start.strftime(time_fmt)
        endDT = end.strftime(time_fmt)
        qobs = self._get_streamflow(sids, startDT, endDT, freq, params)

        dropped = [s for s in sids if f"USGS-{s}" not in qobs]
        if len(dropped) > 0:
            logger.warning(
                f"Dropped {len(dropped)} stations since they don't have discharge data"
                + f" from {startDT} to {endDT}."
            )

        if mmd:
            area = self._get_drainage_area(sids)
            ms2mmd = 1000.0 * 24.0 * 3600.0
            try:
                qobs = qobs.apply(lambda x: x / area.loc[x.name.split("-")[-1]] * ms2mmd)
            except KeyError as ex:
                raise DataNotAvailable("drainage") from ex

        qobs.attrs, long_names = self._get_attrs(siteinfo.loc[cond], mmd)
        if to_xarray:
            return self._to_xarray(qobs, long_names, mmd)
        return qobs

    @staticmethod
    def _to_xarray(qobs: pd.DataFrame, long_names: Dict[str, str], mmd: bool) -> xr.Dataset:
        """Convert a pandas.DataFrame to an xarray.Dataset."""
        ds = xr.Dataset(
            data_vars={
                "discharge": (["time", "station_id"], qobs),
                **{
                    attr: (["station_id"], v)
                    for attr, *v in pd.DataFrame(qobs.attrs).iloc[:-2].itertuples()
                },
            },
            coords={
                "time": qobs.index.tz_localize(None).to_numpy(),
                "station_id": qobs.columns,
            },
        )
        for v, n in long_names.items():
            ds[v].attrs["long_name"] = n
        ds.attrs["tz"] = "UTC"
        ds["discharge"].attrs["units"] = "mm/day" if mmd else "cms"
        ds["dec_lat_va"].attrs["units"] = "decimal_degrees"
        ds["dec_long_va"].attrs["units"] = "decimal_degrees"
        ds["alt_va"].attrs["units"] = "ft"
        return ds

    @staticmethod
    def _get_attrs(siteinfo: pd.DataFrame, mmd: bool) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Get attributes of the stations that have streaflow data."""
        cols = {
            "site_no": "site_identification_number",
            "station_nm": "station_name",
            "dec_lat_va": "latitude",
            "dec_long_va": "longitude",
            "alt_va": "altitude",
            "alt_acy_va": "altitude_accuracy",
            "alt_datum_cd": "altitude_datum",
            "huc_cd": "hydrologic_unit_code",
        }
        if "begin_date" in siteinfo and "end_date" in siteinfo:
            cols.update(
                {
                    "begin_date": "availablity_begin_date",
                    "end_date": "availablity_end_date",
                }
            )
        attr_df = siteinfo[cols.keys()].drop_duplicates().set_index("site_no")
        if "begin_date" in attr_df and "end_date" in attr_df:
            attr_df["begin_date"] = attr_df.begin_date.dt.strftime("%Y-%m-%d")
            attr_df["end_date"] = attr_df.end_date.dt.strftime("%Y-%m-%d")
        attr_df.index = "USGS-" + attr_df.index
        attr_df["units"] = "mm/day" if mmd else "cms"
        attr_df["tz"] = "UTC"
        _ = cols.pop("site_no")
        return attr_df.to_dict(orient="index"), cols

    @staticmethod
    def _check_inputs(
        station_ids: Union[Sequence[str], str], dates: Tuple[str, str], utc: Optional[bool]
    ) -> Tuple[List[str], pd.DatetimeIndex, pd.DatetimeIndex]:
        """Validate inputs."""
        if not isinstance(station_ids, (str, Sequence, Iterable)):
            raise InvalidInputType("ids", "str or list of str")

        sids = [station_ids] if isinstance(station_ids, str) else list(set(station_ids))

        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InvalidInputType("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0], utc=utc)
        end = pd.to_datetime(dates[1], utc=utc)
        return sids, start, end

    @staticmethod
    def _get_drainage_area(station_ids: Sequence[str]) -> pd.DataFrame:
        nldi = NLDI()
        basins = nldi.get_basins(list(station_ids))
        if isinstance(basins, tuple):
            basins = basins[0]
        return basins.to_crs("EPSG:6350").area

    def _get_streamflow(
        self, sids: List[str], startDT: str, endDT: str, freq: str, kwargs: Dict[str, str]
    ) -> pd.DataFrame:
        """Convert json to dataframe."""
        payloads = [
            {
                "sites": ",".join(s),
                "startDT": startDT,
                "endDT": endDT,
                **kwargs,
            }
            for s in tlz.partition_all(1500, sids)
        ]
        urls, kwds = zip(*((f"{self.url}/{freq}", {"params": p}) for p in payloads))
        resp = ar.retrieve(urls, "json", kwds)
        r_ts = {
            t["sourceInfo"]["siteCode"][0]["value"]: t["values"][0]["value"]
            for r in resp
            for t in r["value"]["timeSeries"]
            if len(t["values"][0]["value"]) > 0
        }
        if len(r_ts) == 0:
            raise DataNotAvailable("discharge")

        def to_df(col: str, dic: Dict[str, Any]) -> pd.DataFrame:
            discharge = pd.DataFrame.from_records(dic, exclude=["qualifiers"], index=["dateTime"])
            discharge.index = pd.to_datetime(discharge.index, infer_datetime_format=True)
            if discharge.index.tz is None:
                tz = resp[0]["value"]["timeSeries"][0]["sourceInfo"]["timeZoneInfo"]
                discharge.index = discharge.index.tz_localize(
                    tz["defaultTimeZone"]["zoneAbbreviation"]
                )
            discharge.index = discharge.index.tz_convert("UTC")
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
        try:
            resp = ar.retrieve(urls, "text", kwds)
        except ar.ServiceError as ex:
            raise ZeroMatched(ogc_utils.check_response(str(ex))) from ex

        with contextlib.suppress(StopIteration):
            not_found = next(filter(lambda x: x[0] != "#", resp), None)
            if not_found is not None:
                msg = re.findall("<p>(.*?)</p>", not_found)[1].rsplit(">", 1)[1]
                raise ZeroMatched(f"Server error message:\n{msg}")

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


class WaterQuality:
    """Water Quality Web Service https://www.waterqualitydata.us.

    Notes
    -----
    This class has a number of convenience methods to retrieve data from the
    Water Quality Data. Since there are many parameter combinations that can be
    used to retrieve data, a general method is also provided to retrieve data from
    any of the valid endpoints. You can use ``get_json`` to retrieve stations info
    as a ``geopandas.GeoDataFrame`` or ``get_csv`` to retrieve stations data as a
    ``pandas.DataFrame``. You can construct a dictionary of the parameters and pass
    it to one of these functions. For more information on the parameters, please
    consult the
    `Water Quality Data documentation <https://www.waterqualitydata.us/webservices_documentation>`__.
    """

    def __init__(self) -> None:
        self.wq_url = "https://www.waterqualitydata.us"
        self.keywords = self.get_param_table()

    def _base_url(self, endpoint: str) -> str:
        """Get the base URL for the target endpoint."""
        valid_endpoints = [
            "Station",
            "Result",
            "Activity",
            "ActivityMetric",
            "ProjectMonitoringLocationWeighting",
            "ResultDetectionQuantitationLimit",
            "BiologicalMetric",
        ]
        if endpoint.lower() not in map(str.lower, valid_endpoints):
            raise InvalidInputValue("endpoint", valid_endpoints)
        return f"{self.wq_url}/data/{endpoint}/search"

    def lookup_domain_values(self, endpoint: str) -> List[str]:
        """Get the domain values for the target endpoint."""
        valid_endpoints = [
            "statecode",
            "countycode",
            "sitetype",
            "organization",
            "samplemedia",
            "characteristictype",
            "characteristicname",
            "providers",
        ]
        if endpoint.lower() not in valid_endpoints:
            raise InvalidInputValue("endpoint", valid_endpoints)
        resp = ar.retrieve([f"{self.wq_url}/Codes/{endpoint}?mimeType=json"], "json")
        return [r["value"] for r in resp[0]["codes"]]

    def get_param_table(self) -> pd.DataFrame:
        """Get the parameter table from the USGS Water Quality Web Service."""
        params = pd.read_html(f"{self.wq_url}/webservices_documentation/")
        params = params[0].iloc[:29].drop(columns="Discussion")
        return params.groupby("REST parameter")["Argument"].apply(",".join)

    def station_bybbox(
        self, bbox: Tuple[float, float, float, float], wq_kwds: Optional[Dict[str, str]]
    ) -> gpd.GeoDataFrame:
        """Retrieve station info within bounding box.

        Parameters
        ----------
        bbox : tuple of float
            Bounding box coordinates (west, south, east, north) in epsg:4326.
        wq_kwds : dict, optional
            Water Quality Web Service keyword arguments. Default to None.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of station info within the bounding box.
        """
        kwds = {
            "mimeType": "geojson",
            "bBox": ",".join(f"{b:.06f}" for b in bbox),
            "zip": "no",
            "sorted": "no",
        }
        if wq_kwds is not None:
            self._check_kwds(wq_kwds)
            kwds.update(wq_kwds)

        return self.get_json("station", kwds)

    def station_bydistance(
        self, lon: float, lat: float, radius: float, wq_kwds: Optional[Dict[str, str]]
    ) -> gpd.GeoDataFrame:
        """Retrieve station within a radius (decimal miles) of a point.

        Parameters
        ----------
        lon : float
            Longitude of point.
        lat : float
            Latitude of point.
        radius : float
            Radius (decimal miles) of search.
        wq_kwds : dict, optional
            Water Quality Web Service keyword arguments. Default to None.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of station info within the radius of the point.
        """
        kwds = {
            "mimeType": "geojson",
            "long": f"{lon:.06f}",
            "lat": f"{lat:.06f}",
            "within": f"{radius:.06f}",
            "zip": "no",
            "sorted": "no",
        }
        if wq_kwds is not None:
            self._check_kwds(wq_kwds)
            kwds.update(wq_kwds)

        return self.get_json("station", kwds)

    def data_bystation(
        self, station_ids: Union[str, List[str]], wq_kwds: Optional[Dict[str, str]]
    ) -> pd.DataFrame:
        """Retrieve data for a single station.

        Parameters
        ----------
        station_ids : str or list of str
            Station ID(s). The IDs should have the format "Agency code-Station ID".
        wq_kwds : dict, optional
            Water Quality Web Service keyword arguments. Default to None.

        Returns
        -------
        pandas.DataFrame
            DataFrame of data for the stations.
        """
        siteid = set(station_ids) if isinstance(station_ids, list) else {station_ids}
        if any("-" not in s for s in siteid):
            valid_type = "list of hyphenated IDs like so 'agency code-station ID'"
            raise InvalidInputType("station_ids", valid_type)
        kwds = {
            "mimeType": "csv",
            "siteid": ";".join(siteid),
            "zip": "yes",
            "sorted": "no",
        }
        if wq_kwds is not None:
            self._check_kwds(wq_kwds)
            kwds.update(wq_kwds)

        if len(siteid) > 10:
            return self.get_csv("result", kwds, request_method="POST")
        return self.get_csv("result", kwds)

    def _check_kwds(self, wq_kwds: Dict[str, str]) -> None:
        """Check the validity of the Water Quality Web Service keyword arguments."""
        invalids = [k for k in wq_kwds if k not in self.keywords.index]
        if len(invalids) > 0:
            raise InvalidInputValue("wq_kwds", invalids)

    def get_json(
        self, endpoint: str, kwds: Dict[str, str], request_method: str = "GET"
    ) -> gpd.GeoDataFrame:
        """Get the JSON response from the Water Quality Web Service.

        Parameters
        ----------
        endpoint : str
            Endpoint of the Water Quality Web Service.
        kwds : dict
            Water Quality Web Service keyword arguments.
        request_method : str, optional
            HTTP request method. Default to GET.

        Returns
        -------
        geopandas.GeoDataFrame
            The web service response as a GeoDataFrame.
        """
        req_kwds = [{"params": kwds}] if request_method == "GET" else [{"data": kwds}]
        r = ar.retrieve([self._base_url(endpoint)], "json", req_kwds, request_method=request_method)
        return geoutils.json2geodf(r)

    def get_csv(
        self, endpoint: str, kwds: Dict[str, str], request_method: str = "GET"
    ) -> pd.DataFrame:
        """Get the CSV response from the Water Quality Web Service.

        Parameters
        ----------
        endpoint : str
            Endpoint of the Water Quality Web Service.
        kwds : dict
            Water Quality Web Service keyword arguments.
        request_method : str, optional
            HTTP request method. Default to GET.

        Returns
        -------
        pandas.DataFrame
            The web service response as a DataFrame.
        """
        req_kwds = [{"params": kwds}] if request_method == "GET" else [{"data": kwds}]
        r = ar.retrieve(
            [self._base_url(endpoint)], "binary", req_kwds, request_method=request_method
        )
        return pd.read_csv(io.BytesIO(r[0]), compression="zip")
