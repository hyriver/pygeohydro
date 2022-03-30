"""Accessing data from the supported databases through their APIs."""
import io
import itertools
import warnings
import zipfile
from collections import OrderedDict
from numbers import Number
from ssl import SSLContext
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from unittest.mock import patch

import async_retriever as ar
import cytoolz as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoutils as geoutils
import pyproj
import rasterio as rio
import xarray as xr
from pygeoogc import WMS, ArcGISRESTful, RetrySession, ServiceURL
from pygeoogc import utils as ogc_utils
from pynhd import AGRBase
from shapely.geometry import MultiPolygon, Polygon

from . import helpers
from .exceptions import (
    InvalidInputType,
    InvalidInputValue,
    MissingColumns,
    MissingCRS,
    ServiceUnavailable,
    ZeroMatched,
)
from .helpers import Stats, logger

DEF_CRS = "epsg:4326"
EXPIRE = -1
GTYPE = Union[Polygon, MultiPolygon, Tuple[float, float, float, float]]


__all__ = [
    "get_camels",
    "ssebopeta_bycoords",
    "ssebopeta_byloc",
    "ssebopeta_bygeom",
    "nlcd_bygeom",
    "nlcd_bycoords",
    "cover_statistics",
    "overland_roughness",
    "NID",
    "WBD",
]


def get_camels() -> Tuple[gpd.GeoDataFrame, xr.Dataset]:
    """Get streaflow and basin attributes of all 671 stations in CAMELS dataset.

    Notes
    -----
    For more info on CAMELS visit: https://ral.ucar.edu/solutions/products/camels

    Returns
    -------
    tuple of geopandas.GeoDataFrame and xarray.Dataset
        The first is basin attributes as a ``geopandas.GeoDataFrame`` and the second
        is streamflow data and basin attributes as an ``xarray.Dataset``.
    """
    base_url = "/".join(
        [
            "https://thredds.hydroshare.org/thredds/fileServer/hydroshare",
            "resources/658c359b8c83494aac0f58145b1b04e6/data/contents",
        ]
    )
    urls = [
        f"{base_url}/camels_attributes_v2.0.feather",
        f"{base_url}/camels_attrs_v2_streamflow_v1p2.nc",
    ]
    resp = ar.retrieve_binary(urls)

    attrs = gpd.read_feather(io.BytesIO(resp[0]))
    qobs = xr.open_dataset(io.BytesIO(resp[1]), engine="h5netcdf")
    return attrs, qobs


def ssebopeta_bycoords(
    coords: pd.DataFrame,
    dates: Union[Tuple[str, str], Union[int, List[int]]],
    crs: str = DEF_CRS,
) -> xr.Dataset:
    """Daily actual ET for a dataframe of coords from SSEBop database in mm/day.

    Parameters
    ----------
    coords : pandas.DataFrame
        A dataframe with ``id``, ``x``, ``y`` columns.
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    crs : str, optional
        The CRS of the input coordinates, defaults to epsg:4326.

    Returns
    -------
    xarray.Dataset
        Daily actual ET in mm/day as a dataset with ``time`` and ``location_id`` dimensions.
        The ``location_id`` dimension is the same as the ``id`` column in the input dataframe.
    """
    if not isinstance(coords, pd.DataFrame):
        raise InvalidInputType("coords", "pandas.DataFrame")

    req_cols = ["id", "x", "y"]
    if not set(req_cols).issubset(coords.columns):
        raise MissingColumns(req_cols)

    _coords = gpd.GeoSeries(
        gpd.points_from_xy(coords["x"], coords["y"]), index=coords["id"], crs=crs
    )
    _coords = _coords.to_crs(DEF_CRS)
    co_list = list(zip(_coords.x, _coords.y))

    f_list = helpers.get_ssebopeta_urls(dates)
    session = RetrySession()

    with patch("socket.has_ipv6", False):

        def _ssebop(url: str) -> List[np.ndarray]:  # type: ignore
            r = session.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))

            with rio.MemoryFile() as memfile:
                memfile.write(z.read(z.filelist[0].filename))
                with memfile.open() as src:
                    return list(src.sample(co_list))

        time, eta = zip(*[(t, _ssebop(url)) for t, url in f_list])
    eta_arr = np.array(eta).reshape(len(time), -1)  # type: ignore
    ds = xr.Dataset(
        data_vars={
            "eta": (["time", "location_id"], eta_arr),
            "x": (["location_id"], coords["x"].to_numpy()),
            "y": (["location_id"], coords["y"].to_numpy()),
        },
        coords={
            "time": np.array(time, dtype="datetime64[ns]"),
            "location_id": coords["id"].to_numpy(),
        },
    )
    ds["eta"] = ds["eta"].where(ds["eta"] != 9999, np.nan) * 1e-3
    ds.eta.attrs = {
        "units": "mm/day",
        "long_name": "Actual ET",
        "nodatavals": (np.nan,),
    }
    ds.x.attrs = {"crs": pyproj.CRS(crs).to_string()}
    ds.y.attrs = {"crs": pyproj.CRS(crs).to_string()}
    return ds


def ssebopeta_byloc(
    coords: Tuple[float, float],
    dates: Union[Tuple[str, str], Union[int, List[int]]],
) -> pd.Series:
    """Daily actual ET for a location from SSEBop database in mm/day.

    .. deprecated:: 0.11.5
        Use :func:`ssebopeta_bycoords` instead. For now, this function calls
        :func:`ssebopeta_bycoords` but retains the same functionality, i.e.,
        returns a dataframe and accepts only a single coordinate. Whereas the
        new function returns a ``xarray.Dataset`` and accepts a dataframe
        containing coordinates.

    Parameters
    ----------
    coords : tuple
        Longitude and latitude of a single location as a tuple (lon, lat)
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].

    Returns
    -------
    pandas.Series
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
    if not isinstance(coords, tuple) or len(coords) != 2:
        raise InvalidInputType("coords", "(lon, lat)")
    _coords = pd.DataFrame({"id": [0], "x": [coords[0]], "y": [coords[1]]})
    ds = ssebopeta_bycoords(_coords, dates)
    return ds.to_dataframe().pivot_table(index="time", columns="location_id", values="eta")[0]


def ssebopeta_bygeom(
    geometry: GTYPE,
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
    f_list = helpers.get_ssebopeta_urls(dates)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        gtiff2xarray = tlz.partial(geoutils.gtiff2xarray, geometry=geometry, geo_crs=geo_crs)
    else:
        gtiff2xarray = tlz.partial(geoutils.gtiff2xarray)

    session = RetrySession()

    with patch("socket.has_ipv6", False):

        def _ssebop(t: pd.Timestamp, url: str) -> xr.DataArray:
            resp = session.get(url)
            zfile = zipfile.ZipFile(io.BytesIO(resp.content))
            content = zfile.read(zfile.filelist[0].filename)
            ds: xr.DataArray = gtiff2xarray(r_dict={"eta": content})
            return ds.expand_dims({"time": [t]})

        data = xr.merge(_ssebop(t, url) for t, url in f_list)
    eta: xr.DataArray = data.where(data.eta < data.eta.nodatavals[0]).eta.copy() * 1e-3
    eta.attrs.update(
        {"units": "mm/day", "nodatavals": (np.nan,), "crs": DEF_CRS, "long_name": "Actual ET"}
    )
    return eta


class NLCD:
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
        ``epsg:4326``.
    ssl : bool or SSLContext, optional
        SSLContext to use for the connection, defaults to None. Set to ``False`` to disable
        SSL certification verification.
    """

    def __init__(
        self,
        years: Optional[Mapping[str, Union[int, List[int]]]] = None,
        region: str = "L48",
        crs: str = DEF_CRS,
        ssl: Union[SSLContext, bool, None] = None,
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
        self.region = region.upper()
        self.valid_crs = ogc_utils.valid_wms_crs(ServiceURL().wms.mrlc)
        self.crs = pyproj.CRS(crs).to_string().lower()
        if self.crs not in self.valid_crs:
            raise InvalidInputValue("crs", self.valid_crs)
        self.layers = self.get_layers()
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
            validation=False,
            ssl=ssl,
        )

    def get_layers(self) -> List[str]:
        """Get NLCD layers for the provided years dictionary."""
        valid_regions = ["L48", "HI", "PR", "AK"]
        if self.region not in valid_regions:
            raise InvalidInputValue("region", valid_regions)

        nlcd_meta = helpers.nlcd_helper()

        names = ["impervious", "cover", "canopy", "descriptor"]
        avail_years = {n: nlcd_meta[f"{n}_years"] for n in names}

        if any(
            yr not in avail_years[lyr] or lyr not in names
            for lyr, yrs in self.years.items()
            for yr in yrs
        ):
            vals = [f"\n{lyr}: {', '.join(str(y) for y in yr)}" for lyr, yr in avail_years.items()]
            raise InvalidInputValue("years", vals)

        def layer_name(lyr: str) -> str:
            if lyr == "canopy":
                return "Tree_Canopy"
            if lyr == "cover":
                return "Land_Cover_Science_Product"
            if lyr == "impervious":
                return "Impervious"
            return "Impervious_Descriptor" if self.region == "AK" else "Impervious_descriptor"

        return [
            f"NLCD_{yr}_{layer_name(lyr)}_{self.region}"
            for lyr, yrs in self.years.items()
            for yr in yrs
        ]

    def get_response(
        self, bbox: Tuple[float, float, float, float], resolution: float
    ) -> Dict[str, bytes]:
        """Get response from a url."""
        return self.wms.getmap_bybox(bbox, resolution, self.crs)

    def to_xarray(
        self, r_dict: Dict[str, bytes], geometry: Union[Polygon, MultiPolygon, None] = None
    ) -> xr.Dataset:
        """Convert response to xarray.DataArray."""
        if isinstance(geometry, (Polygon, MultiPolygon)):
            gtiff2xarray = tlz.partial(geoutils.gtiff2xarray, geometry=geometry, geo_crs=self.crs)
        else:
            gtiff2xarray = tlz.partial(geoutils.gtiff2xarray)

        try:
            _ds = gtiff2xarray(r_dict=r_dict)
        except rio.RasterioIOError as ex:
            raise ServiceUnavailable(self.wms.url) from ex

        ds: xr.Dataset = _ds.to_dataset() if isinstance(_ds, xr.DataArray) else _ds
        ds.attrs = _ds.attrs
        for lyr in self.layers:
            name = [n for n in self.units if n in lyr.lower()][-1]
            lyr_name = f"{name}_{lyr.split('_')[1]}"
            ds = ds.rename({lyr: lyr_name})
            ds[lyr_name].attrs["units"] = self.units[name]
            ds[lyr_name] = ds[lyr_name].astype(self.types[name])
            ds[lyr_name].attrs["nodatavals"] = (self.nodata[name],)
            ds[lyr_name] = ds[lyr_name].rio.write_nodata(self.nodata[name])
        return ds

    def __repr__(self) -> str:
        """Print the services properties."""
        return self.wms.__repr__()


def nlcd_bygeom(
    geometry: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    resolution: float,
    years: Optional[Mapping[str, Union[int, List[int]]]] = None,
    region: str = "L48",
    crs: str = DEF_CRS,
    ssl: Union[SSLContext, bool, None] = None,
) -> Dict[int, xr.Dataset]:
    """Get data from NLCD database (2019).

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with the geometry to query. The indices are used
        as keys in the output dictionary.
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
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    ssl : bool or SSLContext, optional
        SSLContext to use for the connection, defaults to None. Set to ``False`` to disable
        SSL certification verification.

    Returns
    -------
    dict of xarray.Dataset or xarray.Dataset
        A single or a ``dict`` of NLCD datasets. If dict, the keys are indices
        of the input ``GeoDataFrame``.
    """
    if resolution < 30:
        logger.warning("NLCD's resolution is 30 m, so finer resolutions are not recommended.")

    if not isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise InvalidInputType("geometry", "GeoDataFrame or GeoSeries")

    if geometry.crs is None:
        raise MissingCRS
    _geometry = geometry.to_crs(crs).geometry.to_dict()

    nlcd_wms = NLCD(years=years, region=region, crs=crs, ssl=ssl)

    ds = {
        i: nlcd_wms.to_xarray(nlcd_wms.get_response(g.bounds, resolution), g)
        for i, g in _geometry.items()
    }
    return ds


def nlcd_bycoords(
    coords: List[Tuple[float, float]],
    years: Optional[Mapping[str, Union[int, List[int]]]] = None,
    region: str = "L48",
    ssl: Union[SSLContext, bool, None] = None,
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
    ssl : bool or SSLContext, optional
        SSLContext to use for the connection, defaults to None. Set to ``False`` to disable
        SSL certification verification.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with the NLCD data and the coordinates.
    """
    if not isinstance(coords, list) or any(len(c) != 2 for c in coords):
        raise InvalidInputType("coords", "list of (lon, lat)")

    nlcd_wms = NLCD(years=years, region=region, crs="epsg:3857", ssl=ssl)
    points = gpd.GeoSeries(gpd.points_from_xy(*zip(*coords)), crs=DEF_CRS)
    points_proj = points.to_crs(nlcd_wms.crs)
    bounds = points_proj.buffer(50, cap_style=3)
    ds_list = [nlcd_wms.to_xarray(nlcd_wms.get_response(b.bounds, 30)) for b in bounds]

    def get_value(da: xr.DataArray, x: float, y: float) -> Number:
        nodata = da.attrs["nodatavals"][0]
        value = da.fillna(nodata).interp(x=[x], y=[y], method="nearest")
        return da.dtype.type(value)[0, 0]  # type: ignore

    values = {
        v: [get_value(ds[v], p.x, p.y) for ds, p in zip(ds_list, points_proj)] for v in ds_list[0]
    }
    return points.to_frame("geometry").merge(
        pd.DataFrame(values), left_index=True, right_index=True
    )


def overland_roughness(cover_da: xr.DataArray) -> xr.DataArray:
    """Estimate overland roughness from land cover data.

    Parameters
    ----------
    cover_da : xarray.DataArray
        Land cover DataArray from a LULC Dataset from the ``nlcd_bygeom`` function.

    Returns
    -------
    xarray.DataArray
        Overland roughness
    """
    if not isinstance(cover_da, xr.DataArray):
        raise InvalidInputType("cover_da", "xarray.DataArray")

    roughness = cover_da.astype(np.float64)
    roughness = roughness.rio.write_nodata(np.nan)
    roughness.name = "roughness"
    roughness.attrs["long_name"] = "overland roughness"
    roughness.attrs["units"] = "-"

    meta = helpers.nlcd_helper()
    get_roughness = np.vectorize(meta["roughness"].get, excluded=["default"])
    roughness.data = get_roughness(cover_da.astype(str), np.nan)
    return roughness


def cover_statistics(cover_da: xr.DataArray) -> Stats:
    """Percentages of the categorical NLCD cover data.

    Parameters
    ----------
    cover_da : xarray.DataArray
        Land cover DataArray from a LULC Dataset from the ``nlcd_bygeom`` function.

    Returns
    -------
    Stats
        A named tuple with the percentages of the cover classes and categories.
    """
    if not isinstance(cover_da, xr.DataArray):
        raise InvalidInputType("cover_da", "xarray.DataArray")

    nlcd_meta = helpers.nlcd_helper()
    val, freq = np.unique(cover_da, return_counts=True)  # type: ignore
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

    return Stats(class_percentage, category_percentage)


class NID:
    """Retrieve data from the National Inventory of Dams web service."""

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.nid
        self.suggest_url = f"{self.base_url}/suggestions"
        self.fields_meta = pd.DataFrame(ar.retrieve_json([f"{self.base_url}/advanced-fields"])[0])
        self.valid_fields = self.fields_meta.name.to_list()
        self.dam_type = {
            -1: "N/A",
            1: "Arch",
            2: "Buttress",
            3: "Concrete",
            4: "Earth",
            5: "Gravity",
            6: "Masonry",
            7: "Multi-Arch",
            8: "Rockfill",
            9: "Roller-Compacted Concrete",
            10: "Stone",
            11: "Timber Crib",
            12: "Other",
        }
        self.dam_purpose = {
            -1: "N/A",
            1: "Debris Control",
            2: "Fire Protection, Stock, Or Small Farm Pond",
            3: "Fish and Wildlife Pond",
            4: "Flood Risk Reduction",
            5: "Grade Stabilization",
            6: "Hydroelectric",
            7: "Irrigation",
            8: "Navigation",
            9: "Recreation",
            10: "Tailings",
            11: "Water Supply",
            12: "Other",
        }

    def _get_json(
        self, urls: Sequence[str], params: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Get JSON response from NID web service.

        Parameters
        ----------
        urls : list of str
            A list of query URLs.
        params : dict, optional
            A list of parameters to pass to the web service, defaults to ``None``.

        Returns
        -------
        list of dict
            List of JSON responses from the web service.
        """
        if not isinstance(urls, list):
            raise InvalidInputType("urls", "list or str")

        if params is None:
            kwds = None
        else:
            kwds = [{"params": {**p, "out": "json"}} for p in params]
        resp = ar.retrieve_json(urls, kwds)
        if len(resp) == 0:
            raise ZeroMatched

        failed = [(i, f"Req_{i}: {r['message']}") for i, r in enumerate(resp) if "error" in r]
        if failed:
            idx, err_msgs = zip(*failed)
            errs = " service requests failed with the following messages:\n"
            errs += "\n".join(err_msgs)
            if len(failed) == len(urls):
                raise ZeroMatched(f"All{errs}")
            resp = [r for i, r in enumerate(resp) if i not in idx]
            logger.warning(f"Some{errs}")
        return resp

    @staticmethod
    def _to_geodf(nid_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert a NID dataframe to a GeoDataFrame.

        Parameters
        ----------
        dams : pd.DataFrame
            NID dataframe

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of NID data
        """
        return gpd.GeoDataFrame(
            nid_df, geometry=gpd.points_from_xy(nid_df.longitude, nid_df.latitude), crs=DEF_CRS
        )

    def get_byfilter(self, query_list: List[Dict[str, List[str]]]) -> List[gpd.GeoDataFrame]:
        """Query dams by filters from the National Inventory of Dams web service.

        Parameters
        ----------
        query_list : list of dict
            List of dictionary of query parameters. For an exhaustive list of the parameters,
            use the advanced fields dataframe that can be accessed via ``NID().fields_meta``.
            Some filter require min/max values such as ``damHeight`` and ``drainageArea``.
            For such filters, the min/max values should be passed like so:
            ``{filter_key: ["[min1 max1]", "[min2 max2]"]}``.

        Returns
        -------
        geopandas.GeoDataFrame
            Query results.

        Examples
        --------
        >>> from pygeohydro import NID
        >>> nid = NID()
        >>> query_list = [
        ...    {"drainageArea": ["[200 500]"]},
        ...    {"nidId": ["CA01222"]},
        ... ]
        >>> dam_dfs = nid.get_byfilter(query_list)
        >>> print(dam_dfs[0].name[0])
        Prairie Portage
        """
        invalid = [k for key in query_list for k in key if k not in self.valid_fields]
        if invalid:
            raise InvalidInputValue("query_dict", self.valid_fields)
        params = [
            {"sy": " ".join(f"@{s}:{fid}" for s, fids in key.items() for fid in fids)}
            for key in query_list
        ]
        return [
            self._to_geodf(pd.DataFrame(r))
            for r in self._get_json([f"{self.base_url}/query"] * len(params), params)
        ]

    def get_bygeom(self, geometry: GTYPE, geo_crs: str) -> gpd.GeoDataFrame:
        """Retrieve NID data within a geometry.

        Parameters
        ----------
        geometry : Polygon, MultiPolygon, or tuple of length 4
            Geometry or bounding box (west, south, east, north) for extracting the data.
        geo_crs : list of str
            The CRS of the input geometry, defaults to epsg:4326.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of NID data

        Examples
        --------
        >>> from pygeohydro import NID
        >>> nid = NID()
        >>> dams = nid.get_bygeom((-69.77, 45.07, -69.31, 45.45), "epsg:4326")
        >>> print(dams.name.iloc[0])
        Little Moose
        """
        _geometry = geoutils.geo2polygon(geometry, geo_crs, DEF_CRS)
        wbd = ArcGISRESTful(ServiceURL().restful.wbd, 4, outformat="json", outfields="huc8")
        resp = wbd.get_features(wbd.oids_bygeom(_geometry), return_geom=False)
        huc_ids = [
            tlz.get_in(["attributes", "huc8"], i) for r in resp for i in tlz.get_in(["features"], r)
        ]

        dams = self.get_byfilter([{"huc8": huc_ids}])[0]
        return dams[dams.within(_geometry)].copy()

    def inventory_byid(self, dam_ids: List[int]) -> gpd.GeoDataFrame:
        """Get extra attributes for dams based on their dam ID.

        Notes
        -----
        This function is meant to be used for getting extra attributes for dams.
        For example, first you need to use either ``get_bygeom`` or ``get_byfilter``
        to get basic attributes of the target dams. Then you can use this function
        to get extra attributes using the ``id`` column of the ``GeoDataFrame``
        that ``get_bygeom`` or ``get_byfilter`` returns.

        Parameters
        ----------
        dam_ids : list of int or str
            List of the target dam IDs (digists only). Note that the dam IDs are not the
            same as the NID IDs.

        Returns
        -------
        pandas.DataFrame
            Dams with extra attributes in addition to the standard NID fields
            that other ``NID`` methods return.

        Examples
        --------
        >>> from pygeohydro import NID
        >>> nid = NID()
        >>> dams = nid.inventory_byid([514871, 459170, 514868, 463501, 463498])
        >>> print(dams.damHeight.max())
        120.0
        """
        urls = [
            f"{self.base_url}/dams/{i}/inventory"
            for i in itertools.takewhile(lambda x: str(x).isdigit(), dam_ids)
        ]
        if len(urls) != len(dam_ids):
            raise InvalidInputType("dam_ids", "list of digits")

        return self._to_geodf(pd.DataFrame(self._get_json(urls)))

    def get_suggestions(
        self, text: str, context_key: str = ""
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get suggestions from the National Inventory of Dams web service.

        Notes
        -----
        This function is useful for exploring and/or narrowing down the filter fields
        that are needed to query the dams using ``get_byfilter``.

        Parameters
        ----------
        text : str
            Text to query for suggestions.
        context_key : str, optional
            Suggestion context, defaults to empty string, i.e., all context keys.
            For a list of valid context keys, see ``NID().fields_meta``.

        Returns
        -------
        tuple of pandas.DataFrame
            The suggestions for the requested text as two DataFrames:
            First, is suggestions found in the dams properties and
            second, those found in the query fields such as states, huc6, etc.

        Examples
        --------
        >>> from pygeohydro import NID
        >>> nid = NID()
        >>> dams, contexts = nid.get_suggestions("texas", "city")
        >>> print(contexts.loc["CITY", "value"])
        Texas City
        """
        if len(context_key) > 0 and context_key not in self.valid_fields:
            raise InvalidInputValue("context", self.valid_fields)

        params = [{"text": text, "contextKey": context_key}]
        resp = self._get_json([f"{self.base_url}/suggestions"] * len(params), params)
        dams = pd.DataFrame(resp[0]["dams"])
        contexts = pd.DataFrame(resp[0]["contexts"])
        return (
            dams if dams.empty else dams.set_index("id"),
            contexts if contexts.empty else contexts.set_index("name"),
        )

    def __repr__(self) -> str:
        """Print the services properties."""
        resp = self._get_json([f"{self.base_url}/metadata"])[0]
        return "\n".join(
            [
                "NID RESTful information:",
                f"URL: {self.base_url}",
                f"Date Refreshed: {resp['dateRefreshed']}",
                f"Version: {resp['version']}",
            ]
        )


class WBD(AGRBase):
    """Access Watershed Boundary Dataset (WBD).

    Notes
    -----
    This file contains Hydrologic Unit (HU) polygon boundaries for the United States,
    Puerto Rico, and the U.S. Virgin Islands.
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Valid layers are:

        - ``wbdline``
        - ``huc2``
        - ``huc4``
        - ``huc6``
        - ``huc8``
        - ``huc10``
        - ``huc12``
        - ``huc14``
        - ``huc16``

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, optional
        Target spatial reference, default to ``EPSG:4326``.
    """

    def __init__(self, layer: str, outfields: Union[str, List[str]] = "*", crs: str = DEF_CRS):
        self.valid_layers = {
            "wbdline": "wbdline",
            "huc2": "2-digit hu (region)",
            "huc4": "4-digit hu (subregion)",
            "huc6": "6-digit hu (basin)",
            "huc8": "8-digit hu  (subbasin)",
            "huc10": "10-digit hu (watershed)",
            "huc12": "12-digit hu (subwatershed)",
            "huc14": "14-digit hu",
            "huc16": "16-digit hu",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InvalidInputValue("layer", list(self.valid_layers))
        super().__init__(ServiceURL().restful.wbd, _layer, outfields, crs)
