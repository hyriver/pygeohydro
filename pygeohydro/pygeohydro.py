"""Accessing data from the supported databases through their APIs."""
import io
import warnings
import zipfile
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from unittest.mock import patch

import async_retriever as ar
import cytoolz as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoutils as geoutils
import rasterio as rio
import xarray as xr
from pygeoogc import WMS, ArcGISRESTful, RetrySession, ServiceURL
from shapely.geometry import MultiPolygon, Polygon

from . import helpers
from .exceptions import (
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


class NID:
    """Retrieve data from the National Inventory of Dams web service.

    Parameters
    ----------
    expire_after : int, optional
        Expiration time for response caching in seconds, defaults to -1 (never expire).
    disable_caching : bool, optional
        If ``True``, disable caching requests, defaults to False.
    """

    def __init__(
        self,
        expire_after: float = EXPIRE,
        disable_caching: bool = False,
    ) -> None:
        self.base_url = ServiceURL().restful.nid
        self.suggest_url = f"{self.base_url}/suggestions"
        self.expire_after = expire_after
        self.disable_caching = disable_caching
        self.fields_meta = pd.DataFrame(
            ar.retrieve(
                [f"{self.base_url}/advanced-fields"],
                "json",
                expire_after=self.expire_after,
                disable=self.disable_caching,
            )[0]
        )
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

    def get_bygeom(self, geometry: int, geo_crs: str) -> gpd.GeoDataFrame:
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
        wbd = ArcGISRESTful(
            ServiceURL().restful.wbd,
            4,
            outformat="json",
            outfields="huc8",
            expire_after=self.expire_after,
            disable_caching=self.disable_caching,
        )
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
        dam_ids : list of int
            List of the target dam IDs. Note that the dam IDs are not the
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
        >>> dams = nid.inventory_byid(['514871', '459170', '514868', '463501', '463498'])
        >>> print(dams.damHeight.max())
        120.0
        """
        if not isinstance(dam_ids, list) and any(not isinstance(i, int) for i in dam_ids):
            raise InvalidInputType("nid_ids", "list of int")

        return self._to_geodf(
            pd.DataFrame(self._get_json([f"{self.base_url}/dams/{i}/inventory" for i in dam_ids]))
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
        ...    {"huc6": ["160502", "100500"], "drainageArea": ["[200 500]"]},
        ...    {"nidId": ["CA01222"]},
        ... ]
        >>> dam_dfs = nid.get_byfilter(query_list)
        >>> print(dams.name[0])
        Stillwater Point Dam
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
        >>> dams, contexts = nid.get_suggestions("texas", "huc2")
        >>> print(contexts.loc["HUC2", "value"])
        12
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

    def _get_json(
        self, urls: List[str], params: Optional[List[Dict[str, str]]] = None
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
        resp = ar.retrieve(
            urls,
            "json",
            kwds,
            expire_after=self.expire_after,
            disable=self.disable_caching,
        )
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

    def __repr__(self) -> str:
        """Print the services properties."""
        resp = self._get_json([f"{self.base_url}/metadata"])[0]
        return "\n".join(
            [
                "Connected to NID RESTful with the following properties:",
                f"URL: {self.base_url}",
                f"Date Refreshed: {resp['dateRefreshed']}",
                f"Version: {resp['version']}",
            ]
        )
