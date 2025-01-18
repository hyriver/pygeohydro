"""Accessing data from the supported databases through their APIs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union, cast

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import xarray as xr

import pygeoutils as geoutils
from pygeohydro import helpers
from pygeohydro.exceptions import (
    InputTypeError,
    InputValueError,
    MissingCRSError,
    ServiceUnavailableError,
)
from pygeohydro.helpers import Stats
from pygeoogc import WMS, ServiceURL
from pygeoogc import utils as ogc_utils

if TYPE_CHECKING:
    from collections.abc import Mapping
    from numbers import Number

    from pyproj import CRS
    from shapely import MultiPolygon, Polygon

    GTYPE = Union[Polygon, MultiPolygon, tuple[float, float, float, float]]

    CRSType = int | str | CRS

__all__ = [
    "cover_statistics",
    "nlcd_area_percent",
    "nlcd_bycoords",
    "nlcd_bygeom",
    "overland_roughness",
]


class NLCD:
    """Get data from NLCD database (2021).

    Parameters
    ----------
    years : dict, optional
        The years for NLCD layers as a dictionary, defaults to
        ``{'impervious': [2021], 'cover': [2021], 'canopy': [2021], "descriptor": [2021]}``.
        Layers that are not in years are ignored, e.g., ``{'cover': [2016, 2021]}`` returns
        land cover data for 2016 and 2021.
    region : str, optional
        Region in the US that the input geometries are located, defaults to ``L48``.
        Valid values are ``L48`` (for CONUS), ``HI`` (for Hawaii), ``AK`` (for Alaska),
        and ``PR`` (for Puerto Rico). Both lower and upper cases are acceptable.
    crs : str, int, or pyproj.CRS, optional
        The spatial reference system to be used for requesting the data, defaults to
        ``epsg:4326``.
    ssl : bool, optional
        Whether to use SSL for the connection, defaults to ``True``.
    """

    def __init__(
        self,
        years: Mapping[str, int | list[int]] | None = None,
        region: str = "L48",
        crs: CRSType = 4326,
        ssl: bool = True,
    ) -> None:
        default_years = {
            "impervious": [2021],
            "cover": [2021],
            "canopy": [2021],
            "descriptor": [2021],
        }
        years = default_years if years is None else years
        if not isinstance(years, dict):
            raise InputTypeError("years", "dict", str(default_years))
        self.years = tlz.valmap(lambda x: x if isinstance(x, list) else [x], years)
        self.years = cast("dict[str, list[int]]", self.years)
        self.region = region.upper()
        base_url = ServiceURL().wms.mrlc
        self.valid_crs = ogc_utils.valid_wms_crs(base_url)
        self.crs = pyproj.CRS(crs).to_string().lower()
        if self.crs not in self.valid_crs:
            raise InputValueError("crs", self.valid_crs)
        self.layers = self.get_layers()
        self.units = {"impervious": "%", "cover": "classes", "canopy": "%", "descriptor": "classes"}
        self.types = {"impervious": "f4", "cover": "u1", "canopy": "f4", "descriptor": "u1"}
        self.nodata = {"impervious": np.nan, "cover": 127, "canopy": np.nan, "descriptor": 127}

        self.wms = WMS(
            base_url,
            layers=list(self.layers.values()),
            outformat="image/geotiff",
            crs=self.crs,
            validation=False,
            ssl=ssl,
        )

    def get_layers(self) -> dict[str, str]:
        """Get NLCD layers for the provided years dictionary."""
        valid_regions = ("L48", "HI", "PR", "AK")
        if self.region not in valid_regions:
            raise InputValueError("region", valid_regions)

        nlcd_meta = helpers.nlcd_helper()

        names = ["impervious", "cover", "canopy", "descriptor"]
        avail_years = {n: nlcd_meta[f"{n}_years"] for n in names}

        if any(
            yr not in avail_years[lyr] or lyr not in names
            for lyr, yrs in self.years.items()
            for yr in yrs
        ):
            vals = [f"\n{lyr}: {', '.join(str(y) for y in yr)}" for lyr, yr in avail_years.items()]
            raise InputValueError("years", vals)

        def layer_name(lyr: str, yr: int) -> str:
            if lyr == "canopy":
                if self.region == "L48":
                    return f"nlcd_tcc_conus_{yr}_v2021-4"
                return f"NLCD_{yr}_Tree_Canopy_{self.region}"
            if lyr == "cover":
                return f"NLCD_{yr}_Land_Cover_{self.region}"
            if lyr == "impervious":
                return f"NLCD_{yr}_Impervious_{self.region}"
            if self.region in ("HI", "PR"):
                raise InputValueError("region (descriptor)", ("L48", "AK"))
            service_lyr = (
                "Impervious_Descriptor" if self.region == "AK" else "Impervious_descriptor"
            )
            return f"NLCD_{yr}_{service_lyr}_{self.region}"

        return {f"{lyr}_{yr}": layer_name(lyr, yr) for lyr, yrs in self.years.items() for yr in yrs}

    def get_map(
        self,
        geometry: Polygon | MultiPolygon,
        resolution: int,
    ) -> xr.Dataset:
        """Get NLCD response and convert it to ``xarray.DataArray``."""
        r_dict = self.wms.getmap_bybox(geometry.bounds, resolution, self.crs)
        gtiff2xarray = tlz.partial(
            geoutils.gtiff2xarray, geometry=geometry, geo_crs=self.crs, nodata=255
        )
        try:
            _ds = gtiff2xarray(r_dict=r_dict)
        except rio.RasterioIOError as ex:
            raise ServiceUnavailableError(self.wms.url) from ex

        ds = _ds.to_dataset() if isinstance(_ds, xr.DataArray) else _ds
        ds.attrs = _ds.attrs
        for lyr_name, lyr in self.layers.items():
            name = lyr_name.split("_")[0]
            ds = ds.rename({lyr: lyr_name})
            ds[lyr_name] = ds[lyr_name].where(ds[lyr_name] < 255, self.nodata[name])
            ds[lyr_name].attrs["units"] = self.units[name]
            ds[lyr_name] = ds[lyr_name].astype(self.types[name])
            ds[lyr_name].attrs["nodatavals"] = (self.nodata[name],)
            ds[lyr_name] = ds[lyr_name].rio.write_nodata(self.nodata[name])
        return ds

    def __repr__(self) -> str:
        """Print the services properties."""
        return self.wms.__repr__()


def nlcd_bygeom(
    geometry: gpd.GeoSeries | gpd.GeoDataFrame,
    resolution: int = 30,
    years: Mapping[str, int | list[int]] | None = None,
    region: str = "L48",
    crs: CRSType = 4326,
    ssl: bool = True,
) -> dict[int | str, xr.Dataset]:
    """Get data from NLCD database (2019).

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with the geometry to query. The indices are used
        as keys in the output dictionary.
    resolution : float, optional
        The data resolution in meters. The width and height of the output are computed in pixel
        based on the geometry bounds and the given resolution. The default is 30 m which is the
        native resolution of NLCD data.
    years : dict, optional
        The years for NLCD layers as a dictionary, defaults to
        ``{'impervious': [2019], 'cover': [2019], 'canopy': [2019], "descriptor": [2019]}``.
        Layers that are not in years are ignored, e.g., ``{'cover': [2016, 2019]}`` returns
        land cover data for 2016 and 2019.
    region : str, optional
        Region in the US that the input geometries are located, defaults to ``L48``.
        Valid values are ``L48`` (for CONUS), ``HI`` (for Hawaii), ``AK`` (for Alaska),
        and ``PR`` (for Puerto Rico). Both lower and upper cases are acceptable.
    crs : str, int, or pyproj.CRS, optional
        The spatial reference system to be used for requesting the data, defaults to
        ``epsg:4326``.
    ssl : bool, optional
        Whether to use SSL for the connection, defaults to ``True``.

    Returns
    -------
    dict of xarray.Dataset or xarray.Dataset
        A single or a ``dict`` of NLCD datasets. If dict, the keys are indices
        of the input ``GeoDataFrame``.
    """
    if resolution < 30:
        warnings.warn(
            "NLCD's resolution is 30 m, so finer resolutions are not recommended.",
            UserWarning,
            stacklevel=2,
        )

    if not isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise InputTypeError("geometry", "GeoDataFrame or GeoSeries")

    if geometry.crs is None:
        raise MissingCRSError
    _geometry = cast("gpd.GeoDataFrame", geometry.to_crs(crs))
    geo_dict = _geometry.geometry.to_dict()

    nlcd_wms = NLCD(years=years, region=region, crs=crs, ssl=ssl)

    return {i: nlcd_wms.get_map(g, resolution) for i, g in geo_dict.items()}


def nlcd_bycoords(
    coords: list[tuple[float, float]],
    years: Mapping[str, int | list[int]] | None = None,
    region: str = "L48",
    ssl: bool = True,
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
        Region in the US that the input geometries are located, defaults to ``L48``.
        Valid values are ``L48`` (for CONUS), ``HI`` (for Hawaii), ``AK`` (for Alaska),
        and ``PR`` (for Puerto Rico). Both lower and upper cases are acceptable.
    ssl : bool, optional
        Whether to use SSL for the connection, defaults to ``True``.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with the NLCD data and the coordinates.
    """
    nlcd_wms = NLCD(years=years, region=region, crs=3857, ssl=ssl)
    points = gpd.GeoSeries(gpd.points_from_xy(*zip(*coords), crs=4326))
    points_proj = points.to_crs(nlcd_wms.crs)
    geoms = points_proj.buffer(50, cap_style="square")
    ds_list = [nlcd_wms.get_map(g, 30) for g in geoms]

    def get_value(da: xr.DataArray, x: float, y: float) -> Number:
        nodata = da.attrs["nodatavals"][0]
        value = da.fillna(nodata).interp(x=[x], y=[y], method="nearest")
        return da.dtype.type(value)[0, 0]

    values = {
        v: [get_value(ds[v], p.x, p.y) for ds, p in zip(ds_list, points_proj)] for v in ds_list[0]
    }
    points = cast("gpd.GeoDataFrame", points.to_frame("geometry"))
    return gpd.GeoDataFrame(
        pd.merge(points, pd.DataFrame(values), left_index=True, right_index=True)
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
        raise InputTypeError("cover_da", "xarray.DataArray")

    roughness = cover_da.astype(np.float64)
    roughness = roughness.rio.write_nodata(np.nan)
    roughness.name = "roughness"
    roughness.attrs["long_name"] = "overland roughness"
    roughness.attrs["units"] = "-"

    meta = helpers.nlcd_helper()
    get_roughness = np.vectorize(meta["roughness"].get, excluded=["default"])
    return roughness.copy(data=get_roughness(cover_da.astype("uint8").astype(str), np.nan))


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
        raise InputTypeError("cover_da", "xarray.DataArray")

    nlcd_meta = helpers.nlcd_helper()
    val, freq = np.unique(cover_da, return_counts=True)
    zero_idx = np.argwhere(val == 127)
    val = np.delete(val, zero_idx).astype(str)
    freq = np.delete(freq, zero_idx)
    freq_dict = dict(zip(val.tolist(), freq.tolist()))
    total_count = freq.sum()

    if any(c not in nlcd_meta["classes"] for c in freq_dict):
        raise InputValueError("ds", list(nlcd_meta["classes"]))

    class_percentage = {
        nlcd_meta["classes"][k].split(" -")[0].strip(): v / total_count * 100.0
        for k, v in freq_dict.items()
    }
    category_percentage = {
        k: sum(freq_dict[c] for c in v if c in freq_dict) / total_count * 100.0
        for k, v in nlcd_meta["categories"].items()
        if k != "Background"
    }

    return Stats(class_percentage, category_percentage)


def _area_percent(nlcd: xr.Dataset, year: int) -> dict[str, float]:
    """Calculate the percentage of the area for each land cover class."""
    cover_nodata = nlcd[f"cover_{year}"].rio.nodata
    if np.isnan(cover_nodata):
        msk = ~nlcd[f"cover_{year}"].isnull()
    elif cover_nodata > 0:
        msk = nlcd[f"cover_{year}"] < cover_nodata
    else:
        msk = nlcd[f"cover_{year}"] > cover_nodata
    cell_total = msk.sum()

    msk = nlcd[f"cover_{year}"].isin(range(21, 25))
    urban = msk.sum() / cell_total
    natural = 1 - urban

    impervious = nlcd.where(msk)[f"impervious_{year}"].mean() * urban / 100
    developed = urban - impervious
    natural = natural.compute().item() * 100
    developed = developed.compute().item() * 100
    impervious = impervious.compute().item() * 100
    return {
        "natural": natural,
        "developed": developed,
        "impervious": impervious,
        "urban": developed + impervious,
    }


def nlcd_area_percent(
    geo_df: gpd.GeoSeries | gpd.GeoDataFrame,
    year: int = 2019,
    region: str = "L48",
) -> pd.DataFrame:
    """Compute the area percentages of the natural, developed, and impervious areas.

    Notes
    -----
    This function uses imperviousness and land use/land cover data from NLCD
    to compute the area percentages of the natural, developed, and impervious areas.
    It considers land cover classes of 21 to 24 as urban and the rest as natural.
    Then, uses imperviousness percentage to partition the urban area into developed
    and impervious areas. So, ``urban = developed + impervious`` and always
    ``natural + urban = natural + developed + impervious = 100``.

    Parameters
    ----------
    geo_df : geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with the geometry to query. The indices are used
        as keys in the output dictionary.
    year : int, optional
        Year of the NLCD data, defaults to 2019. Available years are 2021, 2019, 2016,
        2013, 2011, 2008, 2006, 2004, and 2001.
    region : str, optional
        Region in the US that the input geometries are located, defaults to ``L48``.
        Valid values are ``L48`` (for CONUS), ``HI`` (for Hawaii), ``AK`` (for Alaska),
        and ``PR`` (for Puerto Rico). Both lower and upper cases are acceptable.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the same index as input ``geo_df`` and columns are the area
        percentages of the natural, developed, impervious, and urban
        (sum of developed and impervious) areas. Sum of urban and natural percentages
        is always 100, as well as the sum of natural, developed, and impervious
        percentages.
    """
    valid_year = (2021, 2019, 2016, 2013, 2011, 2008, 2006, 2004, 2001)
    if year not in valid_year:
        raise InputValueError("year", valid_year)

    if not isinstance(geo_df, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise InputTypeError("geometry", "GeoDataFrame or GeoSeries")

    if geo_df.crs is None:
        raise MissingCRSError

    geoms = geo_df.to_crs(4326).geometry  # pyright: ignore[reportOptionalMemberAccess]

    wms = NLCD(years={"impervious": year, "cover": year}, region=region, ssl=False)

    return pd.DataFrame.from_dict(
        {i: _area_percent(wms.get_map(g, 30), year) for i, g in geoms.items()},
        orient="index",
    )
