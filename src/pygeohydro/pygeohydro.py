"""Accessing data from the supported databases through their APIs."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import io
import itertools
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union, cast
from unittest.mock import patch
from zipfile import ZipFile

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import rasterio as rio
import xarray as xr
from rasterio.io import MemoryFile
from rioxarray import _io as rxr
from shapely.errors import GEOSException

import async_retriever as ar
import pygeoogc as ogc
import pygeoutils as geoutils
from pygeohydro import helpers
from pygeohydro.exceptions import (
    DependencyError,
    InputTypeError,
    InputValueError,
    MissingColumnError,
    ServiceError,
    ZeroMatchedError,
)
from pygeoogc import RetrySession, ServiceURL
from pygeoutils.exceptions import EmptyResponseError
from pynhd.core import AGRBase, ScienceBase

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyproj import CRS
    from shapely import MultiPolygon, Polygon

    CRSType = int | str | CRS

    GTYPE = Union[Polygon, MultiPolygon, tuple[float, float, float, float]]

__all__ = [
    "EHydro",
    "get_camels",
    "soil_gnatsgo",
    "soil_polaris",
    "soil_properties",
    "soil_soilgrids",
    "ssebopeta_bycoords",
    "ssebopeta_bygeom",
]


SG_ATTRS = {
    "bdod": {
        "description": "Bulk density of the fine earth fraction",
        "long_name": "Bulk Density",
        "mapped_units": "cg/cm³",
        "conversion_factor": 100,
        "conventional_units": "kg/dm³",
    },
    "cec": {
        "description": "Cation Exchange Capacity of the soil",
        "long_name": "Cation Exchange Capacity",
        "mapped_units": "mmol(c)/kg",
        "conversion_factor": 10,
        "conventional_units": "cmol(c)/kg",
    },
    "cfvo": {
        "description": "Volumetric fraction of coarse fragments (> 2 mm)",
        "long_name": "Coarse Fragments Vol",
        "mapped_units": "cm3/dm3 (vol‰)",
        "conversion_factor": 10,
        "conventional_units": "cm3/100cm3 (vol%)",
    },
    "clay": {
        "description": "Proportion of clay particles (< 0.002 mm) in the fine earth fraction",
        "long_name": "Clay Content",
        "mapped_units": "g/kg",
        "conversion_factor": 10,
        "conventional_units": "g/100g (%)",
    },
    "nitrogen": {
        "description": "Total nitrogen (N)",
        "long_name": "Nitrogen Content",
        "mapped_units": "cg/kg",
        "conversion_factor": 100,
        "conventional_units": "g/kg",
    },
    "phh2o": {
        "description": "Soil pH",
        "long_name": "Ph In H2O",
        "mapped_units": "pHx10",
        "conversion_factor": 10,
        "conventional_units": "pH",
    },
    "sand": {
        "description": "Proportion of sand particles (> 0.05 mm) in the fine earth fraction",
        "long_name": "Sand Content",
        "mapped_units": "g/kg",
        "conversion_factor": 10,
        "conventional_units": "g/100g (%)",
    },
    "silt": {
        "description": "Proportion of silt particles (≥ 0.002 mm and ≤ 0.05 mm) in the fine earth fraction",
        "long_name": "Silt Content",
        "mapped_units": "g/kg",
        "conversion_factor": 10,
        "conventional_units": "g/100g (%)",
    },
    "soc": {
        "description": "Soil organic carbon content in the fine earth fraction",
        "long_name": "Soil Organic Carbon",
        "mapped_units": "dg/kg",
        "conversion_factor": 10,
        "conventional_units": "g/kg",
    },
    "ocd": {
        "description": "Organic carbon density",
        "long_name": "Organic Carbon Density",
        "mapped_units": "hg/m³",
        "conversion_factor": 10,
        "conventional_units": "kg/m³",
    },
    "ocs": {
        "description": "Organic carbon stocks",
        "long_name": "Organic Carbon Stock",
        "mapped_units": "t/ha",
        "conversion_factor": 10,
        "conventional_units": "kg/m²",
    },
}

PS_ATTRS = {
    "silt": {
        "description": "silt percentage",
        "long_name": "Silt Content Percentage",
        "units": "%",
    },
    "sand": {
        "description": "sand percentage",
        "long_name": "Sand Content Percentage",
        "units": "%",
    },
    "clay": {
        "description": "clay percentage",
        "long_name": "Clay Content Percentage",
        "units": "%",
    },
    "bd": {
        "description": "bulk density",
        "long_name": "Bulk Density of Soil",
        "units": "g/cm3",
    },
    "theta_s": {
        "description": "saturated soil water content",
        "long_name": "Saturated Soil Water Content",
        "units": "m3/m3",
    },
    "theta_r": {
        "description": "residual soil water content",
        "long_name": "Residual Soil Water Content",
        "units": "m3/m3",
    },
    "ksat": {
        "description": "saturated hydraulic conductivity",
        "long_name": "Saturated Hydraulic Conductivity",
        "units": "log10(cm/hr)",
    },
    "ph": {
        "description": "soil pH in H2O",
        "long_name": "Soil pH in Water",
        "units": "-",
    },
    "om": {
        "description": "organic matter",
        "long_name": "Organic Matter Content",
        "units": "log10(%)",
    },
    "lambda": {
        "description": "pore size distribution index (Brooks-Corey)",
        "long_name": "Brooks-Corey Pore Size Distribution Index",
        "units": "-",
    },
    "hb": {
        "description": "bubbling pressure (Brooks-Corey)",
        "long_name": "Brooks-Corey Bubbling Pressure",
        "units": "log10(kPa)",
    },
    "n": {
        "description": "measure of the pore size distribution (Van Genuchten)",
        "long_name": "Van Genuchten Pore Size Distribution Parameter",
        "units": "-",
    },
    "alpha": {
        "description": "scale parameter inversely proportional to mean pore diameter (Van Genuchten)",
        "long_name": "Van Genuchten Scale Parameter",
        "units": "log10(kPa-1)",
    },
}


def get_camels() -> tuple[gpd.GeoDataFrame, xr.Dataset]:
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
    base_url = "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents"
    urls = [
        f"{base_url}/camels_attributes_v2.0.feather",
        f"{base_url}/camels_attrs_v2_streamflow_v1p2.nc",
    ]
    resp = ar.retrieve_binary(urls)

    attrs = gpd.read_feather(io.BytesIO(resp[0]))
    qobs = xr.open_dataset(io.BytesIO(resp[1]), engine="h5netcdf")
    qobs["discharge"] = xr.where(qobs["discharge"] < 0, np.nan, qobs["discharge"])

    url = "/".join(
        (
            "https://gist.githubusercontent.com/cheginit",
            "229c83c89eee3801a586bcb3ebb4e825/raw/newman_ids.txt",
        )
    )
    ids = ar.retrieve_text([url])[0].split(",")
    attrs["Newman_2017"] = attrs.index.isin(ids)
    qobs["Newman_2017"] = qobs.station_id.isin(ids)
    desc = ". ".join(
        (
            "Whether station is in Newman et al. (2017) dataset",
            "https://doi.org/10.1175/JHM-D-16-0284.1",
        )
    )
    qobs["Newman_2017"].attrs["description"] = desc
    return attrs, qobs


def ssebopeta_bycoords(
    coords: pd.DataFrame,
    dates: tuple[str, str] | int | list[int],
    crs: CRSType = 4326,
) -> xr.Dataset:
    """Daily actual ET for a dataframe of coords from SSEBop database in mm/day.

    Parameters
    ----------
    coords : pandas.DataFrame
        A dataframe with ``id``, ``x``, ``y`` columns.
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    crs : str, int, or pyproj.CRS, optional
        The CRS of the input coordinates, defaults to ``epsg:4326``.

    Returns
    -------
    xarray.Dataset
        Daily actual ET in mm/day as a dataset with ``time`` and ``location_id`` dimensions.
        The ``location_id`` dimension is the same as the ``id`` column in the input dataframe.
    """
    if not isinstance(coords, pd.DataFrame):
        raise InputTypeError("coords", "pandas.DataFrame")

    req_cols = ["id", "x", "y"]
    if not set(req_cols).issubset(coords.columns):
        raise MissingColumnError(req_cols)

    _coords = gpd.GeoSeries(
        gpd.points_from_xy(coords["x"], coords["y"], crs=crs), index=coords["id"]
    ).to_crs(4326)
    co_list = list(zip(_coords.x, _coords.y))

    f_list = helpers.get_ssebopeta_urls(dates)
    with patch("socket.has_ipv6", False), RetrySession() as session:

        def _ssebop(url: str) -> list[npt.NDArray[np.float64]]:
            r = session.get(url)
            with ZipFile(io.BytesIO(r.content)) as z, MemoryFile() as memfile:
                memfile.write(z.read(z.filelist[0].filename))
                with memfile.open() as src:
                    return list(src.sample(co_list))

        time, eta = zip(*[(t, _ssebop(url)) for t, url in f_list])
    eta_arr = np.array(eta).reshape(len(time), -1)
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


def ssebopeta_bygeom(
    geometry: GTYPE,
    dates: tuple[str, str] | int | list[int],
    geo_crs: CRSType = 4326,
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
    geometry : shapely.Polygon or tuple
        The geometry for downloading clipping the data. For a tuple bbox,
        the order should be (west, south, east, north).
    dates : tuple or list, optional
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    geo_crs : str, int, or pyproj.CRS, optional
        The CRS of the input geometry, defaults to ``epsg:4326``.

    Returns
    -------
    xarray.DataArray
        Daily actual ET within a geometry in mm/day at 1 km resolution
    """
    f_list = helpers.get_ssebopeta_urls(dates)

    try:
        _ = geoutils.geo2polygon(geometry)
    except geoutils.exceptions.InputTypeError as ex:
        raise InputTypeError("geometry", "(Multi)Polygon or tuple of length 4") from ex

    gtiff2xarray = functools.partial(geoutils.gtiff2xarray, geometry=geometry, geo_crs=geo_crs)
    with patch("socket.has_ipv6", False), RetrySession() as session:

        def _ssebop(t: pd.Timestamp, url: str) -> xr.DataArray | xr.Dataset:
            resp = session.get(url)
            with ZipFile(io.BytesIO(resp.content)) as zfile:
                content = zfile.read(zfile.filelist[0].filename)
                return gtiff2xarray(r_dict={"eta": content}).expand_dims({"time": [t]})

        data = xr.merge(itertools.starmap(_ssebop, f_list))
    eta = data.eta.where(data.eta < data.eta.rio.nodata) * 1e-3
    eta = cast("xr.DataArray", eta)
    eta.attrs.update(
        {
            "units": "mm/day",
            "nodatavals": (np.nan,),
            "crs": 4326,
            "long_name": "Actual ET",
        }
    )
    return eta


def soil_properties(
    properties: list[str] | str = "*", soil_dir: str | Path = "cache"
) -> xr.Dataset:
    """Get soil properties dataset in the United States from ScienceBase.

    Notes
    -----
    This function downloads the source zip files from
    `ScienceBase <https://www.sciencebase.gov/catalog/item/5fd7c19cd34e30b9123cb51f>`__
    , extracts the included ``.tif`` files, and return them as an ``xarray.Dataset``.

    Parameters
    ----------
    properties : list of str or str, optional
        Soil properties to extract, default to "*", i.e., all the properties.
        Available properties are ``awc`` for available water capacity, ``fc`` for
        field capacity, and ``por`` for porosity.
    soil_dir : str or pathlib.Pathlib.Path
        Directory to store zip files or if exists read from them, defaults to
        ``./cache``.
    """
    valid_props = {
        "awc": {"name": "awc", "units": "mm/m", "long_name": "Available Water Capacity"},
        "fc": {"name": "fc", "units": "mm/m", "long_name": "Field Capacity"},
        "por": {"name": "porosity", "units": "mm/m", "long_name": "Porosity"},
    }
    if properties == "*":
        prop = list(valid_props)
    else:
        prop = [properties] if isinstance(properties, str) else properties
        if not all(p in valid_props for p in prop):
            raise InputValueError("properties", list(valid_props))

    soil = ScienceBase.get_file_urls("5fd7c19cd34e30b9123cb51f")
    pat = "|".join(prop)
    _files, urls = zip(*soil[soil.index.str.contains(f"{pat}.*zip")].url.to_dict().items())
    urls = list(urls)

    root = Path(soil_dir)
    root.mkdir(exist_ok=True, parents=True)
    files = [root.joinpath(f) for f in _files]
    files = ogc.streaming_download(urls, fnames=files)
    files = [f for f in files if f is not None]

    def get_tif(file: Path) -> xr.DataArray:
        """Get the .tif file from a zip file."""
        with ZipFile(file) as z:
            try:
                fname = next(f.filename for f in z.filelist if f.filename.endswith(".tif"))
            except StopIteration as ex:
                raise ZeroMatchedError from ex
            ds = rxr.open_rasterio(io.BytesIO(z.read(fname)))
            ds = cast("xr.DataArray", ds)
            ds = ds.squeeze("band", drop=True)
            ds.name = valid_props[file.stem.split("_")[0]]["name"]
            ds.attrs["units"] = valid_props[file.stem.split("_")[0]]["units"]
            ds.attrs["long_name"] = valid_props[file.stem.split("_")[0]]["long_name"]
            return ds

    soil = xr.merge((get_tif(f) for f in files), combine_attrs="drop_conflicts")
    _ = soil.attrs.pop("_FillValue", None)
    _ = soil.attrs.pop("units", None)
    _ = soil.attrs.pop("long_name", None)
    return soil


def _open_tiff(file: Path, name: str) -> xr.DataArray:
    """Open a .tif file."""
    ds = rxr.open_rasterio(file).squeeze(drop=True)
    ds.name = name
    return xr.where(ds != ds.rio.nodata, ds, np.nan).rio.write_nodata(np.nan)


def soil_gnatsgo(layers: list[str] | str, geometry: GTYPE, crs: CRSType = 4326) -> xr.Dataset:
    """Get US soil data from the gNATSGO dataset.

    Notes
    -----
    This function uses Microsoft's Planetary Computer service to get the data.
    The dataset's description and its supported soil properties can be found at:
    https://planetarycomputer.microsoft.com/dataset/gnatsgo-rasters

    Parameters
    ----------
    layers : list of str or str
        Target layer(s). Available layers can be found at the dataset's website
        `here <https://planetarycomputer.microsoft.com/dataset/gnatsgo-rasters>`__.
    geometry : Polygon, MultiPolygon, or tuple of length 4
        Geometry or bounding box of the region of interest.
    crs : int, str, or pyproj.CRS, optional
        The input geometry CRS, defaults to ``epsg:4326``.

    Returns
    -------
    xarray.Dataset
        Requested soil properties.
    """
    try:
        import planetary_computer
        import pystac_client
    except ImportError as ex:
        raise DependencyError("soil_gnatsgo", ["pystac-client", "planetary-computer"]) from ex

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    bounds = geoutils.geo2polygon(geometry, crs, 4326).bounds
    search = catalog.search(collections=["gnatsgo-rasters"], bbox=bounds)
    lyr_href = tlz.merge_with(
        set, ({n: a.href for n, a in i.assets.items()} for i in search.items())
    )
    layers_ = [layers] if isinstance(layers, str) else list(layers)
    layers_ = [lyr.lower() for lyr in layers_]

    def get_layer(lyr: str) -> xr.DataArray:
        fpaths = ogc.streaming_download(list(lyr_href[lyr]), file_extention="tiff")
        fpaths = [f for f in fpaths if f is not None]
        ds = xr.merge(_open_tiff(f, lyr) for f in fpaths)
        affine = ds.rio.transform(recalc=True)
        with rio.open(fpaths[0]) as src:
            ds_crs = src.crs
        ds = ds.rio.write_transform(affine)
        ds = ds.rio.write_crs(ds_crs)
        ds = ds.rio.write_coordinate_system()
        return ds

    soil = xr.merge((get_layer(lyr) for lyr in layers_), combine_attrs="drop_conflicts")
    poly = geoutils.geo2polygon(geometry, crs, soil.rio.crs)
    soil = geoutils.xarray_geomask(soil, poly, soil.rio.crs)
    _ = soil.attrs.pop("_FillValue", None)
    _ = soil.attrs.pop("units", None)
    _ = soil.attrs.pop("long_name", None)
    return soil


def soil_soilgrids(
    layers: list[str] | str,
    geometry: GTYPE,
    geo_crs: CRSType = 4326,
) -> xr.Dataset:
    """Get soil data from SoilGrids for the area of interest at 250m resolution.

    Notes
    -----
    For more information on the SoilGrids dataset, visit
    `ISRIC <https://www.isric.org/explore/soilgrids/faq-soilgrids#What_do_the_filename_codes_mean>`__.

    Parameters
    ----------
    layers : list of str
        SoilGrids layers to get. Available options are:
        ``bdod_*``, ``cec_*``, ``cfvo_*``, ``clay_*``, ``nitrogen_*``, ``ocd_*``,
        ``ocs_*``, ``phh2o_*``, ``sand_*``, ``silt_*``, and ``soc_*``
        represents depths in cm and can be one of ``5`` (0-5 cm), ``15``
        (5-15 cm), ``30`` (15-30 cm), ``60`` (30-60 cm), ``100`` (60-100 cm), or ``200``
        (100-200 cm). For example, ``bdod_5`` is the mean bulk density of
        the fine earth fraction at 0-5 cm depth, and ``bdod_200`` is the mean bulk
        density of the fine earth fraction at 100-200 cm depth.
    geometry : Polygon, MultiPolygon, or tuple of length 4
        Geometry to get DEM within. It can be a polygon or a boundong box
        of form (xmin, ymin, xmax, ymax).
    geo_crs : int, str, of pyproj.CRS, optional
        CRS of the input geometry, defaults to ``epsg:4326``.

    Returns
    -------
    xarray.DataArray
        The request DEM at the specified resolution.
    """
    layers_ = [layers] if isinstance(layers, str) else list(layers)
    valid_depths = {
        "5": "0-5cm",
        "15": "5-15cm",
        "30": "15-30cm",
        "60": "30-60cm",
        "100": "60-100cm",
        "200": "100-200cm",
    }
    valid_layers = [
        f"{layer}_{depth}" for layer, depth in itertools.product(SG_ATTRS, valid_depths)
    ]
    invalid_layers = [layer for layer in layers_ if layer not in valid_layers]
    if invalid_layers:
        raise InputValueError("layers", valid_layers, ", ".join(invalid_layers))
    lyr_names, lyr_depths = zip(*[layer.split("_") for layer in layers_])
    base_url = "https://files.isric.org/soilgrids/latest/data"
    geometry_ = geoutils.geo2polygon(geometry)
    bounds = geometry_.bounds

    def _read_layer(lyr: str, depth: str) -> xr.DataArray:
        """Read a SoilGrids layer."""
        ds = rxr.open_rasterio(f"{base_url}/{lyr}/{lyr}_{depth}_mean.vrt")
        ds = cast("xr.DataArray", ds)
        ds = (
            ds.squeeze(drop=True)
            .rio.clip_box(*bounds, crs=geo_crs)
            .rio.clip([geometry_], crs=geo_crs)
        )
        ds = ds.where(ds != ds.rio.nodata)
        ds = ds.rio.write_nodata(np.nan)

        # Convert mapped units to conventional units
        attributes = SG_ATTRS[lyr]
        ds = ds / attributes["conversion_factor"]

        ds.name = f"{lyr}_{depth.replace('-', '_')}_mean"
        ds.attrs["long_name"] = f"Mean {attributes['long_name']} ({depth})"
        ds.attrs["description"] = attributes["description"]
        ds.attrs["units"] = attributes["conventional_units"]

        return ds

    return xr.merge([_read_layer(lyr, valid_depths[d]) for lyr, d in zip(lyr_names, lyr_depths)])


def soil_polaris(
    layers: list[str] | str,
    geometry: GTYPE,
    geo_crs: CRSType = 4326,
) -> xr.Dataset:
    """Get soil data from Polaris for the area of interest at 30m resolution.

    Notes
    -----
    For more information on the Polaris dataset, read the original paper
    `here <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR022797>`__.

    Parameters
    ----------
    layers : list of str
        SoilGrids layers to get. Available options are:
        ``bd_*``, ``ksat_*``, ``ph_*``, ``om_*``, ``sand_*``, ``silt_*``, ``clay_*``,
        ``theta_s_*``, ``theta_r_*``, ``lambda_*``, ``hb_*``, ``n_*``, and ``alpha_*``
        where ``*`` represents depths in cm and can be one of ``5`` (0-5 cm), ``15``
        (5-15 cm), ``30`` (15-30 cm), ``60`` (30-60 cm), ``100`` (60-100 cm), or ``200``
        (100-200 cm). For example, ``bd_5`` is the mean bulk density of
        the fine earth fraction at 0-5 cm depth, and ``bd_200`` is the mean bulk
        density of the fine earth fraction at 100-200 cm depth.
    geometry : Polygon, MultiPolygon, or tuple of length 4
        Geometry to get DEM within. It can be a polygon or a boundong box
        of form (xmin, ymin, xmax, ymax).
    geo_crs : int, str, of pyproj.CRS, optional
        CRS of the input geometry, defaults to ``epsg:4326``.

    Returns
    -------
    xarray.DataArray
        The request DEM at the specified resolution.
    """
    layers_ = [layers] if isinstance(layers, str) else list(layers)
    valid_depths = {
        "5": "0_5",
        "15": "5_15",
        "30": "15_30",
        "60": "30_60",
        "100": "60_100",
        "200": "100_200",
    }
    valid_layers = [
        f"{layer}_{depth}" for layer, depth in itertools.product(PS_ATTRS, valid_depths)
    ]
    invalid_layers = [layer for layer in layers_ if layer not in valid_layers]
    if invalid_layers:
        raise InputValueError("layers", valid_layers, ", ".join(invalid_layers))
    lyr_names, lyr_depths = zip(*[layer.split("_") for layer in layers_])
    base_url = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt"
    geometry_ = geoutils.geo2polygon(geometry)
    bounds = geometry_.bounds

    def _read_layer(lyr: str, depth: str) -> xr.DataArray:
        """Read a SoilGrids layer."""
        ds = rxr.open_rasterio(f"{base_url}/{lyr}_mean_{depth}.vrt")
        ds = cast("xr.DataArray", ds)
        ds = (
            ds.squeeze(drop=True)
            .rio.clip_box(*bounds, crs=geo_crs)
            .rio.clip([geometry_], crs=geo_crs)
        )
        ds = ds.where(ds != ds.rio.nodata)
        ds = ds.rio.write_nodata(np.nan)

        attributes = PS_ATTRS[lyr]
        ds.name = f"{lyr}_{depth}cm_mean"
        ds.attrs["long_name"] = f"Mean {attributes['long_name']} ({depth.replace('_', '-')} cm)"
        ds.attrs["description"] = attributes["description"]
        ds.attrs["units"] = attributes["units"]
        return ds

    return xr.merge([_read_layer(lyr, valid_depths[d]) for lyr, d in zip(lyr_names, lyr_depths)])


class EHydro(AGRBase):
    """Access USACE Hydrographic Surveys (eHydro).

    Notes
    -----
    For more info visit: https://navigation.usace.army.mil/Survey/Hydro

    Methods
    -------
    bygeom(geom, geo_crs=4326, sql_clause="", distance=None, return_m=False, return_geom=True)
        Get features within a geometry that can be combined with a SQL where clause.
    byids(field, fids, return_m=False, return_geom=True)
        Get features by object IDs.
    bysql(sql_clause, return_m=False, return_geom=True)
        Get features using a valid SQL 92 WHERE clause.

    Parameters
    ----------
    data_type : str, optional
        Type of the survey data to retrieve, defaults to ``points``.
        Note that the ``points`` data type gets the best available point
        cloud data, i.e., if ``SurveyPointHD`` is available, it will be
        returned, otherwise ``SurveyPoint`` will be returned.
        Available types are:

        - ``points``: Point clouds
        - ``outlines``: Polygons of survey outlines
        - ``contours``: Depth contours
        - ``bathymetry``: Bathymetry data

        Note that point clouds are not available for all surveys.
    cache_dir : str or pathlib.Path, optional
        Directory to store the downloaded raw data, defaults to ``./ehydro_cache``.
    """

    def __init__(
        self,
        data_type: Literal["points", "outlines", "bathymetry", "contours"] = "points",
        cache_dir: str | Path = "ehydro_cache",
    ):
        super().__init__(ServiceURL().restful.ehydro)
        self.data_type = data_type
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        layer = {
            "points": "SurveyPoint",
            "outlines": "SurveyJob",
            "bathymetry": "Bathymetry_Vector",
            "contours": "ElevationContour_ALL",
        }
        if self.data_type not in layer:
            raise InputValueError("data_type", list(layer))
        self._layer = layer[self.data_type]
        self._error_msg = "No survey data found within the given bound."
        self._engine = "pyogrio" if importlib.util.find_spec("pyogrio") else "fiona"
        if self._engine == "pyogrio":
            import pyogrio

            pyogrio.set_gdal_config_options(
                {"OGR_GEOMETRY_ACCEPT_UNCLOSED_RING": "YES", "OGR_ORGANIZE_POLYGONS": "SKIP"}
            )
            warnings.filterwarnings("ignore", message=".*Non closed ring detected.*")
            warnings.filterwarnings("ignore", message=".*translated to Simple Geometry.*")
            self._get_layers = pyogrio.list_layers
        else:
            import logging

            import fiona  # pyright: ignore[reportMissingImports]

            root_logger = logging.getLogger("fiona")
            root_logger.setLevel(logging.ERROR)
            self._get_layers = fiona.listlayers

        bins = AGRBase(ServiceURL().restful.ehydro_bins)
        extent = bins.client.client.extent
        if extent is None:
            msg = "Unable to get the extent of eHydro"
            raise ServiceError(msg)
        self._survey_grid = bins.bygeom(extent)

        warnings.filterwarnings("ignore", message=".*3D.*")
        warnings.filterwarnings("ignore", message=".*OGR_GEOMETRY_ACCEPT_UNCLOSED_RING.*")

    def __post_process(self, survey: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        urls = survey.loc[
            ~survey["sourcedatalocation"].str.contains("placeholder"), "sourcedatalocation"
        ].to_list()
        if not urls:
            raise ZeroMatchedError(self._error_msg)
        fnames = [Path(self.cache_dir, Path(u).name) for u in urls]
        fnames = ogc.streaming_download(urls, fnames=fnames)
        fnames = [f for f in fnames if f is not None]
        if not fnames:
            raise ZeroMatchedError(self._error_msg)

        def get_depth(fname: Path) -> gpd.GeoDataFrame:
            with ZipFile(fname, "r") as zip_ref:
                gdb = next(
                    (
                        f"zip://{fname}!{n.filename}"
                        for n in zip_ref.filelist
                        if n.filename.endswith(".gdb/")
                    ),
                    None,
                )
                if gdb is None:
                    raise ZeroMatchedError(self._error_msg)

            layers = self._get_layers(gdb)
            if self._layer not in layers:
                raise ZeroMatchedError(self._error_msg)
            if self._layer == "SurveyPoint" and "SurveyPointHD" in layers:
                self._layer = "SurveyPointHD"

            gdf = None
            if self._engine == "pyogrio":
                with contextlib.suppress(GEOSException):
                    gdf = gpd.read_file(gdb, layer=self._layer, engine="pyogrio", use_arrow=True)
            if gdf is None:
                gdf = gpd.read_file(gdb, layer=self._layer)
            return gdf.to_crs(5070)  # pyright: ignore[reportCallIssue,reportReturnType]

        return gpd.GeoDataFrame(pd.concat((get_depth(f) for f in fnames), ignore_index=True))

    def _getfeatures(
        self, oids: Iterator[tuple[str, ...]], return_m: bool = False, return_geom: bool = True
    ) -> gpd.GeoDataFrame:
        """Send a request for getting data based on object IDs.

        Parameters
        ----------
        return_m : bool
            Whether to activate the Return M (measure) in the request, defaults to False.
        return_geom : bool, optional
            Whether to return the geometry of the feature, defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            The requested features as a GeoDataFrame.
        """
        try:
            return self.__post_process(
                geoutils.json2geodf(
                    self.client.get_features(oids, return_m, return_geom), self.client.client.crs
                )
            )
        except EmptyResponseError as ex:
            raise ZeroMatchedError(self._error_msg) from ex

    @property
    def survey_grid(self) -> gpd.GeoDataFrame:
        """Full survey availability on hexagonal grid cells of 35 km resolution."""
        return self._survey_grid
