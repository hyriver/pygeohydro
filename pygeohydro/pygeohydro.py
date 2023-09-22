"""Accessing data from the supported databases through their APIs."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import warnings
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Sequence, Tuple, Union, cast
from unittest.mock import patch

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import rasterio as rio
import xarray as xr
from rioxarray import _io as rxr
from shapely import __version__ as shapely_version

if shapely_version[0] == "2":
    from shapely.errors import GEOSException
else:
    GEOSException = BaseException
import async_retriever as ar
import pygeoogc as ogc
import pygeoutils as geoutils
from pygeohydro import helpers
from pygeohydro.exceptions import (
    DependencyError,
    InputTypeError,
    InputValueError,
    MissingColumnError,
    ZeroMatchedError,
)
from pygeoogc import RetrySession, ServiceURL
from pygeoutils import EmptyResponseError
from pynhd.core import AGRBase, ScienceBase

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

    GTYPE = Union[Polygon, MultiPolygon, Tuple[float, float, float, float]]
    CRSTYPE = Union[int, str, pyproj.CRS]

__all__ = [
    "get_camels",
    "ssebopeta_bycoords",
    "ssebopeta_bygeom",
    "soil_properties",
    "soil_gnatsgo",
    "NID",
    "EHydro",
]


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
    base_url = "/".join(
        (
            "https://thredds.hydroshare.org/thredds/fileServer/hydroshare",
            "resources/658c359b8c83494aac0f58145b1b04e6/data/contents",
        )
    )
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
    crs: CRSTYPE = 4326,
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
        gpd.points_from_xy(coords["x"], coords["y"]), index=coords["id"], crs=crs
    )
    _coords = _coords.to_crs(4326)
    co_list = list(zip(_coords.x, _coords.y))

    f_list = helpers.get_ssebopeta_urls(dates)
    session = RetrySession()

    with patch("socket.has_ipv6", False), RetrySession() as session:

        def _ssebop(url: str) -> list[npt.NDArray[np.float64]]:
            r = session.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))

            with rio.MemoryFile() as memfile:
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
    geo_crs: CRSTYPE = 4326,
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
    except geoutils.InputTypeError as ex:
        raise InputTypeError("geometry", "(Multi)Polygon or tuple of length 4") from ex

    gtiff2xarray = tlz.partial(geoutils.gtiff2xarray, geometry=geometry, geo_crs=geo_crs)
    with patch("socket.has_ipv6", False), RetrySession() as session:

        def _ssebop(t: pd.Timestamp, url: str) -> xr.DataArray:
            resp = session.get(url)
            zfile = zipfile.ZipFile(io.BytesIO(resp.content))
            content = zfile.read(zfile.filelist[0].filename)
            ds = gtiff2xarray(r_dict={"eta": content})
            ds = cast("xr.DataArray", ds)
            return ds.expand_dims({"time": [t]})

        data = xr.merge(itertools.starmap(_ssebop, f_list))
    eta: xr.DataArray = data.eta
    eta = eta.where(eta < eta.rio.nodata) * 1e-3
    eta.attrs.update(
        {
            "units": "mm/day",
            "nodatavals": (np.nan,),
            "crs": 4326,
            "long_name": "Actual ET",
        }
    )
    return eta


class NID:
    """Retrieve data from the National Inventory of Dams web service."""

    def __init__(self) -> None:
        self.base_url = ServiceURL().restful.nid
        self.suggest_url = f"{self.base_url}/suggestions"
        resp = ar.retrieve_json([f"{self.base_url}/advanced-fields"])
        resp = cast("list[dict[str, Any]]", resp)
        self.fields_meta = pd.DataFrame(resp[0])
        self.valid_fields = self.fields_meta["name"]
        self.dam_type = {
            pd.NA: "N/A",
            None: "N/A",
            1: "Arch",
            2: "Multi-Arch",
            3: "Stone",
            4: "Roller-Compacted Concrete",
            5: "Rockfill",
            6: "Buttress",
            7: "Masonry",
            8: "Earth",
            9: "Gravity",
            10: "Timber Crib",
            11: "Concrete",
            12: "Other",
        }
        self.dam_type.update({str(k): v for k, v in self.dam_type.items() if str(k).isdigit()})
        self.dam_purpose = {
            pd.NA: "N/A",
            None: "N/A",
            1: "Tailings",
            2: "Irrigation",
            3: "Navigation",
            4: "Fish and Wildlife Pond",
            5: "Recreation",
            6: "Hydroelectric",
            7: "Debris Control",
            8: "Water Supply",
            9: "Flood Risk Reduction",
            10: "Fire Protection, Stock, Or Small Farm Pond",
            11: "Grade Stabilization",
            12: "Other",
        }
        self.dam_purpose.update(
            {str(k): v for k, v in self.dam_purpose.items() if str(k).isdigit()}
        )
        self.data_units = {
            "distance": "mile",
            "damHeight": "ft",
            "hydraulicHeight": "ft",
            "structuralHeight": "ft",
            "nidHeight": "ft",
            "damLength": "ft",
            "volume": "cubic yards",
            "nidStorage": "acre-ft",
            "maxStorage": "acre-ft",
            "normalStorage": "acre-ft",
            "surfaceArea": "acre",
            "drainageArea": "square miles",
            "maxDischarge": "cfs",
            "spillwayWidth": "ft",
        }
        self._nid_inventory_path = Path("cache", "nid_inventory.feather")

    @property
    def nid_inventory_path(self) -> Path:
        """Path to the NID inventory feather file."""
        return self._nid_inventory_path

    @nid_inventory_path.setter
    def nid_inventory_path(self, value: Path | str) -> None:
        self._nid_inventory_path = Path(value)
        self._nid_inventory_path.parent.mkdir(parents=True, exist_ok=True)

    def stage_nid_inventory(self, fname: str | Path | None = None) -> None:
        """Download the entire NID inventory data and save to a feather file.

        Parameters
        ----------
        fname : str, pathlib.Path, optional
            The path to the file to save the data to, defaults to
            ``./cache/nid_inventory.feather``.
        """
        fname = self.nid_inventory_path if fname is None else Path(fname)
        if fname.suffix != ".feather":
            fname = fname.with_suffix(".feather")

        self.nid_inventory_path = fname
        if not self.nid_inventory_path.exists():
            url = "https://nid.sec.usace.army.mil/api/nation/gpkg"
            _ = ogc.streaming_download(url, fnames=fname.with_suffix(".gpkg"))
            dams = (
                gpd.read_file(fname.with_suffix(".gpkg"), engine="pyogrio", use_arrow=True)
                if importlib.util.find_spec("pyogrio")
                else gpd.read_file(fname.with_suffix(".gpkg"))
            )

            dams = dams.astype(
                {
                    "name": str,
                    "otherNames": str,
                    "formerNames": str,
                    "nidId": str,
                    "otherStructureId": str,
                    "federalId": str,
                    "ownerNames": str,
                    "ownerTypeIds": str,
                    "primaryOwnerTypeId": str,
                    "stateFedId": str,
                    "separateStructuresCount": str,
                    "designerNames": str,
                    "nonFederalDamOnFederalId": str,
                    "stateRegulatedId": str,
                    "jurisdictionAuthorityId": str,
                    "stateRegulatoryAgency": str,
                    "permittingAuthorityId": str,
                    "inspectionAuthorityId": str,
                    "enforcementAuthorityId": str,
                    "sourceAgency": str,
                    "latitude": "f8",
                    "longitude": "f8",
                    "county": str,
                    "state": str,
                    "city": str,
                    "distance": "f8",
                    "riverName": str,
                    "congDist": str,
                    "countyState": str,
                    "location": str,
                    "fedOwnerIds": str,
                    "fedFundingIds": str,
                    "fedDesignIds": str,
                    "fedConstructionIds": str,
                    "fedRegulatoryIds": str,
                    "fedInspectionIds": str,
                    "fedOperationIds": str,
                    "fedOtherIds": str,
                    "primaryPurposeId": str,
                    "purposeIds": str,
                    "primaryDamTypeId": str,
                    "damTypeIds": str,
                    "coreTypeIds": str,
                    "foundationTypeIds": str,
                    "damHeight": "f8",
                    "hydraulicHeight": "f8",
                    "structuralHeight": "f8",
                    "nidHeight": "f8",
                    "nidHeightId": str,
                    "damLength": "f8",
                    "volume": "f8",
                    "yearCompleted": "Int32",
                    "yearCompletedId": str,
                    "nidStorage": "f8",
                    "maxStorage": "f8",
                    "normalStorage": "f8",
                    "surfaceArea": "f8",
                    "drainageArea": "f8",
                    "maxDischarge": "f8",
                    "spillwayTypeId": str,
                    "spillwayWidth": "f8",
                    "numberOfLocks": "Int32",
                    "lengthOfLocks": "f8",
                    "widthOfLocks": "f8",
                    "yearsModified": str,
                    "outletGateTypes": str,
                    "dataUpdated": "datetime64[ns]",
                    "inspectionDate": str,
                    "inspectionFrequency": "f4",
                    "hazardId": str,
                    "conditionAssessId": str,
                    "conditionAssessDate": "datetime64[ns]",
                    "eapId": str,
                    "eapLastRevDate": "datetime64[ns]",
                    "websiteUrl": str,
                    "privateDamId": str,
                    "politicalPartyId": str,
                    "id": "int32",
                    "systemId": "int32",
                    "huc2": str,
                    "huc4": str,
                    "huc6": str,
                    "huc8": str,
                    "zipcode": str,
                    "nation": str,
                    "stateKey": str,
                    "femaRegion": str,
                    "femaCommunity": str,
                }
            )
            for c in dams:
                if (dams[c] == "Yes").any():
                    dams[c] = dams[c] == "Yes"
            dams.loc[dams["yearCompleted"] < 1000, "yearCompleted"] = pd.NA
            dams.to_feather(fname)
            fname.with_suffix(".gpkg").unlink()

    @property
    def df(self):
        """Entire NID inventory (``csv`` version) as a ``pandas.DataFrame``."""
        fname = self.nid_inventory_path
        par_name = fname.with_suffix(".parquert")
        if par_name.exists():
            return pd.read_parquet(par_name)
        url = "https://nid.sec.usace.army.mil/api/nation/csv"
        _ = ogc.streaming_download(url, fnames=fname.with_suffix(".csv"))
        dams = pd.read_csv(fname.with_suffix(".csv"), header=1, engine="pyarrow")
        dams.to_parquet(par_name)
        return dams

    @property
    def gdf(self):
        """Entire NID inventory (``gpkg`` version) as a ``geopandas.GeoDataFrame``."""
        self.stage_nid_inventory()
        return gpd.read_feather(self.nid_inventory_path)

    @staticmethod
    def _get_json(
        urls: Sequence[str], params: list[dict[str, str]] | None = None
    ) -> list[dict[str, Any]]:
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
            raise InputTypeError("urls", "list or str")

        kwds = None if params is None else [{"params": {**p, "out": "json"}} for p in params]
        resp = ar.retrieve_json(urls, kwds)
        resp = cast("list[dict[str, Any]]", resp)
        if not resp:
            raise ZeroMatchedError

        failed = [(i, f"Req_{i}: {r['message']}") for i, r in enumerate(resp) if "error" in r]
        if failed:
            idx, err_msgs = zip(*failed)
            idx = cast("Tuple[int]", idx)
            err_msgs = cast("Tuple[str]", err_msgs)
            errs = " service requests failed with the following messages:\n"
            errs += "\n".join(err_msgs)
            if len(failed) == len(urls):
                raise ZeroMatchedError(f"All{errs}")
            resp = [r for i, r in enumerate(resp) if i not in idx]
            fail_count = f"{len(failed)} of {len(urls)}"
            warnings.warn(f"{fail_count}{errs}", UserWarning, stacklevel=2)
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
            nid_df,
            geometry=gpd.points_from_xy(nid_df["longitude"], nid_df["latitude"]),
            crs=4326,
        )

    def get_byfilter(self, query_list: list[dict[str, list[str]]]) -> list[gpd.GeoDataFrame]:
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
        list of geopandas.GeoDataFrame
            Query results in the same order as the input query list.

        Examples
        --------
        >>> from pygeohydro import NID
        >>> nid = NID()
        >>> query_list = [
        ...    {"drainageArea": ["[200 500]"]},
        ...    {"nidId": ["CA01222"]},
        ... ]
        >>> dam_dfs = nid.get_byfilter(query_list)
        """
        fields = self.valid_fields.to_list()
        invalid = [k for key in query_list for k in key if k not in fields]
        if invalid:
            raise InputValueError("query_dict", fields)
        params = [
            {"sy": " ".join(f"@{s}:{fid}" for s, fids in key.items() for fid in fids)}
            for key in query_list
        ]
        return [
            self._to_geodf(pd.DataFrame(r))
            for r in self._get_json([f"{self.base_url}/query"] * len(params), params)
        ]

    def get_bygeom(self, geometry: GTYPE, geo_crs: CRSTYPE) -> gpd.GeoDataFrame:
        """Retrieve NID data within a geometry.

        Parameters
        ----------
        geometry : Polygon, MultiPolygon, or tuple of length 4
            Geometry or bounding box (west, south, east, north) for extracting the data.
        geo_crs : list of str
            The CRS of the input geometry.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of NID data

        Examples
        --------
        >>> from pygeohydro import NID
        >>> nid = NID()
        >>> dams = nid.get_bygeom((-69.77, 45.07, -69.31, 45.45), 4326)
        """
        _geometry = geoutils.geo2polygon(geometry, geo_crs, self.gdf.crs)
        idx = self.gdf.sindex.query(_geometry, "contains")
        return self.gdf.iloc[idx].copy()

    def inventory_byid(self, federal_ids: list[int]) -> gpd.GeoDataFrame:
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
        federal_ids : list of str
            List of the target dam Federal IDs.

        Returns
        -------
        pandas.DataFrame
            Dams with extra attributes in addition to the standard NID fields
            that other ``NID`` methods return.

        Examples
        --------
        >>> from pygeohydro import NID
        >>> nid = NID()
        >>> dams = nid.inventory_byid(['KY01232', 'GA02400', 'NE04081', 'IL55070', 'TN05345'])
        """
        ids = set(federal_ids) if isinstance(federal_ids, (list, tuple)) else {federal_ids}
        ids = {str(i).upper() for i in ids}
        urls = [f"{self.base_url}/dams/{i}/inventory" for i in ids]
        if len(urls) != len(ids):
            raise InputTypeError("dam_ids", "list of Federal IDs")
        return self._to_geodf(pd.DataFrame(self._get_json(urls)).set_index("id"))

    def get_suggestions(
        self, text: str, context_key: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        >>> dams, contexts = nid.get_suggestions("houston", "city")
        """
        fields = self.valid_fields.to_list()
        params = {"text": text}
        if context_key:
            if context_key not in fields:
                raise InputValueError("context", fields)
            params["contextKey"] = context_key
        resp = self._get_json([f"{self.base_url}/suggestions"], [params])
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
    sb = ScienceBase()
    soil = sb.get_file_urls("5fd7c19cd34e30b9123cb51f")
    pat = "|".join(prop)
    _files, urls = zip(*soil[soil.index.str.contains(f"{pat}.*zip")].url.to_dict().items())
    urls = list(urls)

    root = Path(soil_dir)
    root.mkdir(exist_ok=True, parents=True)
    files = [root.joinpath(f) for f in _files]
    _ = ogc.streaming_download(urls, fnames=files)

    def get_tif(file: Path) -> xr.DataArray:
        """Get the .tif file from a zip file."""
        with zipfile.ZipFile(file) as z:
            try:
                fname = next(f.filename for f in z.filelist if f.filename.endswith(".tif"))
            except StopIteration as ex:
                raise ZeroMatchedError from ex
            ds = rxr.open_rasterio(io.BytesIO(z.read(fname)))
            ds = cast("xr.DataArray", ds)
            if "band" in ds.dims:
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
    ds = rxr.open_rasterio(file)
    ds = cast("xr.DataArray", ds)
    if "band" in ds.dims:
        ds = ds.squeeze("band", drop=True)
    ds.name = name
    ds = xr.where(ds != ds.rio.nodata, ds, np.nan)
    ds = ds.rio.write_nodata(np.nan)
    return ds


def soil_gnatsgo(layers: list[str] | str, geometry: GTYPE, crs: CRSTYPE = 4326) -> xr.Dataset:
    """Get US soil data from the gNATSGO dataset.

    Notes
    -----
    This function uses Microsoft's Planetary Computer service to get the data.
    The dataset's description and its suppoerted soil properties can be found at:
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
    lyrs = [layers.lower()] if isinstance(layers, str) else map(str.lower, layers)

    def get_layer(lyr: str) -> xr.DataArray:
        fpaths = ogc.streaming_download(list(lyr_href[lyr]), file_extention="tiff")
        ds = xr.merge(_open_tiff(f, lyr) for f in fpaths)
        affine = ds.rio.transform(recalc=True)
        with rio.open(fpaths[0]) as src:
            ds_crs = src.crs
        ds = ds.rio.write_transform(affine)
        ds = ds.rio.write_crs(ds_crs)
        ds = ds.rio.write_coordinate_system()
        return ds

    ds = xr.merge((get_layer(lyr) for lyr in lyrs), combine_attrs="drop_conflicts")
    poly = geoutils.geo2polygon(geometry, crs, ds.rio.crs)
    ds = geoutils.xarray_geomask(ds, poly, ds.rio.crs)
    _ = ds.attrs.pop("_FillValue", None)
    _ = ds.attrs.pop("units", None)
    _ = ds.attrs.pop("long_name", None)
    return ds


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
    """

    def __init__(
        self, data_type: Literal["points", "outlines", "bathymetry", "contours"] = "points"
    ):
        super().__init__(ServiceURL().restful.ehydro)
        self.data_type = data_type
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

            import fiona

            root_logger = logging.getLogger("fiona")
            root_logger.setLevel(logging.ERROR)
            self._get_layers = fiona.listlayers

        bins = AGRBase(ServiceURL().restful.ehydro_bins)
        self._survey_grid = bins.bygeom(bins.client.client.extent)

        warnings.filterwarnings("ignore", message=".*3D.*")
        warnings.filterwarnings("ignore", message=".*OGR_GEOMETRY_ACCEPT_UNCLOSED_RING.*")

    def __post_process(self, survey: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        urls = survey["sourcedatalocation"].to_list()
        fnames = [Path("cache", Path(u).name) for u in urls]
        _ = ogc.streaming_download(urls, fnames=fnames)

        def get_depth(fname: Path) -> gpd.GeoDataFrame:
            with zipfile.ZipFile(fname, "r") as zip_ref:
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

            if self._layer == "SurveyPoint" and "SurveyPointHD" in self._get_layers(gdb):
                self._layer = "SurveyPointHD"

            if self._engine == "pyogrio":
                with contextlib.suppress(GEOSException):
                    return gpd.read_file(gdb, layer=self._layer, engine="pyogrio", use_arrow=True)
            return gpd.read_file(gdb, layer=self._layer)

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
