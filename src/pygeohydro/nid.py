"""Accessing data from the supported databases through their APIs."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast

import geopandas as gpd
import pandas as pd
import requests

import async_retriever as ar
import pygeoogc as ogc
import pygeoutils as geoutils
from pygeohydro.exceptions import (
    InputTypeError,
    InputValueError,
    ServiceError,
    ZeroMatchedError,
)
from pygeoogc import ServiceURL
from pygeoutils.exceptions import EmptyResponseError

if TYPE_CHECKING:
    from pyproj import CRS
    from shapely import MultiPolygon, Polygon

    GTYPE = Union[Polygon, MultiPolygon, tuple[float, float, float, float]]

    CRSType = int | str | CRS

__all__ = ["NID"]


def _remote_file_modified(file_path: Path) -> bool:
    """Check if the file is older than the last modification date of the NID web service."""
    if not file_path.exists():
        return True
    url = "https://nid.sec.usace.army.mil/api/nation/gpkg"
    # we need to get the redirect URL so that we can get the last modified date, so
    # we need to send a request with the Range header set to 0-0 to avoid downloading
    # the entire file.
    response = requests.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=50)
    if response.status_code not in (200, 206):
        raise ServiceError(response.reason, url)
    response = requests.head(response.url, timeout=50)
    if response.status_code != 200:
        raise ServiceError(response.reason, url)

    remote_last_modified = datetime.strptime(
        response.headers["Last-Modified"], "%a, %d %b %Y %H:%M:%S GMT"
    ).replace(tzinfo=timezone.utc)
    local_last_modified = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    return local_last_modified < remote_last_modified


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
        dtype_str = {str(k): v for k, v in self.dam_type.items() if str(k).isdigit()}
        self.dam_type = self.dam_type | dtype_str
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
        purp_str = {str(k): v for k, v in self.dam_purpose.items() if str(k).isdigit()}
        self.dam_purpose = self.dam_purpose | purp_str
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
        self._nid_inventory_path = Path("cache", "full_nid_inventory.parquet")

    @property
    def nid_inventory_path(self) -> Path:
        """Path to the NID inventory parquet file."""
        return self._nid_inventory_path

    @nid_inventory_path.setter
    def nid_inventory_path(self, value: Path | str) -> None:
        self._nid_inventory_path = Path(value)
        self._nid_inventory_path.parent.mkdir(parents=True, exist_ok=True)

    def stage_nid_inventory(self, fname: str | Path | None = None) -> None:
        """Download the entire NID inventory data and save to a parquet file.

        Parameters
        ----------
        fname : str, pathlib.Path, optional
            The path to the file to save the data to, defaults to
            ``./cache/full_nid_inventory.parquet``.
        """
        fname = self.nid_inventory_path if fname is None else Path(fname)
        if fname.suffix != ".parquet":
            fname = fname.with_suffix(".parquet")

        self.nid_inventory_path = fname
        gpkg_file = fname.with_suffix(".gpkg")
        if _remote_file_modified(gpkg_file) or not self.nid_inventory_path.exists():
            gpkg_file.unlink(missing_ok=True)
            url = "https://nid.sec.usace.army.mil/api/nation/gpkg"
            fname_ = ogc.streaming_download(url, fnames=gpkg_file)
            if fname_ is None:
                raise EmptyResponseError
            dams = (
                gpd.read_file(fname_, engine="pyogrio", use_arrow=True)
                if importlib.util.find_spec("pyogrio")
                else gpd.read_file(fname_)
            )
            dams = cast("gpd.GeoDataFrame", dams)

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
                    "separateStructuresCount": str,
                    "isAssociatedStructureId": str,
                    "designerNames": str,
                    "nonFederalDamOnFederalId": str,
                    "primaryPurposeId": str,
                    "purposeIds": str,
                    "sourceAgency": str,
                    "stateFedId": str,
                    "latitude": "f8",
                    "longitude": "f8",
                    "state": str,
                    "county": str,
                    "countyState": str,
                    "city": str,
                    "distance": "f8",
                    "riverName": str,
                    "congDist": str,
                    "stateRegulatedId": str,
                    "jurisdictionAuthorityId": str,
                    "stateRegulatoryAgency": str,
                    "permittingAuthorityId": str,
                    "inspectionAuthorityId": str,
                    "enforcementAuthorityId": str,
                    "fedRegulatedId": str,
                    "fedOwnerIds": str,
                    "fedFundingIds": str,
                    "fedDesignIds": str,
                    "fedConstructionIds": str,
                    "fedRegulatoryIds": str,
                    "fedInspectionIds": str,
                    "fedOperationIds": str,
                    "fedOtherIds": str,
                    "secretaryAgricultureBuiltId": str,
                    "nrcsWatershedAuthorizationId": str,
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
                    "yearsModified": str,
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
                    "secondaryLengthOfLocks": "Int32",
                    "secondaryWidthOfLocks": "Int32",
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
                    "usaceDivision": str,
                    "usaceDistrict": str,
                    "operationalStatusId": str,
                    "operationalStatusDate": "datetime64[ms]",
                    "inundationNidAddedId": str,
                    "huc2": str,
                    "huc4": str,
                    "huc6": str,
                    "huc8": str,
                    "zipcode": str,
                    "nation": str,
                    "stateKey": str,
                    "femaRegion": str,
                    "femaCommunity": str,
                    "aiannh": str,
                }
            )
            for c in dams:
                if (dams[c] == "Yes").any():
                    dams[c] = dams[c] == "Yes"
            dams.loc[dams["yearCompleted"] < 1000, "yearCompleted"] = pd.NA
            dams.to_parquet(fname)

    @property
    def df(self):
        """Entire NID inventory (``csv`` version) as a ``pandas.DataFrame``."""
        fname = self.nid_inventory_path
        par_name = fname.with_suffix(".parquert")
        if par_name.exists():
            return pd.read_parquet(par_name)
        url = "https://nid.sec.usace.army.mil/api/nation/csv"
        fname = ogc.streaming_download(url, fnames=fname.with_suffix(".csv"))
        if fname is None:
            raise EmptyResponseError
        dams = pd.read_csv(fname, header=1, engine="pyarrow")
        dams.to_parquet(par_name)
        return dams

    @property
    def gdf(self):
        """Entire NID inventory (``gpkg`` version) as a ``geopandas.GeoDataFrame``."""
        self.stage_nid_inventory()
        return gpd.read_parquet(self.nid_inventory_path)

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

        kwds = None if params is None else [{"params": p | {"out": "json"}} for p in params]
        resp = ar.retrieve_json(urls, kwds)
        resp = cast("list[dict[str, Any]]", resp)
        if not resp:
            raise ZeroMatchedError

        failed = [(i, f"Req_{i}: {r['message']}") for i, r in enumerate(resp) if "error" in r]
        if failed:
            idx, err_msgs = zip(*failed)
            idx = cast("tuple[int]", idx)
            err_msgs = cast("tuple[str]", err_msgs)
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
        return gpd.GeoDataFrame(  # pyright: ignore[reportCallIssue]
            nid_df,
            geometry=gpd.points_from_xy(nid_df["longitude"], nid_df["latitude"], crs=4326),
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

    def get_bygeom(self, geometry: GTYPE, geo_crs: CRSType) -> gpd.GeoDataFrame:
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

    def inventory_byid(self, federal_ids: list[str]) -> gpd.GeoDataFrame:
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
        if not isinstance(federal_ids, Iterable) or isinstance(federal_ids, (str, int)):
            raise InputTypeError("federal_ids", "list of str (Federal IDs)")

        if not all(isinstance(i, str) for i in federal_ids):
            raise InputTypeError("federal_ids", "list of str (Federal IDs)")

        urls = [f"{self.base_url}/dams/{i.upper()}/inventory" for i in set(federal_ids)]
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
