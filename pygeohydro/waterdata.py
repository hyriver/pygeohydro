"""Accessing data from the supported databases through their APIs."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Union

import async_retriever as ar
import cytoolz as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
import pyproj
import xarray as xr
from loguru import logger
from pygeoogc import ServiceURL
from pygeoogc import ZeroMatchedError as ZeroMatchedErrorOGC
from pygeoogc import utils as ogc_utils
from pynhd import AGRBase, WaterData
from pynhd.core import ScienceBase

from .exceptions import DataNotAvailableError, InputTypeError, InputValueError, ZeroMatchedError

if TYPE_CHECKING:
    CRSTYPE = Union[int, str, pyproj.CRS]

T_FMT = "%Y-%m-%d"
__all__ = ["NWIS", "WBD", "huc_wb_full", "irrigation_withdrawals"]


class NWIS:
    """Access NWIS web service.

    Notes
    -----
    More information about query parameters and codes that NWIS accepts
    can be found at its help
    `webpage <https://help.waterdata.usgs.gov/codes-and-parameters>`__.
    """

    def __init__(self) -> None:
        self.url = ServiceURL().restful.nwis

    @staticmethod
    def retrieve_rdb(url: str, payloads: list[dict[str, str]]) -> pd.DataFrame:
        """Retrieve and process requests with RDB format.

        Parameters
        ----------
        url : str
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
        try:
            resp = ar.retrieve_text(
                [url] * len(payloads),
                [{"params": {**p, "format": "rdb"}} for p in payloads],
            )
        except ar.ServiceError as ex:
            raise ZeroMatchedError(ogc_utils.check_response(str(ex))) from ex

        with contextlib.suppress(StopIteration):
            not_found = next(filter(lambda x: x[0] != "#", resp), None)
            if not_found is not None:
                msg = re.findall("<p>(.*?)</p>", not_found)[1].rsplit(">", 1)[1]
                raise ZeroMatchedError(f"Server error message:\n{msg}")

        data = [r.strip().split("\n") for r in resp if r[0] == "#"]
        data = [t.split("\t") for d in data for t in d if "#" not in t]
        if not data:
            raise ZeroMatchedError

        rdb_df = pd.DataFrame.from_dict(dict(zip(data[0], d)) for d in data[2:])
        if "agency_cd" in rdb_df:
            rdb_df = rdb_df[~rdb_df.agency_cd.str.contains("agency_cd|5s")].copy()
        return rdb_df

    @staticmethod
    def _validate_usgs_queries(
        queries: list[dict[str, str]], expanded: bool = False
    ) -> list[dict[str, str]]:
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
            raise InputTypeError("query", "list of dict")

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
        if not_valid:
            raise InputValueError(f"query keys ({', '.join(not_valid)})", valid_query_keys)

        _queries = queries.copy()
        if expanded:
            _ = [
                q.pop(k) for k in ("outputDataTypeCd", "outputDataType") for q in _queries if k in q
            ]
            output_type = {"siteOutput": "expanded"}
        else:
            output_type = {"siteOutput": "basic"}

        return [{**query, **output_type, "format": "rdb"} for query in _queries]

    def get_info(
        self,
        queries: dict[str, str] | list[dict[str, str]],
        expanded: bool = False,
        fix_names: bool = True,
    ) -> gpd.GeoDataFrame:
        """Send multiple queries to USGS Site Web Service.

        Parameters
        ----------
        queries : dict or list of dict
            A single or a list of valid queries.
        expanded : bool, optional
            Whether to get expanded site information for example drainage area,
            default to False.
        fix_names : bool, optional
            If ``True``, reformat station names and some small annoyances,
            defaults to ``True``.

        Returns
        -------
        geopandas.GeoDataFrame
            A correctly typed ``GeoDataFrame`` containing site(s) information.
        """
        queries = [queries] if isinstance(queries, dict) else queries

        payloads = self._validate_usgs_queries(queries, False)
        sites = self.retrieve_rdb(f"{self.url}/site", payloads)

        def fix_station_nm(station_nm: str) -> str:
            name = station_nm.title().rsplit(" ", 1)
            if len(name) == 1:
                return name[0]

            name[0] = name[0] if name[0][-1] == "," else f"{name[0]},"
            name[1] = name[1].replace(".", "")
            return " ".join((name[0], name[1].upper() if len(name[1]) == 2 else name[1].title()))

        if fix_names and "station_nm" in sites:
            sites["station_nm"] = [fix_station_nm(n) for n in sites["station_nm"]]

        for c in sites.select_dtypes("object"):
            sites[c] = sites[c].str.strip().astype(str)

        numeric_cols = ["dec_lat_va", "dec_long_va", "alt_va", "alt_acy_va"]

        if expanded:
            payloads = self._validate_usgs_queries(queries, True)
            sites = sites.merge(
                self.retrieve_rdb(f"{self.url}/site", payloads),
                on="site_no",
                how="outer",
                suffixes=("", "_overlap"),
            )
            sites = sites.filter(regex="^(?!.*_overlap)")
            numeric_cols += ["drain_area_va", "contrib_drain_area_va"]

        with contextlib.suppress(KeyError):
            sites["begin_date"] = pd.to_datetime(sites["begin_date"])
            sites["end_date"] = pd.to_datetime(sites["end_date"])

        gii = WaterData("gagesii", 4326, False)
        try:
            gages = gii.byid("staid", sites.site_no.to_list())
        except ZeroMatchedErrorOGC:
            gages = gpd.GeoDataFrame()

        if len(gages) > 0:
            sites = pd.merge(
                sites,
                gages[["staid", "drain_sqkm", "hcdn_2009"]],
                left_on="site_no",
                right_on="staid",
                how="left",
            )
            sites = sites.drop(columns=["staid"])
            sites["hcdn_2009"] = sites.hcdn_2009 == "yes"
        else:
            sites["hcdn_2009"] = False
            sites["drain_sqkm"] = np.nan

        numeric_cols += ["drain_sqkm"]
        if "count_nu" in sites:
            numeric_cols.append("count_nu")
        sites[numeric_cols] = sites[numeric_cols].apply(pd.to_numeric, errors="coerce")

        return gpd.GeoDataFrame(
            sites,
            geometry=gpd.points_from_xy(sites.dec_long_va, sites.dec_lat_va),
            crs="epsg:4326",
        )

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

    @staticmethod
    def _to_xarray(qobs: pd.DataFrame, long_names: dict[str, str], mmd: bool) -> xr.Dataset:
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
                "time": qobs.index.tz_localize(None).to_numpy("datetime64[ns]"),
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
    def _get_attrs(siteinfo: pd.DataFrame, mmd: bool) -> tuple[dict[str, Any], dict[str, str]]:
        """Get attributes of the stations that have streaflow data."""
        cols = {
            "site_no": "site identification number",
            "station_nm": "station name",
            "dec_lat_va": "latitude",
            "dec_long_va": "longitude",
            "alt_va": "altitude",
            "alt_acy_va": "altitude accuracy",
            "alt_datum_cd": "altitude datum",
            "huc_cd": "hydrologic unit code",
        }
        if "begin_date" in siteinfo and "end_date" in siteinfo:
            cols.update(
                {
                    "begin_date": "availability begin date",
                    "end_date": "availability end date",
                }
            )
        attr_df = siteinfo[cols.keys()].groupby("site_no").first()
        if "begin_date" in attr_df and "end_date" in attr_df:
            attr_df["begin_date"] = attr_df.begin_date.dt.strftime(T_FMT)
            attr_df["end_date"] = attr_df.end_date.dt.strftime(T_FMT)
        attr_df.index = "USGS-" + attr_df.index
        attr_df["units"] = "mm/day" if mmd else "cms"
        attr_df["tz"] = "UTC"
        _ = cols.pop("site_no")
        return attr_df.to_dict(orient="index"), cols

    @staticmethod
    def _check_inputs(
        station_ids: Sequence[str] | str,
        dates: tuple[str, str],
        utc: bool | None,
    ) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
        """Validate inputs."""
        if not isinstance(station_ids, (str, Sequence, Iterable)):
            raise InputTypeError("ids", "str or list of str")

        sids = [station_ids] if isinstance(station_ids, str) else list(set(station_ids))
        sids_df = pd.Series(sids, dtype=str)
        sids_df = sids_df.str.lower().str.replace("usgs-", "").str.zfill(8)
        if not sids_df.str.isnumeric().all():
            raise InputTypeError("station_ids", "only digits")

        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InputTypeError("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0], utc=utc)
        end = pd.to_datetime(dates[1], utc=utc)
        return sids_df.to_list(), start, end

    def _drainage_area_sqm(self, siteinfo: pd.DataFrame, freq: str) -> pd.Series:
        """Get drainage area of the stations."""
        area = siteinfo[["site_no", "drain_sqkm"]].copy()
        if area["drain_sqkm"].isna().any():
            sids = area[area["drain_sqkm"].isna()].site_no
            queries = [
                {
                    "parameterCd": "00060",
                    "siteStatus": "all",
                    "outputDataTypeCd": freq,
                    "sites": ",".join(s),
                }
                for s in tlz.partition_all(1500, sids)
            ]
            info = self.get_info(queries, expanded=True)

            def get_idx(ids: list[str]) -> tuple[pd.Index, pd.Index]:
                return info.site_no.isin(ids), area.site_no.isin(ids)

            i_idx, a_idx = get_idx(sids)
            # Drainage areas in info are in sq mi and should be converted to sq km
            area.loc[a_idx, "drain_sqkm"] = info.loc[i_idx, "contrib_drain_area_va"] * 0.38610
            if area["drain_sqkm"].isna().any():
                sids = area[area["drain_sqkm"].isna()].site_no
                i_idx, a_idx = get_idx(sids)
                area.loc[a_idx, "drain_sqkm"] = info.loc[i_idx, "drain_area_va"] * 0.38610

        if area["drain_sqkm"].isna().all():
            raise DataNotAvailableError("drainage")
        return area.set_index("site_no").drain_sqkm * 1e6

    def _get_streamflow(
        self,
        sids: Sequence[str],
        start_dt: str,
        end_dt: str,
        freq: str,
        kwargs: dict[str, str],
    ) -> pd.DataFrame:
        """Convert json to dataframe."""
        payloads = [
            {
                "sites": ",".join(s),
                "startDT": start_dt,
                "endDT": end_dt,
                **kwargs,
            }
            for s in tlz.partition_all(1500, sids)
        ]
        resp = ar.retrieve_json(
            [f"{self.url}/{freq}"] * len(payloads), [{"params": p} for p in payloads]
        )

        def get_site_id(site_cd: dict[str, str]) -> str:
            """Get site id."""
            return f"{site_cd['agencyCode']}-{site_cd['value']}"

        r_ts = {
            get_site_id(t["sourceInfo"]["siteCode"][0]): t["values"][0]["value"]
            for r in resp
            for t in r["value"]["timeSeries"]
        }
        if not r_ts:
            raise DataNotAvailableError("discharge")

        def to_df(col: str, values: dict[str, Any]) -> pd.DataFrame:
            try:
                discharge = pd.DataFrame.from_records(
                    values, exclude=["qualifiers"], index=["dateTime"]
                )
            except KeyError:
                return pd.DataFrame()
            discharge["value"] = pd.to_numeric(discharge["value"], errors="coerce")
            tz = resp[0]["value"]["timeSeries"][0]["sourceInfo"]["timeZoneInfo"]
            tz_dict = {
                "CST": "US/Central",
                "MST": "US/Mountain",
                "PST": "US/Pacific",
                "EST": "US/Eastern",
            }
            time_zone = tz_dict.get(
                tz["defaultTimeZone"]["zoneAbbreviation"],
                tz["defaultTimeZone"]["zoneAbbreviation"],
            )
            discharge.index = [pd.Timestamp(i, tz=time_zone) for i in discharge.index]
            discharge.index = discharge.index.tz_convert("UTC")  # type: ignore[attr-defined]
            discharge.columns = [col]
            return discharge

        qobs = pd.concat(itertools.starmap(to_df, r_ts.items()), axis=1)
        if len(qobs) == 0:
            raise DataNotAvailableError("discharge")
        qobs[qobs.le(0)] = np.nan
        # Convert cfs to cms
        return qobs * np.float_power(0.3048, 3)

    def get_streamflow(
        self,
        station_ids: Sequence[str] | str,
        dates: tuple[str, str],
        freq: str = "dv",
        mmd: bool = False,
        to_xarray: bool = False,
    ) -> pd.DataFrame | xr.Dataset:
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
            raise InputValueError("freq", valid_freqs)
        utc = True if freq == "iv" else None

        sids, start, end = self._check_inputs(station_ids, dates, utc)

        queries = [
            {
                "parameterCd": "00060",
                "siteStatus": "all",
                "outputDataTypeCd": freq,
                "sites": ",".join(s),
                "startDt": start.strftime("%Y-%m-%d"),
                "endDt": end.strftime("%Y-%m-%d"),
            }
            for s in tlz.partition_all(1500, sids)
        ]

        try:
            siteinfo = self.get_info(queries)
        except ZeroMatchedError as ex:
            raise DataNotAvailableError("discharge") from ex

        params = {
            "format": "json",
            "parameterCd": "00060",
            "siteStatus": "all",
        }
        if freq == "dv":
            params.update({"statCd": "00003"})
            siteinfo = siteinfo[
                (siteinfo.stat_cd == "00003")
                & (siteinfo.parm_cd == "00060")
                & (start.tz_localize(None) < siteinfo.end_date)
                & (end.tz_localize(None) > siteinfo.begin_date)
            ]
        else:
            siteinfo = siteinfo[
                (siteinfo.parm_cd == "00060")
                & (start.tz_localize(None) < siteinfo.end_date)
                & (end.tz_localize(None) > siteinfo.begin_date)
            ]
        sids = list(siteinfo.site_no.unique())
        if not sids:
            raise DataNotAvailableError("discharge")

        time_fmt = T_FMT if utc is None else "%Y-%m-%dT%H:%M%z"
        start_dt = start.strftime(time_fmt)
        end_dt = end.strftime(time_fmt)
        qobs = self._get_streamflow(sids, start_dt, end_dt, freq, params)

        n_orig = len(sids)
        sids = [s.split("-")[1] for s in qobs]
        if len(sids) != n_orig:
            logger.warning(
                f"Dropped {n_orig - len(sids)} stations since they don't have discharge data"
                + f" from {start_dt} to {end_dt}."
            )
        siteinfo = siteinfo[siteinfo.site_no.isin(sids)]
        if mmd:
            area_sqm = self._drainage_area_sqm(siteinfo, freq)
            ms2mmd = 1000.0 * 24.0 * 3600.0
            try:
                qobs = pd.DataFrame(
                    {c: q / area_sqm.loc[c.split("-")[-1]] * ms2mmd for c, q in qobs.items()}
                )
            except KeyError as ex:
                raise DataNotAvailableError("drainage") from ex

        qobs.attrs, long_names = self._get_attrs(siteinfo, mmd)
        if to_xarray:
            return self._to_xarray(qobs, long_names, mmd)
        return qobs


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

    def get_param_table(self) -> pd.Series:
        """Get the parameter table from the USGS Water Quality Web Service."""
        params = pd.read_html(f"{self.wq_url}/webservices_documentation/")
        params = params[0].iloc[:29].drop(columns="Discussion")
        return params.groupby("REST parameter")["Argument"].apply(",".join)

    def lookup_domain_values(self, endpoint: str) -> list[str]:
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
            raise InputValueError("endpoint", valid_endpoints)
        resp = ar.retrieve_json([f"{self.wq_url}/Codes/{endpoint}?mimeType=json"])
        return [r["value"] for r in resp[0]["codes"]]

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
            raise InputValueError("endpoint", valid_endpoints)
        return f"{self.wq_url}/data/{endpoint}/search"

    def get_json(
        self, endpoint: str, kwds: dict[str, str], request_method: str = "GET"
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
        return geoutils.json2geodf(
            ar.retrieve_json([self._base_url(endpoint)], req_kwds, request_method=request_method)
        )

    def _check_kwds(self, wq_kwds: dict[str, str]) -> None:
        """Check the validity of the Water Quality Web Service keyword arguments."""
        invalids = [k for k in wq_kwds if k not in self.keywords.index]
        if invalids:
            raise InputValueError("wq_kwds", invalids)

    def station_bybbox(
        self, bbox: tuple[float, float, float, float], wq_kwds: dict[str, str] | None
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
        self, lon: float, lat: float, radius: float, wq_kwds: dict[str, str] | None
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

    def get_csv(
        self, endpoint: str, kwds: dict[str, str], request_method: str = "GET"
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
        r = ar.retrieve_binary([self._base_url(endpoint)], req_kwds, request_method=request_method)
        return pd.read_csv(io.BytesIO(r[0]), compression="zip")

    def data_bystation(
        self, station_ids: str | list[str], wq_kwds: dict[str, str] | None
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
            raise InputTypeError("station_ids", valid_type)
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


class WBD(AGRBase):
    """Access Watershed Boundary Dataset (WBD).

    Notes
    -----
    This web service offers Hydrologic Unit (HU) polygon boundaries for
    the United States, Puerto Rico, and the U.S. Virgin Islands.
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
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.
    """

    def __init__(self, layer: str, outfields: str | list[str] = "*", crs: CRSTYPE = 4326):
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
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(ServiceURL().restful.wbd, _layer, outfields, crs)


def huc_wb_full(huc_lvl: int) -> gpd.GeoDataFrame:
    """Get the full watershed boundary for a given HUC level.

    Notes
    -----
    This function is designed for cases where the full watershed boundary is needed
    for a given HUC level. If only a subset of the HUCs is needed, then use
    the ``pygeohydro.WBD`` class. The full dataset is downloaded from the National Maps'
    `WBD staged products <https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/WBD/HU2/Shape/>`__.

    Parameters
    ----------
    huc_lvl : int
        HUC level, must be even numbers between 2 and 16.

    Returns
    -------
    geopandas.GeoDataFrame
        The full watershed boundary for the given HUC level.
    """
    valid_hucs = [2, 4, 6, 8, 10, 12, 14, 16]
    if huc_lvl not in valid_hucs:
        raise InputValueError("huc_lvl", list(map(str, valid_hucs)))

    base_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape"

    urls = [f"{base_url}/WBD_{h2:02}_HU2_Shape.zip" for h2 in range(1, 23)]
    fnames = [Path("cache", Path(url).name) for url in urls]
    _ = ogc.streaming_download(urls, fnames=fnames)  # type: ignore
    keys = (p.stem.split("_")[1] for p in fnames)
    engine = "pyogrio" if importlib.util.find_spec("pyogrios") else "fiona"
    huc = gpd.GeoDataFrame(
        pd.concat(
            (gpd.read_file(f"{p}!Shape/WBDHU{huc_lvl}.shp", engine=engine) for p in fnames),
            keys=keys,
        )
    )
    huc = huc.reset_index().rename(columns={"level_0": "huc2"}).drop(columns="level_1")
    return huc


def irrigation_withdrawals() -> xr.Dataset:
    """Get monthly water use for irrigation at HUC12-level for CONUS.

    Notes
    -----
    Dataset is retrieved from https://doi.org/10.5066/P9FDLY8P.
    """
    sb = ScienceBase()
    item = sb.get_file_urls("5ff7acf4d34ea5387df03d73")
    urls = item.loc[item.index.str.contains(".csv"), "url"]
    resp = ar.retrieve_text(urls.tolist())
    irr = {}
    for name, r in zip(urls.index, resp):
        df = pd.read_csv(io.StringIO(r), usecols=lambda x: "m3" in x or "huc12t" in x)
        df["huc12t"] = df["huc12t"].str.strip("'")
        df = df.rename(columns={"huc12t": "huc12"}).set_index("huc12")
        df = df.rename(columns={c: c[:3].capitalize() for c in df})
        irr[name[-6:-4]] = df.copy()
    ds = xr.Dataset(irr)
    ds = ds.rename({"dim_1": "month"})
    long_names = {
        "GW": "groundwater_withdrawal",
        "SW": "surface_water_withdrawal",
        "TW": "total_withdrawal",
        "CU": "consumptive_use",
    }
    for v, n in long_names.items():
        ds[v].attrs["long_name"] = n
        ds[v].attrs["units"] = "m3"
    ds.attrs["description"] = " ".join(
        (
            "Estimated Monthly Water Use for Irrigation by",
            "12-Digit Hydrologic Unit in the Conterminous United States for 2015",
        )
    )
    ds.attrs["source"] = "https://doi.org/10.5066/P9FDLY8P"
    return ds
