"""Accessing NWIS."""

# pyright: reportArgumentType=false,reportCallIssue=false,reportReturnType=false
from __future__ import annotations

import contextlib
import itertools
import re
import warnings
from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
    overload,
)

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import async_retriever as ar
import pynhd
from pygeohydro.exceptions import (
    DataNotAvailableError,
    InputTypeError,
    InputValueError,
    ZeroMatchedError,
)
from pygeoogc import ServiceURL
from pygeoogc import utils as ogc_utils
from pynhd import NLDI

try:
    from pandas.errors import IntCastingNaNError
except ImportError:
    IntCastingNaNError = ValueError

YEAR_END = "Y" if int(pd.__version__.split(".")[0]) < 2 else "YE"
T_FMT = "%Y-%m-%d"
__all__ = ["NWIS", "streamflow_fillna"]

if TYPE_CHECKING:
    ArrayLike = TypeVar("ArrayLike", pd.Series, pd.DataFrame, xr.DataArray)


def streamflow_fillna(streamflow: ArrayLike, missing_max: int = 5) -> ArrayLike:
    """Fill missing data (NAN) in daily streamflow observations.

    It drops stations with more than ``missing_max`` days missing data
    per year. Missing data in the remaining stations, are filled with
    day-of-year average over the entire dataset.

    Parameters
    ----------
    streamflow : xarray.DataArray or pandas.DataFrame or pandas.Series
        Daily streamflow observations with at least 10 years of daily data.
    missing_max : int
        Maximum allowed number of missing daily data per year for filling,
        defaults to 5.

    Returns
    -------
    xarray.DataArray or pandas.DataFrame or pandas.Series
        Streamflow observations with missing data filled for stations with
        less than ``missing_max`` days of missing data.
    """
    if isinstance(streamflow, xr.DataArray):
        df = streamflow.to_pandas()
    elif isinstance(streamflow, (pd.DataFrame, pd.Series)):
        df = streamflow.copy()
    else:
        raise InputTypeError("streamflow", "xarray.DataArray, pandas.DataFrame, or pandas.Series")

    if isinstance(df, pd.Series):
        df = df.to_frame(df.name or "0" * 8)

    df = cast("pd.DataFrame", df)
    df.columns = df.columns.astype(str)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index).date)
    if df.index.year.unique().size < 10:
        raise InputTypeError("streamflow", "array with at least 10 years of daily data")

    df[df < 0] = np.nan
    s_nan = pd.DataFrame.from_dict(
        {yr: q.isna().sum() for yr, q in df.resample(YEAR_END)}, orient="index"
    )
    if np.all(s_nan == 0):
        return streamflow

    s_fill = s_nan[s_nan <= missing_max].dropna(axis=1).columns.tolist()
    if not s_fill:
        msg = f"Found no column with less than {missing_max} days of missing data."
        raise ValueError(msg)

    df = df[s_fill].copy()
    df["dayofyear"] = pd.to_datetime(df.index).dayofyear
    daymean = df.groupby("dayofyear").mean().to_dict()

    fillval = pd.DataFrame({s: df["dayofyear"].map(daymean[s]) for s in s_fill})
    df = df[s_fill].fillna(fillval)

    if isinstance(streamflow, pd.Series):
        return df.squeeze()

    if isinstance(streamflow, pd.DataFrame):
        return df

    return xr.DataArray(  # pyright: ignore[reportGeneralTypeIssues]
        df,
        coords={
            "time": df.index.to_numpy("datetime64[ns]"),
            "station_id": s_fill,
        },
        name=streamflow.name or "discharge",
    )


class NWIS:
    """Access NWIS web service.

    Notes
    -----
    More information about query parameters and codes that NWIS accepts
    can be found at its help
    `webpage <https://help.waterdata.usgs.gov/codes-and-parameters>`__.
    """

    url: str = ServiceURL().restful.nwis

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
                [{"params": p | {"format": "rdb"}} for p in payloads],
            )
        except ar.ServiceError as ex:
            raise ZeroMatchedError(ogc_utils.check_response(str(ex))) from ex

        with contextlib.suppress(StopIteration):
            zero_len = next(filter(lambda x: len(x) == 0, resp), None)
            if zero_len is not None:
                raise ZeroMatchedError

            not_found = next(filter(lambda x: not x.startswith("#"), resp), None)
            if not_found is not None:
                msg = re.findall("<p>(.*?)</p>", not_found)[1].rsplit(">", 1)[1]
                msg = f"Server error message:\n{msg}"
                raise ZeroMatchedError(msg)

        data = [
            line.split("\t") for r in resp for line in r.splitlines() if not line.startswith("#")
        ]
        if not data:
            raise ZeroMatchedError

        rdb_df = pd.DataFrame.from_dict(dict(zip(data[0], d)) for d in data[2:])  # pyright: ignore[reportArgumentType]
        if "agency_cd" in rdb_df:
            rdb_df = rdb_df[~rdb_df["agency_cd"].str.contains("agency_cd|5s")].copy()
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
            invalid_keys = f"query keys ({', '.join(not_valid)})"
            raise InputValueError(invalid_keys, valid_query_keys)

        _queries = queries.copy()
        if expanded:
            _ = [
                q.pop(k)
                for k in ("outputDataTypeCd", "outputDataType", "seriesCatalogOutput")
                for q in _queries
                if k in q
            ]
            output_type = {"siteOutput": "expanded"}
        else:
            output_type = {"siteOutput": "basic"}

        return [query | output_type | {"format": "rdb"} for query in _queries]

    @staticmethod
    def _nhd_info(site_ids: list[str]) -> pd.DataFrame:
        """Get drainage area of the stations from NHDPlus."""
        site_list = [f"USGS-{s}" for s in site_ids]
        area = NLDI().getfeature_byid("nwissite", site_list)
        try:
            area["comid"] = area["comid"].astype("int32")
        except (TypeError, IntCastingNaNError):
            area["comid"] = area["comid"].astype("Int32")
        nhd_area = pynhd.streamcat("fert", comids=area["comid"].dropna().to_list(), area_sqkm=True)
        area = area.merge(
            nhd_area[["comid", "wsareasqkm"]], left_on="comid", right_on="comid", how="left"
        )
        area["identifier"] = area["identifier"].str.replace("USGS-", "")
        return (
            area[["identifier", "comid", "reachcode", "measure", "wsareasqkm"]]
            .copy()
            .rename(
                columns={
                    "identifier": "site_no",
                    "comid": "nhd_id",
                    "reachcode": "nhd_reachcode",
                    "measure": "nhd_measure",
                    "wsareasqkm": "nhd_areasqkm",
                }
            )
        )

    @classmethod
    def get_info(
        cls,
        queries: dict[str, str] | list[dict[str, str]],
        expanded: bool = False,
        fix_names: bool = True,
        nhd_info: bool = False,
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
        nhd_info : bool, optional
            If ``True``, get NHD information for each site, defaults to ``False``.
            This will add four new columns: ``nhd_comid``, ``nhd_areasqkm``,
            ``nhd_reachcode``, and ``nhd_measure``. Where ``nhd_id`` is the NHD
            COMID of the flowline that the site is located in, ``nhd_reachcode``
            is the NHD Reach Code that the site is located in, and ``nhd_measure``
            is the measure along the flowline that the site is located at.

        Returns
        -------
        geopandas.GeoDataFrame
            A correctly typed ``GeoDataFrame`` containing site(s) information.
        """
        queries = [queries] if isinstance(queries, dict) else queries

        payloads = cls._validate_usgs_queries(queries, False)
        sites = cls.retrieve_rdb(f"{cls.url}/site", payloads)

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
            payloads = cls._validate_usgs_queries(queries, True)
            sites = sites.merge(
                cls.retrieve_rdb(f"{cls.url}/site", payloads),
                on="site_no",
                how="outer",
                suffixes=("", "_overlap"),
            )
            sites = sites.filter(regex="^(?!.*_overlap)")
            numeric_cols += ["drain_area_va", "contrib_drain_area_va"]

        with contextlib.suppress(KeyError):
            sites["begin_date"] = pd.to_datetime(
                sites["begin_date"], errors="coerce", yearfirst=True
            ).fillna(sites["begin_date"])
            sites["end_date"] = pd.to_datetime(
                sites["end_date"], errors="coerce", yearfirst=True
            ).fillna(sites["end_date"])

        if nhd_info:
            nhd = cls._nhd_info(sites["site_no"].to_list())
            sites = pd.merge(sites, nhd, left_on="site_no", right_on="site_no", how="left")

        urls = [
            "/".join(
                (
                    "https://gist.githubusercontent.com/cheginit",
                    "2dbc4b9c096f19089dbadb522821e8f3/raw/hcdn_2009_station_ids.txt",
                )
            )
        ]
        resp = ar.retrieve_text(urls)
        sites["hcdn_2009"] = sites["site_no"].isin(resp[0].split(","))

        if "count_nu" in sites:
            numeric_cols.append("count_nu")
        sites[numeric_cols] = sites[numeric_cols].apply(pd.to_numeric, errors="coerce")

        return gpd.GeoDataFrame(
            sites,
            geometry=gpd.points_from_xy(sites["dec_long_va"], sites["dec_lat_va"]),
            crs=4326,
        )

    @classmethod
    def get_parameter_codes(cls, keyword: str) -> pd.DataFrame:
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
        >>> codes.loc[codes.parameter_cd == "00060", "parm_nm"].iloc[0]
        'Discharge, cubic feet per second'
        """
        url = "https://help.waterdata.usgs.gov/code/parameter_cd_nm_query"
        kwds = [{"parm_nm_cd": keyword, "fmt": "rdb"}]
        return cls.retrieve_rdb(url, kwds)

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
            attr_df["begin_date"] = attr_df["begin_date"].dt.strftime(T_FMT)
            attr_df["end_date"] = attr_df["end_date"].dt.strftime(T_FMT)
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

    @classmethod
    def _drainage_area_sqm(cls, siteinfo: pd.DataFrame, freq: str) -> pd.Series:
        """Get drainage area of the stations."""
        if "nhd_areasqkm" not in siteinfo:
            area = cls._nhd_info(siteinfo["site_no"].to_list())
            area = area[["site_no", "nhd_areasqkm"]].copy()
        else:
            area = siteinfo[["site_no", "nhd_areasqkm"]].copy()  # pyright: ignore[reportGeneralTypeIssues]
        if area["nhd_areasqkm"].isna().any():  # pyright: ignore[reportGeneralTypeIssues]
            sids = area[area["nhd_areasqkm"].isna()].site_no
            queries = [
                {
                    "parameterCd": "00060",
                    "siteStatus": "all",
                    "outputDataTypeCd": freq,
                    "sites": ",".join(s),
                }
                for s in tlz.partition_all(1500, sids)
            ]
            info = cls.get_info(queries, expanded=True)

            def get_idx(ids: list[str]) -> tuple[pd.Index, pd.Index]:
                return info.site_no.isin(ids), area.site_no.isin(ids)

            i_idx, a_idx = get_idx(sids)
            # Drainage areas in info are in sq mi and should be converted to sq km
            area.loc[a_idx, "nhd_areasqkm"] = info.loc[i_idx, "contrib_drain_area_va"] * 0.38610
            if area["nhd_areasqkm"].isna().any():  # pyright: ignore[reportGeneralTypeIssues]
                sids = area[area["nhd_areasqkm"].isna()].site_no
                i_idx, a_idx = get_idx(sids)
                area.loc[a_idx, "nhd_areasqkm"] = info.loc[i_idx, "drain_area_va"] * 0.38610

        if area["nhd_areasqkm"].isna().all():  # pyright: ignore[reportGeneralTypeIssues]
            raise DataNotAvailableError("drainage")
        return area.set_index("site_no").nhd_areasqkm * 1e6

    @classmethod
    def _get_streamflow(
        cls,
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
            }
            | kwargs
            for s in tlz.partition_all(1500, sids)
        ]
        resp = ar.retrieve_json(
            [f"{cls.url}/{freq}"] * len(payloads), [{"params": p} for p in payloads]
        )
        resp = cast("list[dict[str, Any]]", resp)

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
            time_zone = {
                "CST": "US/Central",
                "MST": "US/Mountain",
                "PST": "US/Pacific",
                "EST": "US/Eastern",
            }.get(
                tz["defaultTimeZone"]["zoneAbbreviation"],
                tz["defaultTimeZone"]["zoneAbbreviation"],
            )
            discharge.index = pd.DatetimeIndex(
                pd.Timestamp(i, tz=time_zone) for i in discharge.index
            ).tz_convert("UTC")
            discharge.columns = [col]
            return discharge

        qobs = pd.concat(itertools.starmap(to_df, r_ts.items()), axis=1)
        if len(qobs) == 0:
            raise DataNotAvailableError("discharge")
        qobs[qobs.le(0)] = np.nan
        # Convert cfs to cms
        return qobs * np.float_power(0.3048, 3)

    @classmethod
    @overload
    def get_streamflow(
        cls,
        station_ids: Sequence[str] | str,
        dates: tuple[str, str],
        freq: str = "dv",
        mmd: bool = False,
        *,
        to_xarray: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @classmethod
    @overload
    def get_streamflow(
        cls,
        station_ids: Sequence[str] | str,
        dates: tuple[str, str],
        freq: str = "dv",
        mmd: bool = False,
        *,
        to_xarray: Literal[True],
    ) -> xr.Dataset: ...

    @classmethod
    def get_streamflow(
        cls,
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

        sids, start, end = cls._check_inputs(station_ids, dates, utc)

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
            siteinfo = cls.get_info(queries)
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
        qobs = cls._get_streamflow(sids, start_dt, end_dt, freq, params)

        n_orig = len(sids)
        sids = [s.split("-")[1] for s in qobs]
        if len(sids) != n_orig:
            warnings.warn(
                (
                    f"Dropped {n_orig - len(sids)} stations since they don't have discharge data"
                    f" from {start_dt} to {end_dt}."
                ),
                UserWarning,
                stacklevel=2,
            )
        siteinfo = siteinfo[siteinfo.site_no.isin(sids)]
        if mmd:
            area_sqm = cls._drainage_area_sqm(siteinfo, freq)
            ms2mmd = 1000.0 * 24.0 * 3600.0
            try:
                qobs = pd.DataFrame(
                    {c: q / area_sqm.loc[c.split("-")[-1]] * ms2mmd for c, q in qobs.items()}
                )
            except KeyError as ex:
                raise DataNotAvailableError("drainage") from ex

        qobs.attrs, long_names = cls._get_attrs(siteinfo, mmd)
        if to_xarray:
            return cls._to_xarray(qobs, long_names, mmd)
        return qobs
