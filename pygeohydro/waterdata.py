"""Accessing data from the supported databases through their APIs."""
import contextlib
import io
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import async_retriever as ar
import cytoolz as tlz
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoogc as ogc
import pygeoutils as geoutils
import xarray as xr
from pygeoogc import ServiceURL
from pygeoogc import ZeroMatched as ZeroMatchedOGC
from pygeoogc import utils as ogc_utils
from pynhd import WaterData

from .exceptions import DataNotAvailable, InvalidInputType, InvalidInputValue, ZeroMatched
from .helpers import logger

DEF_CRS = "epsg:4326"
T_FMT = "%Y-%m-%d"
EXPIRE = -1


class NWIS:
    """Access NWIS web service."""

    def __init__(self) -> None:
        self.url = ServiceURL().restful.nwis

    def retrieve_rdb(self, url: str, payloads: List[Dict[str, str]]) -> pd.DataFrame:
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
                [url] * len(payloads), [{"params": {**p, "format": "rdb"}} for p in payloads]
            )
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

    def get_info(
        self, queries: Union[Dict[str, str], List[Dict[str, str]]], expanded: bool = False
    ) -> gpd.GeoDataFrame:
        """Send multiple queries to USGS Site Web Service.

        Parameters
        ----------
        queries : dict or list of dict
            A single or a list of valid queries.
        expanded : bool, optional
            Whether to get expanded sit information for example drainage area,
            default to False.

        Returns
        -------
        geopandas.GeoDataFrame
            A correctly typed ``GeoDataFrame`` containing site(s) information.
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

        with contextlib.suppress(KeyError):
            sites["begin_date"] = pd.to_datetime(sites["begin_date"])
            sites["end_date"] = pd.to_datetime(sites["end_date"])

        gii = WaterData("gagesii", DEF_CRS)

        def _get_hcdn(site_no: str) -> Tuple[float, Optional[bool]]:
            try:
                gage = gii.byid("staid", site_no)
                return gage.drain_sqkm.iloc[0], len(gage.hcdn_2009.iloc[0]) > 0  # noqa: TC300
            except (AttributeError, KeyError, ZeroMatchedOGC):
                return np.nan, None

        sites["drain_sqkm"], sites["hcdn_2009"] = zip(*[_get_hcdn(n) for n in sites.site_no])

        float_cols += ["drain_sqkm"]
        sites[float_cols] = sites[float_cols].apply(pd.to_numeric, errors="coerce")

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
    def _get_attrs(siteinfo: pd.DataFrame, mmd: bool) -> Tuple[Dict[str, Any], Dict[str, str]]:
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
        station_ids: Union[Sequence[str], str], dates: Tuple[str, str], utc: Optional[bool]
    ) -> Tuple[List[str], pd.Timestamp, pd.Timestamp]:
        """Validate inputs."""
        if not isinstance(station_ids, (str, Sequence, Iterable)):
            raise InvalidInputType("ids", "str or list of str")

        sids = [station_ids] if isinstance(station_ids, str) else list(set(station_ids))

        if not isinstance(dates, tuple) or len(dates) != 2:
            raise InvalidInputType("dates", "tuple", "(start, end)")

        start = pd.to_datetime(dates[0], utc=utc)
        end = pd.to_datetime(dates[1], utc=utc)
        return sids, start, end

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

            def get_idx(ids: List[str]) -> Tuple[pd.Index, pd.Index]:
                return info.site_no.isin(ids), area.site_no.isin(ids)

            i_idx, a_idx = get_idx(sids)
            # Drainage areas in info are in sq mi and should be converted to sq km
            area.loc[a_idx, "drain_sqkm"] = info.loc[i_idx, "contrib_drain_area_va"] * 0.38610
            if area["drain_sqkm"].isna().any():
                sids = area[area["drain_sqkm"].isna()].site_no
                i_idx, a_idx = get_idx(sids)
                area.loc[a_idx, "drain_sqkm"] = info.loc[i_idx, "drain_area_va"] * 0.38610

        if area["drain_sqkm"].isna().all():
            raise DataNotAvailable("drainage")
        return area.set_index("site_no").drain_sqkm * 1e6

    def _get_streamflow(
        self, sids: Sequence[str], start_dt: str, end_dt: str, freq: str, kwargs: Dict[str, str]
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

        def get_site_id(site_cd: Dict[str, str]) -> str:
            """Get site id."""
            return f"{site_cd['agencyCode']}-{site_cd['value']}"

        r_ts = {
            get_site_id(t["sourceInfo"]["siteCode"][0]): t["values"][0]["value"]
            for r in resp
            for t in r["value"]["timeSeries"]
            if len(t["values"][0]["value"]) > 0
        }
        if len(r_ts) == 0:
            raise DataNotAvailable("discharge")

        def to_df(col: str, values: Dict[str, Any]) -> pd.DataFrame:
            discharge = pd.DataFrame.from_records(
                values, exclude=["qualifiers"], index=["dateTime"]
            )
            discharge.index = pd.to_datetime(discharge.index, infer_datetime_format=True)
            if discharge.index.tz is None:
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

                discharge.index = discharge.index.tz_localize(time_zone)
            discharge.index = discharge.index.tz_convert("UTC")
            discharge.columns = [col]
            return discharge

        qobs = pd.concat([to_df(s, t) for s, t in r_ts.items()], axis=1)
        # Convert cfs to cms
        return qobs.astype("float64") * 0.028316846592

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
        if len(sids) == 0:
            raise DataNotAvailable("discharge")

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
                    {c: q / area_sqm.loc[c.split("-")[-1]] * ms2mmd for c, q in qobs.iteritems()}
                )
            except KeyError as ex:
                raise DataNotAvailable("drainage") from ex

        qobs.attrs, long_names = self._get_attrs(siteinfo, mmd)
        if to_xarray:
            return self._to_xarray(qobs, long_names, mmd)
        return qobs


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

    sites = sites.groupby("site_no")[cols_old[1:]].agg(set).reset_index()
    sites = sites.rename(columns=dict(zip(cols_old, cols_new)))

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
            raise InvalidInputValue("endpoint", valid_endpoints)
        return f"{self.wq_url}/data/{endpoint}/search"

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
        return geoutils.json2geodf(
            ar.retrieve_json([self._base_url(endpoint)], req_kwds, request_method=request_method)
        )

    def _check_kwds(self, wq_kwds: Dict[str, str]) -> None:
        """Check the validity of the Water Quality Web Service keyword arguments."""
        invalids = [k for k in wq_kwds if k not in self.keywords.index]
        if len(invalids) > 0:
            raise InvalidInputValue("wq_kwds", invalids)

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
        r = ar.retrieve_binary([self._base_url(endpoint)], req_kwds, request_method=request_method)
        return pd.read_csv(io.BytesIO(r[0]), compression="zip")

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
