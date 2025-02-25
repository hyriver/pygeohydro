"""Accessing WaterData related APIs."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import cytoolz.curried as tlz
import pandas as pd

import async_retriever as ar
import pygeoutils as geoutils
from pygeohydro.exceptions import InputTypeError, InputValueError, ServiceError

if TYPE_CHECKING:
    import geopandas as gpd

__all__ = ["SensorThings", "WaterQuality"]


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
        resp = cast("list[dict[str, Any]]", resp)
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
        self,
        endpoint: str,
        kwds: dict[str, str],
        request_method: Literal["get", "GET", "post", "POST"] = "GET",
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
        resp = ar.retrieve_json([self._base_url(endpoint)], req_kwds, request_method=request_method)
        resp = cast("list[dict[str, Any]]", resp)
        return geoutils.json2geodf(resp)

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
        self,
        endpoint: str,
        kwds: dict[str, str],
        request_method: Literal["get", "GET", "post", "POST"] = "GET",
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


class SensorThings:
    """Class for interacting with SensorThings API."""

    def __init__(self) -> None:
        self.base_url = "https://api.water.usgs.gov/sta/v1.1/Things"

    @overload
    @staticmethod
    def _get_urls(url: str, kwd: dict[str, Any] | None = ...) -> dict[str, Any]: ...

    @overload
    @staticmethod
    def _get_urls(
        url: list[str], kwd: list[dict[str, Any]] | None = ...
    ) -> list[dict[str, Any]]: ...

    @staticmethod
    def _get_urls(
        url: str | list[str], kwd: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        urls = url if isinstance(url, list) else [url]
        if kwd:
            kwds = kwd if isinstance(kwd, list) else [kwd]
            if len(urls) == 1 and len(urls) != len(kwds):
                urls = urls * len(kwds)
        else:
            kwds = None
        resp = ar.retrieve_json(urls, kwds)
        resp = cast("list[dict[str, Any]]", resp)
        if isinstance(url, str):
            return resp[0]
        return resp

    @staticmethod
    def odata_helper(
        columns: list[str] | None = None,
        conditionals: str | None = None,
        expand: dict[str, dict[str, str]] | None = None,
        max_count: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Generate Odata filters for SensorThings API.

        Parameters
        ----------
        columns : list of str, optional
            Columns to be selected from the database, defaults to ``None``.
        conditionals : str, optional
            Conditionals to be applied to the database, defaults to ``None``.
            Note that the conditionals should have the form of
            ``cond1 operator 'value' and/or cond2 operator 'value``.
            For example:
            ``properties/monitoringLocationType eq 'Stream' and ...``
        expand : dict of dict, optional
            Expand the properties of the selected columns, defaults to ``None``.
            Note that the ``expand`` should have the form of
            ``{Property: {func: value, ...}}``. For example: ``{"Locations":
            {"select": "location", "filter": "ObservedProperty/@iot.id eq '00060'"}}``
        max_count : int, optional
            Maximum number of items to be returned, defaults to ``None``.
        extra_params : dict, optional
            Extra parameters to be added to the Odata filter, defaults to ``None``.

        Returns
        -------
        odata : dict
            Odata filter for the SensorThings API.
        """
        odata = {}
        if columns is not None:
            odata["select"] = ",".join(columns)

        if conditionals is not None:
            odata["filter"] = conditionals

        def _odata(kwds: dict[str, str]) -> str:
            return ";".join(f"${k}={v}" for k, v in kwds.items())

        if expand is not None:
            odata["expand"] = ",".join(f"{func}({_odata(od)})" for func, od in expand.items())

        if max_count is not None:
            odata["top"] = max_count

        if extra_params is not None:
            odata.update(extra_params)
        return odata

    def query_byodata(
        self, odata: dict[str, Any], outformat: str = "json"
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """Query the SensorThings API by Odata filter.

        Parameters
        ----------
        odata : str
            Odata filter for the SensorThings API.
        outformat : str, optional
            Format of the response, defaults to ``json``.
            Valid values are ``json`` and ``geojson``.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            Requested data.
        """
        valid_formats = ["json", "geojson"]
        if outformat not in valid_formats:
            raise InputValueError("format", valid_formats)

        kwds = odata.copy()
        if outformat == "geojson":
            kwds.update({"resultFormat": "GeoJSON"})

        kwds = {"params": {f"${k}": v for k, v in kwds.items()}}
        resp = self._get_urls(self.base_url, kwds)

        if "message" in resp:
            raise ServiceError(resp["message"])

        if outformat == "json":
            data = resp["value"]
            data = cast("list[dict[str, Any]]", data)
            while "@iot.nextLink" in resp:
                resp = self._get_urls(resp["@iot.nextLink"])
                data.extend(resp["value"])
            return pd.json_normalize(data)
        return geoutils.json2geodf(resp)

    def sensor_info(self, sensor_ids: str | list[str]) -> pd.DataFrame:
        """Query the SensorThings API by a sensor ID.

        Parameters
        ----------
        sensor_ids : str or list of str
            A single or list of sensor IDs, e.g., ``USGS-09380000``.

        Returns
        -------
        pandas.DataFrame
            Requested sensor data.
        """
        sensor_ids = [sensor_ids] if isinstance(sensor_ids, str) else sensor_ids
        urls = [f"{self.base_url}('{i}')" for i in sensor_ids]
        data = pd.json_normalize(self._get_urls(urls))
        columns = data.columns[data.columns.str.endswith("Link")]
        return data.drop(columns=columns)  # pyright: ignore[reportCallIssue,reportArgumentType]

    def sensor_property(self, sensor_property: str, sensor_ids: str | list[str]) -> pd.DataFrame:
        """Query a sensor property.

        Parameters
        ----------
        sensor_property : str or list of str
            A sensor property, Valid properties are ``Datastreams``,
            ``MultiDatastreams``, ``Locations``, ``HistoricalLocations``,
            ``TaskingCapabilities``.
        sensor_ids : str or list of str
            A single or list of sensor IDs, e.g., ``USGS-09380000``.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the requested property.
        """
        sensor_ids = [sensor_ids] if isinstance(sensor_ids, str) else sensor_ids
        urls = [f"{self.base_url}('{i}')" for i in sensor_ids]
        resp = self._get_urls(urls)
        links = tlz.merge_with(
            list,
            (
                {p.split("@")[0]: r.pop(p, None) for p in list(r) if p.endswith("navigationLink")}
                for r in resp
            ),
        )
        links = cast("dict[str, list[str]]", links)

        if sensor_property not in links:
            raise InputValueError("properties", list(links))
        resp = self._get_urls(links[sensor_property])
        return pd.concat(pd.json_normalize(r["value"]) for r in resp).reset_index(drop=True)
