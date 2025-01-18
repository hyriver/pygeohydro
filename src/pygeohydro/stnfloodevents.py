"""Access USGS Short-Term Network (STN) via Restful API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd

import async_retriever as ar
from pygeohydro.exceptions import InputValueError
from pygeoogc import ServiceURL

if TYPE_CHECKING:
    from pyproj import CRS

    CRSType = int | str | CRS

__all__ = ["STNFloodEventData", "stn_flood_event"]


class STNFloodEventData:
    """Client for STN Flood Event Data's RESTFUL Service API.

    Advantages of using this client are:

    - The user does not need to know the details of RESTFUL in
      general and of this API specifically.
    - Parses the data and returns Python objects
      (e.g., pandas.DataFrame, geopandas.GeoDataFrame) instead of JSON.
    - Convenience functions are offered for data dictionaries.
    - Geo-references the data where applicable.

    Attributes
    ----------
    service_url : str
        The service url of the STN Flood Event Data RESTFUL Service API.
    data_dictionary_url : str
        The data dictionary url of the STN Flood Event Data RESTFUL Service API.
    service_crs : int
        The CRS of the data from the service which is ``EPSG:4326``.
    instruments_query_params : set
        The accepted query parameters for the instruments data type.
        Accepted values are ``SensorType``, ``CurrentStatus``, ``States``,
        ``Event``, ``County``, ``DeploymentType``, ``EventType``,
        ``EventStatus``, and ``CollectionCondition``.
    peaks_query_params : set
        The accepted query parameters for the peaks data type.
        Accepted values are ``EndDate``, ``States``, ``Event``, ``StartDate``,
        ``County``, ``EventType``, and ``EventStatus``.
    hwms_query_params : set
        The accepted query parameters for the hwms data type.
        Accepted values are ``EndDate``, ``States``, ``Event``, ``StartDate``,
        ``County``, ``EventType``, and ``EventStatus``.
    sites_query_params : set
        The accepted query parameters for the sites data type.
        Accepted values are ``OPDefined``, ``HousingTypeOne``, ``NetworkName``,
        ``HousingTypeSeven``, ``RDGOnly``, ``HWMOnly``, ``Event``,
        ``SensorOnly``, ``State``, ``SensorType``, and ``HWMSurveyed``.


    Notes
    -----
    Point data from the service is assumed to be in the WGS84
    coordinate reference system (``EPSG:4326``).

    References
    ----------
    * `USGS Short-Term Network (STN) <https://stn.wim.usgs.gov/STNWeb/#/>`__
    * `All Sensors API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/Sensor/AllSensors>`__
    * `All Peak Summary API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/PeakSummary/AllPeakSummaries>`__
    * `All HWM API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/HWM/AllHWMs>`__
    * `All Sites API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/Site/AllSites>`__
    * `USGS Flood Event Viewer: Providing Hurricane and Flood Response Data <https://www.usgs.gov/mission-areas/water-resources/science/usgs-flood-event-viewer-providing-hurricane-and-flood>`__
    * `A USGS guide for finding and interpreting high-water marks <https://www.usgs.gov/media/videos/a-usgs-guide-finding-and-interpreting-high-water-marks>`__
    * `High-Water Marks and Flooding <https://www.usgs.gov/special-topics/water-science-school/science/high-water-marks-and-flooding>`__
    * `Identifying and preserving high-water mark data <https://doi.org/10.3133/tm3A24>`__
    """

    # Per Athena Clark, Lauren Privette, and Hans Vargas at USGS
    # this is the CRS used for visualization on STN front-end.
    service_crs: ClassVar[int] = 4326
    service_url: ClassVar[str] = ServiceURL().restful.stnflood
    instruments_query_params: ClassVar[set[str]] = {
        "Event",
        "EventType",
        "EventStatus",
        "States",
        "County",
        "CurrentStatus",
        "CollectionCondition",
        "SensorType",
        "DeploymentType",
    }
    peaks_query_params: ClassVar[set[str]] = {
        "Event",
        "EventType",
        "EventStatus",
        "States",
        "County",
        "StartDate",
        "EndDate",
    }
    hwms_query_params: ClassVar[set[str]] = {
        "Event",
        "EventType",
        "EventStatus",
        "States",
        "County",
        "StartDate",
        "EndDate",
    }
    sites_query_params: ClassVar[set[str]] = {
        "Event",
        "State",
        "SensorType",
        "NetworkName",
        "OPDefined",
        "HWMOnly",
        "HWMSurveyed",
        "SensorOnly",
        "RDGOnly",
        "HousingTypeOne",
        "HousingTypeSeven",
    }

    @staticmethod
    def _geopandify(
        input_list: list[dict[str, Any]],
        x_col: str,
        y_col: str,
        crs: CRSType | None,
        service_crs: CRSType,
    ) -> gpd.GeoDataFrame:
        """Georeference a list of dictionaries to a GeoDataFrame.

        Parameters
        ----------
        input_list : list of dict
            The list of dictionaries to be converted to a geodataframe.
        x_col : str
            The name of the column containing the x-coordinate.
        y_col : str
            The name of the column containing the y-coordinate.
        crs : int, str, or pyproj.CRS
            Desired Coordinate reference system (CRS) of output.
            Only used for GeoDataFrames outputs.
        service_crs : int, str, or pyproj.CRS
            The coordinate reference system of the data from the service.

        Returns
        -------
        geopandas.GeoDataFrame
            The geo-referenced GeoDataFrame.
        """
        df = pd.DataFrame(input_list)

        if crs is None:
            crs = service_crs

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[x_col], df[y_col], crs=service_crs),
        )
        gdf = cast("gpd.GeoDataFrame", gdf.to_crs(crs))
        return gdf

    @staticmethod
    def _delist_dict(d: dict[str, list[float] | float]) -> dict[str, float]:
        """De-lists all unit length lists in a dictionary."""

        def delist(x: list[float] | float) -> float:
            if not isinstance(x, list):
                return x
            if len(x) == 1:
                return x[0]
            return np.nan

        return {k: delist(v) for k, v in d.items()}

    @classmethod
    @overload
    def get_all_data(
        cls,
        data_type: str,
        *,
        as_list: Literal[False] = False,
        crs: CRSType = 4326,
        async_retriever_kwargs: dict[str, Any] | None = None,
    ) -> gpd.GeoDataFrame | pd.DataFrame: ...

    @classmethod
    @overload
    def get_all_data(
        cls,
        data_type: str,
        *,
        as_list: Literal[True],
        crs: CRSType = 4326,
        async_retriever_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    @classmethod
    def get_all_data(
        cls,
        data_type: str,
        as_list: bool = False,
        crs: CRSType = 4326,
        async_retriever_kwargs: dict[str, Any] | None = None,
    ) -> gpd.GeoDataFrame | pd.DataFrame | list[dict[str, Any]]:
        """Retrieve all data from the STN Flood Event Data API.

        Parameters
        ----------
        data_type : str
            The data source from STN Flood Event Data API.
            It can be ``instruments``, ``peaks``, ``hwms``, or ``sites``.
        as_list : bool, optional
            If True, return the data as a list, defaults to False.
        crs : int, str, or pyproj.CRS, optional
            Desired Coordinate reference system (CRS) of output.
            Only used for GeoDataFrames with ``hwms`` and ``sites`` data types.
        async_retriever_kwargs : dict, optional
            Additional keyword arguments to pass to
            ``async_retriever.retrieve_json()``. The ``url`` and ``request_kwds``
            options are already set.

        Returns
        -------
        geopandas.GeoDataFrame or pandas.DataFrame or list of dict
            The retrieved data as a GeoDataFrame, DataFrame, or a list of dictionaries.

        Raises
        ------
        InputValueError
            If the input data_type is not one of
            ``instruments``, ``peaks``, ``hwms``, or ``sites``

        See Also
        --------
        :meth:`~get_filtered_data` : Retrieves filtered data for a given data type.
        :meth:`~data_dictionary` : Retrieves the data dictionary for a given data type.

        Notes
        -----
        Notice schema differences between the data dictionaries, filtered data
        queries, and all data queries. This is a known issue and is being addressed
        by USGS.

        Examples
        --------
        >>> from pygeohydro.stnfloodevents import STNFloodEventData
        >>> data = STNFloodEventData.get_all_data(data_type="instruments")
        >>> data.shape[1]
        18
        >>> data.columns
        Index(['instrument_id', 'sensor_type_id', 'deployment_type_id',
               'location_description', 'serial_number', 'interval', 'site_id',
               'event_id', 'inst_collection_id', 'housing_type_id', 'sensor_brand_id',
               'vented', 'instrument_status', 'data_files', 'files', 'last_updated',
               'last_updated_by', 'housing_serial_number'],
               dtype='object')
        """
        dtype_dict = {
            "instruments": "Instruments.json",
            "peaks": "PeakSummaries.json",
            "hwms": "HWMs.json",
            "sites": "Sites.json",
        }

        endpoint = dtype_dict.get(data_type)
        if endpoint is None:
            raise InputValueError(data_type, list(dtype_dict))

        if async_retriever_kwargs is None:
            async_retriever_kwargs = {}
        else:
            _ = async_retriever_kwargs.pop("url", None)

        resp = ar.retrieve_json([f"{cls.service_url}/{endpoint}"], **async_retriever_kwargs)
        data = [cls._delist_dict(d) for d in resp[0]]  # pyright: ignore[reportArgumentType]

        if as_list:
            return data

        _xy_cols = {
            "instruments": None,
            "peaks": None,
            "hwms": ("longitude_dd", "latitude_dd"),
            "sites": ("longitude_dd", "latitude_dd"),
        }
        xy_cols = _xy_cols[data_type]
        if xy_cols is None:
            return pd.DataFrame(data)

        x_col, y_col = xy_cols
        return cls._geopandify(data, x_col, y_col, crs, cls.service_crs)

    @classmethod
    @overload
    def get_filtered_data(
        cls,
        data_type: str,
        query_params: dict[str, Any] | None = None,
        *,
        as_list: Literal[False] = False,
        crs: CRSType = 4326,
        async_retriever_kwargs: dict[str, Any] | None = None,
    ) -> gpd.GeoDataFrame | pd.DataFrame: ...

    @classmethod
    @overload
    def get_filtered_data(
        cls,
        data_type: str,
        query_params: dict[str, Any] | None = None,
        *,
        as_list: Literal[True],
        crs: CRSType = 4326,
        async_retriever_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    @classmethod
    def get_filtered_data(
        cls,
        data_type: str,
        query_params: dict[str, Any] | None = None,
        as_list: bool = False,
        crs: CRSType = 4326,
        async_retriever_kwargs: dict[str, Any] | None = None,
    ) -> gpd.GeoDataFrame | pd.DataFrame | list[dict[str, Any]]:
        """Retrieve filtered data from the STN Flood Event Data API.

        Parameters
        ----------
        data_type : str
            The data source from STN Flood Event Data API.
            It can be ``instruments``, ``peaks``, ``hwms``, or ``sites``.
        query_params : dict, optional
            RESTFUL API query parameters. For accepted values, see
            the STNFloodEventData class attributes :attr:`~instruments_query_params`,
            :attr:`~peaks_query_params`, :attr:`~hwms_query_params`, and
            :attr:`~sites_query_params` for available values.

            Also, see the API documentation for each data type for more information:
                - `instruments <https://stn.wim.usgs.gov/STNServices/Documentation/Sensor/FilteredSensors>`__
                - `peaks <https://stn.wim.usgs.gov/STNServices/Documentation/PeakSummary/FilteredPeakSummaries>`__
                - `hwms <https://stn.wim.usgs.gov/STNServices/Documentation/HWM/FilteredHWMs>`__
                - `sites <https://stn.wim.usgs.gov/STNServices/Documentation/Site/FilteredSites>`__

        as_list : bool, optional
            If True, return the data as a list, defaults to False.
        crs : int, str, or pyproj.CRS, optional
            Desired Coordinate reference system (CRS) of output.
            Only used for GeoDataFrames outputs.
        async_retriever_kwargs : dict, optional
            Additional keyword arguments to pass to
            ``async_retriever.retrieve_json()``. The ``url`` and ``request_kwds``
            options are already set.

        Returns
        -------
        geopandas.GeoDataFrame or pandas.DataFrame or list of dict
            The retrieved data as a GeoDataFrame, DataFrame, or a
            list of dictionaries.

        Raises
        ------
        InputValueError
            If the input data_type is not one of
            ``instruments``, ``peaks``, ``hwms``, or ``sites``
        InputValueError
            If any of the input query_params are not in accepted
            parameters (See :attr:`~instruments_query_params`,
            :attr:`~peaks_query_params`, :attr:`~hwms_query_params`,
            or :attr:`~sites_query_params`).

        See Also
        --------
        :meth:`~get_all_data` : Retrieves all data for a given data type.
        :meth:`~data_dictionary` : Retrieves the data dictionary for a given data type.

        Notes
        -----
        Notice schema differences between the data dictionaries,
        filtered data queries, and all data queries. This is a known
        issue and is being addressed by USGS.

        Examples
        --------
        >>> from pygeohydro.stnfloodevents import STNFloodEventData
        >>> query_params = {"States": "SC, CA"}
        >>> data = STNFloodEventData.get_filtered_data(data_type="instruments", query_params=query_params)
        >>> data.shape[1]
        34
        >>> data.columns
        Index(['sensorType', 'deploymentType', 'eventName', 'collectionCondition',
            'housingType', 'sensorBrand', 'statusId', 'timeStamp', 'site_no',
            'latitude', 'longitude', 'siteDescription', 'networkNames', 'stateName',
            'countyName', 'siteWaterbody', 'siteHDatum', 'sitePriorityName',
            'siteZone', 'siteHCollectMethod', 'sitePermHousing', 'instrument_id',
            'sensor_type_id', 'deployment_type_id', 'location_description',
            'serial_number', 'housing_serial_number', 'interval', 'site_id',
            'vented', 'instrument_status', 'data_files', 'files', 'geometry'],
            dtype='object')
        """
        endpoint_dict = {
            "instruments": "Instruments/FilteredInstruments.json",
            "peaks": "PeakSummaries/FilteredPeaks.json",
            "hwms": "HWMs/FilteredHWMs.json",
            "sites": "Sites/FilteredSites.json",
        }

        try:
            endpoint = endpoint_dict[data_type]
        except KeyError as ke:
            raise InputValueError(data_type, list(endpoint_dict.keys())) from ke

        allowed_query_param_dict = {
            "instruments": cls.instruments_query_params,
            "peaks": cls.peaks_query_params,
            "hwms": cls.hwms_query_params,
            "sites": cls.sites_query_params,
        }

        allowed_query_params = allowed_query_param_dict[data_type]

        if query_params is None:
            query_params = {}

        if not set(query_params.keys()).issubset(allowed_query_params):
            raise InputValueError("query_param", list(allowed_query_params))

        if async_retriever_kwargs is None:
            async_retriever_kwargs = {}
        else:
            async_retriever_kwargs.pop("url", None)
            async_retriever_kwargs.pop("request_kwds", None)

        resp = ar.retrieve_json(
            [f"{cls.service_url}/{endpoint}"],
            request_kwds=[{"params": query_params}],
            **async_retriever_kwargs,
        )
        data = [cls._delist_dict(d) for d in resp[0]]  # pyright: ignore[reportArgumentType]
        if as_list:
            return data

        xy_cols = {
            "instruments": ("longitude", "latitude"),
            "peaks": ("longitude_dd", "latitude_dd"),
            "hwms": ("longitude", "latitude"),
            "sites": ("longitude_dd", "latitude_dd"),
        }
        x_col, y_col = xy_cols[data_type]
        return cls._geopandify(data, x_col, y_col, crs, cls.service_crs)


def stn_flood_event(
    data_type: str, query_params: dict[str, Any] | None = None
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Retrieve data from the STN Flood Event Data API.

    Parameters
    ----------
    data_type : str
        The data source from STN Flood Event Data API.
        It can be ``instruments``, ``peaks``, ``hwms``, or ``sites``.
    query_params : dict, optional
        RESTFUL API query parameters, defaults to ``None`` which returns
        a ``pandas.DataFrame`` of information about the given ``data_type``.
        For accepted values, see the ``STNFloodEventData`` class attributes
        :attr:`~.STNFloodEventData.instruments_query_params`,
        :attr:`~.STNFloodEventData.peaks_query_params`,
        :attr:`~.STNFloodEventData.hwms_query_params`, and
        :attr:`~.STNFloodEventData.sites_query_params` for available values.

        Also, see the API documentation for each data type for more information:

        - `instruments <https://stn.wim.usgs.gov/STNServices/Documentation/Sensor/FilteredSensors>`__
        - `peaks <https://stn.wim.usgs.gov/STNServices/Documentation/PeakSummary/FilteredPeakSummaries>`__
        - `hwms <https://stn.wim.usgs.gov/STNServices/Documentation/HWM/FilteredHWMs>`__
        - `sites <https://stn.wim.usgs.gov/STNServices/Documentation/Site/FilteredSites>`__

    Returns
    -------
    geopandas.GeoDataFrame or pandas.DataFrame
        The retrieved data as a GeoDataFrame or DataFrame
        (if ``query_params`` is not passed).

    Raises
    ------
    InputValueError
        If the input data_type is not one of
        ``instruments``, ``peaks``, ``hwms``, or ``sites``
    InputValueError
        If any of the input query_params are not in accepted
        parameters.

    References
    ----------
    * `USGS Short-Term Network (STN) <https://stn.wim.usgs.gov/STNWeb/#/>`__
    * `Filtered Sensors API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/Sensor/FilteredSensors>`__
    * `Peak Summary API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/PeakSummary/FilteredPeakSummaries>`__
    * `Filtered HWM API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/HWM/FilteredHWMs>`__
    * `Filtered Sites API Documentation <https://stn.wim.usgs.gov/STNServices/Documentation/Site/FilteredSites>`__
    * `USGS Flood Event Viewer: Providing Hurricane and Flood Response Data <https://www.usgs.gov/mission-areas/water-resources/science/usgs-flood-event-viewer-providing-hurricane-and-flood>`__
    * `A USGS guide for finding and interpreting high-water marks <https://www.usgs.gov/media/videos/a-usgs-guide-finding-and-interpreting-high-water-marks>`__
    * `High-Water Marks and Flooding  <https://www.usgs.gov/special-topics/water-science-school/science/high-water-marks-and-flooding>`__
    * `Identifying and preserving high-water mark data <https://doi.org/10.3133/tm3A24>`__

    Notes
    -----
    Notice schema differences between the data dictionaries,
    filtered data queries, and all data queries. This is a known
    issue and is being addressed by USGS.

    Examples
    --------
    >>> query_params = {"States": "SC, CA"}
    >>> data = stn_flood_event("instruments", query_params=query_params)
    >>> data.shape[1]
    34
    >>> data.columns
    Index(['sensorType', 'deploymentType', 'eventName', 'collectionCondition',
        'housingType', 'sensorBrand', 'statusId', 'timeStamp', 'site_no',
        'latitude', 'longitude', 'siteDescription', 'networkNames', 'stateName',
        'countyName', 'siteWaterbody', 'siteHDatum', 'sitePriorityName',
        'siteZone', 'siteHCollectMethod', 'sitePermHousing', 'instrument_id',
        'sensor_type_id', 'deployment_type_id', 'location_description',
        'serial_number', 'housing_serial_number', 'interval', 'site_id',
        'vented', 'instrument_status', 'data_files', 'files', 'geometry'],
        dtype='object')
    """
    if query_params is None:
        return STNFloodEventData.get_all_data(data_type)
    return STNFloodEventData.get_filtered_data(data_type, query_params)
