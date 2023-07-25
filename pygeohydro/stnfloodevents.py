"""
Access USGS Short-Term Network (STN) via Restful API.

TODO:
    - Add RESTfulURLs to pygeoogc's
    - Documentation
    - Testing

References
----------
 .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
"""

from io import StringIO
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
from exceptions import InputValueError
from numpy import nan
from pyproj import CRS

import async_retriever as ar

CRSTYPE = Union[int, str, CRS]

# Per Athena Clark, Lauren Privette, and Hans Vargas at USGS
# this is the CRS used for visualization on STN front-end.
DEFAULT_CRS = "EPSG:3857"


class STNFloodEventData:
    """
    Python client to retrieve data from the STN Flood Event Data RESTFUL Service API.

    Advantages of using this client are:
        - The user does not need to know the details of RESTFUL in general and of this API specifically.
        - Parses the data and returns Python objects (e.g., pandas.DataFrame, geopandas.GeoDataFrame) instead of JSON.
        - Convenience functions are offered for data dictionaries.
        - Geo-references the data where applicable.

    Attributes
    ----------
    base_url : str
        The base url of the STN Flood Event Data RESTFUL Service API.
    service_url : str
        The service url of the STN Flood Event Data RESTFUL Service API.
    data_dictionary_url : str
        The data dictionary url of the STN Flood Event Data RESTFUL Service API.
    instruments_query_params : Set[str]
        The accepted query parameters for the instruments data type.
    peaks_query_params : Set[str]
        The accepted query parameters for the peaks data type.
    hwms_query_params : Set[str]
        The accepted query parameters for the hwms data type.
    sites_query_params : Set[str]
        The accepted query parameters for the sites data type.

    Methods
    -------
    get_data_dictionary
        Retrieves the data dictionary for a given data type.
    get_all_data
        Retrieves all data for a given data type.
    get_filtered_data
        Retrieves filtered data for a given data type.

    References
    ----------
    .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
    """

    # TODO: Add to pygeoogc's RESTfulURLs
    base_url = "https://stn.wim.usgs.gov"
    service_url = urljoin(base_url, "STNServices/")
    data_dictionary_url = urljoin(base_url, "STNWeb/datadictionary/")

    # accepted query parameters for instruments data type
    instruments_query_params = {
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

    # accepted query parameters for peaks data type
    peaks_query_params = {
        "Event",
        "EventType",
        "EventStatus",
        "States",
        "County",
        "StartDate",
        "EndDate",
    }

    # accepted query parameters for hwms data type
    hwms_query_params = {
        "Event",
        "EventType",
        "EventStatus",
        "States",
        "County",
        "StartDate",
        "EndDate",
    }

    # accepted query parameters for sites data type
    sites_query_params = {
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

    '''
    @classmethod
    def _geopandify_various_crs(
        cls,
        input_list: List[Dict],
        crs: Optional[CRSTYPE] = DEFAULT_CRS,
        x_column: str = "longitude_dd",
        y_column: str = "latitude_dd",
        crs_col: str = "hdatum_id",
    ) -> gpd.GeoDataFrame:
        """
        Georeferences a list of dictionaries.

        NOT IMPLEMENTED:
        This function is not implemented yet as STN USGS reps Athena Clark, Lauren Privette, and Hans Vargas informed us that 'EPSG:3857' web mercator is assumed for visualization purposes on STN front-end. We chose to to use the same assumption as USGS and avoid assigning independent CRSs to each data point. We will revisit this in the future if needed.

        The rest of the functionality uses the alternate function _geopandify().

        Parameters
        ----------
        input_list : List[Dict]
            The list of dictionaries to be converted to a geodataframe.
        crs : Optional[CRSTYPE], default = DEFAULT_CRS
            Desired the coordinate reference system.
        x_column : str, default = 'longitude'
            The column name of the x-coordinate.
        y_column : str, default = 'latitude'
            The column name of the y-coordinate.
        crs_col : str, default = 'hdatum_id'
            The column name of the horizontal datum.

        Returns
        -------
        gpd.GeoDataFrame
            The geo-referenced dataframe.
        """
        """
        NOTE: Horizontal Datum Codes
        ----------------------------
        From HWM data dictionary:
            1 local control point  2 NAD83  3 NAD27  4 WGS84 (from Digital Map)  5 NAD 83( CORS96) epoch 2002  6 NAD 83 (NSRS2007) epoch 2007  7 NAD 83 (2011) epoch 2010'

        From Site data dictionary:
            1 local control point 2 NAD83 3 NAD27 4 WGS84 (from Digital Map) 5 NAD 83( CORS96) epoch 2002 6 NAD 83 (NSRS2007) epoch 2007 7 NAD 83 (2011) epoch 2010'

        NOTE: Vertical Datum Codes
        --------------------------
        From HWM data dictionary:
            1 local control point  2 NAVD88  3 Above Ground Level  4 NGVD29  6 PRVD02  7 VIVD09  8 International Great Lakes Datum of 1985

        From peaks data dictionary:
            1 local control point  2 NAVD88  3 Above Ground Level  4 NGVD29  6 PRVD02  7 VIVD09  8 International Great Lakes Datum of 1985'
        """

        # TODO: These are not correct. Need to find the correct EPSG codes.
        hdatum_dict = {
            1: "EPSG:4329",  # local control point
            2: "EPSG:4269",  # NAD83
            3: "EPSG:4267",  # NAD27
            4: "EPSG:4326",  # WGS84 (from Digital Map)
            5: "EPSG:4269",  # NAD 83( CORS96) epoch 2002
            6: "EPSG:4269",  # NAD 83 (NSRS2007) epoch 2007
            7: "EPSG:4249",  # NAD 83 (2011) epoch 2010
            "local control point": "EPSG:4329",
            "NAD83": "EPSG:4269",
            "NAD27": "EPSG:4267",
            "WGS84 (from Digital Map)": "EPSG:4326",
            "NAD 83 (2011) epoch 2010": "EPSG:4269",
            9: "EPSG:4329",  # this is not defined in data dictionaries but does appear in data.
        }

        def apply_hdatum(row, crs=crs):
            """Compute geometry for each row."""
            try:
                hdatum_value = row[crs_col]
            except KeyError:
                # TODO: Handle this better
                hdatum_value = row["siteHDatum"]
                warnings.warn(
                    f"Key {crs_col} not found in data dictionary. Using 'siteHDatum' instead."
                )

            try:
                from_crs = hdatum_dict[hdatum_value]
            except KeyError:
                from_crs = DEFAULT_CRS  # default CRS is
                warnings.warn(
                    f"Key {row[crs_col]} not found in data dictionary. Using default CRS, {DEFAULT_CRS}."
                )

            geom = gpd.points_from_xy([row[x_column]], [row[y_column]], crs=from_crs).to_crs(crs)

            return geom[0]

        df = pd.DataFrame(input_list)
        df["geometry"] = df.apply(apply_hdatum, axis=1, crs=crs)

        return gpd.GeoDataFrame(df, crs=crs)
    '''

    @classmethod
    def _geopandify(
        cls,
        input_list: List[Dict],
        crs: Optional[CRSTYPE] = DEFAULT_CRS,
        x_column: str = "longitude_dd",
        y_column: str = "latitude_dd",
    ) -> gpd.GeoDataFrame:
        """
        Georeference a list of dictionaries.

        Parameters
        ----------
        input_list : List[Dict]
            The list of dictionaries to be converted to a geodataframe.
        crs : Optional[CRSTYPE], default = DEFAULT_CRS
            Desired the coordinate reference system.
        x_column : str, default = 'longitude'
            The column name of the x-coordinate.
        y_column : str, default = 'latitude'
            The column name of the y-coordinate.

        Returns
        -------
        gpd.GeoDataFrame
            The geo-referenced dataframe.
        """
        df = pd.DataFrame(input_list)

        df["geometry"] = gpd.points_from_xy(df[x_column], df[y_column], crs=DEFAULT_CRS).to_crs(crs)

        return gpd.GeoDataFrame(df, crs=crs)

    @classmethod
    def _delist_dict(cls, d):
        """De-lists all unit length lists in a dictionary."""

        def delist(x):
            if isinstance(x, list) and len(x) == 1:
                return x[0]
            elif isinstance(x, list) and len(x) == 0:
                return nan
            return x

        return {k: delist(v) for k, v in d.items()}

    @classmethod
    def data_dictionaries(
        cls, data_type: str, as_dict: bool = False, async_retriever_kwargs: Optional[Dict] = None
    ) -> Union[pd.DataFrame, Dict]:
        """
        Retrieve data dictionaries from the STN Flood Event Data API.

        Parameters
        ----------
        data_type : str
            Type of the data to retrieve. It can be 'instruments', 'peaks', 'hwms', or 'sites'.
        as_dict : bool, default = False
            If True, return the data dictionary as a dictionary. Otherwise, it returns as pd.DataFrame.
        async_retriever_kwargs : Optional[Dict], default = None
            Additional keyword arguments to pass to `async_retriever.retrieve_text()`. URL is already set.

        Returns
        -------
        Union[pd.DataFrame, Dict]
            The retrieved data dictionary as pd.DataFrame or Dict.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)

        See Also
        --------
        get_all_data : Retrieves all data for a given data type.
        get_filtered_data : Retrieves filtered data for a given data type.

        Examples
        --------
        >>> from stnfloodevents import STNFloodEventData
        >>> data = STNFloodEventData.data_dictionaries(data_type="instruments", as_dict=False)
        >>> data.shape
        (26, 2)
        >>> data.columns
        Index(['Field', 'Definition'], dtype='object')
        """
        dtype_dict = {
            "instruments": "Instruments.csv",
            "peaks": "FilteredPeaks.csv",
            "hwms": "FilteredHWMs.csv",
            "sites": "sites.csv",
        }

        try:
            file_name = dtype_dict[data_type]
        except KeyError as ke:
            raise InputValueError(data_type, list(dtype_dict.keys())) from ke

        if async_retriever_kwargs is None:
            async_retriever_kwargs = {}
        else:
            async_retriever_kwargs.pop("url", None)

        # retrieve
        response = ar.retrieve_text(
            [urljoin(cls.data_dictionary_url, file_name)], **async_retriever_kwargs
        )[0]

        # convert to DataFrame
        data = pd.read_csv(StringIO(response))

        if "Field" not in data.columns:
            data.iloc[0] = data.columns.tolist()
            data.columns = ["Field", "Definition"]

        data["Definition"] = data["Definition"].apply(lambda x: x.replace("\r\n", "  "))

        # concatenate definitions corresponding to NaN fields until a non-NaN field is encountered
        data_dict = {"Field": [], "Definition": []}

        for _, row in data.iterrows():
            if pd.isna(row["Field"]):
                data_dict["Definition"][-1] += " " + row["Definition"]
            else:
                data_dict["Field"].append(row["Field"])
                data_dict["Definition"].append(row["Definition"])

        if as_dict:
            return data_dict
        return pd.DataFrame(data_dict)

    @classmethod
    def get_all_data(
        cls,
        data_type: str,
        as_list: Optional[bool] = False,
        crs: Optional[str] = DEFAULT_CRS,
        async_retriever_kwargs: Optional[Dict] = None,
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame, List[Dict]]:
        """
        Retrieve all data from the STN Flood Event Data API for instruments, peaks, hwms, and sites.

        Parameters
        ----------
        data_type : str
            The data source from STN Flood Event Data API. It can be 'instruments', 'peaks', 'hwms', or 'sites'.
        as_list : Optional[bool], default = False
            If True, return the data as a list.
        crs : Optional[str], default = DEFAULT_CRS
            Desired Coordinate reference system (CRS) of output.
        async_retriever_kwargs : Optional[Dict], default = None
            Additional keyword arguments to pass to `async_retriever.retrieve_json()`. URL is already set.

        Returns
        -------
        Union[gpd.GeoDataFrame, pd.DataFrame, List[Dict]]
            The retrieved data as a GeoDataFrame, DataFrame, or a list of dictionaries.

        Raises
        ------
        InputValueError
            If the input data_type is not one of 'instruments', 'peaks', 'hwms', or 'sites'.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
        .. [2] [All Sensors API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/Sensor/AllSensors)
        .. [3] [All Peak Summary API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/PeakSummary/AllPeakSummaries)
        .. [4] [All HWM API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/HWM/AllHWMs)
        .. [5] [All Sites API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/Site/AllSites)

        See Also
        --------
        get_filtered_data : Retrieves filtered data for a given data type.
        get_data_dictionary : Retrieves the data dictionary for a given data type.

        Examples
        --------
        >>> from stnfloodevents import STNFloodEventData
        >>> data = STNFloodEventData.get_all_data(data_type="instruments")
        >>> data.shape
        (4624, 18)
        >>> data.columns
        Index(['instrument_id', 'sensor_type_id', 'deployment_type_id',
               'location_description', 'serial_number', 'interval', 'site_id',
               'event_id', 'inst_collection_id', 'housing_type_id', 'sensor_brand_id',
               'vented', 'instrument_status', 'data_files', 'files', 'last_updated',
               'last_updated_by', 'housing_serial_number'],
               dtype='object')
        """
        # non-filtered endpoints
        endpoint_dict = {
            "instruments": "Instruments.json",
            "peaks": "PeakSummaries.json",
            "hwms": "HWMs.json",
            "sites": "Sites.json",
        }

        try:
            endpoint = endpoint_dict[data_type]
        except KeyError as ke:
            raise InputValueError(data_type, list(endpoint_dict.keys())) from ke

        if async_retriever_kwargs is None:
            async_retriever_kwargs = {}
        else:
            async_retriever_kwargs.pop("url", None)

        # retrieve data
        data = ar.retrieve_json([urljoin(cls.service_url, endpoint)], **async_retriever_kwargs)[0]

        # delists all unit length lists in a dictionary
        data = [cls._delist_dict(d) for d in data]

        # denotes the fields that are considered as x and y coordinates by data type
        x_and_y_columns = {
            "instruments": None,
            "peaks": None,
            "hwms": ("longitude_dd", "latitude_dd"),
            "sites": ("longitude_dd", "latitude_dd"),
        }

        if as_list:
            return data
        elif x_and_y_columns[data_type] is None:
            return pd.DataFrame(data)
        else:
            x_column, y_column = x_and_y_columns[data_type]

            return cls._geopandify(data, crs=crs, x_column=x_column, y_column=y_column)

    @classmethod
    def get_filtered_data(
        cls,
        data_type: str,
        query_params: Optional[Dict] = None,
        as_list: Optional[bool] = False,
        crs: Optional[str] = DEFAULT_CRS,
        async_retriever_kwargs: Optional[Dict] = None,
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame, List[Dict]]:
        """
        Retrieve filtered data from the STN Flood Event Data API for instruments, peaks, hwms, and sites.

        Parameters
        ----------
        data_type : str
            The data source from STN Flood Event Data API. It can be 'instruments', 'peaks', 'hwms', or 'sites'.
        query_params : Optional[Dict], default = None
            RESTFUL API query parameters. For accepted values, see the STNFloodEventData class attributes instruments_accepted_params, peaks_accepted_params, hwms_accepted_params, and sites_accepted_params for available values.

            Also, see the API documentation for each data type for more information:
                - [instruments](https://stn.wim.usgs.gov/STNServices/Documentation/Sensor/FilteredSensors)
                - [peaks](https://stn.wim.usgs.gov/STNServices/Documentation/PeakSummary/FilteredPeakSummaries)
                - [hwms](https://stn.wim.usgs.gov/STNServices/Documentation/HWM/FilteredHWMs)
                - [sites](https://stn.wim.usgs.gov/STNServices/Documentation/Site/FilteredSites)
        as_list : Optional[bool], default = False
            If True, return the data as a list.
        crs : Optional[str], default = DEFAULT_CRS
            Desired Coordinate reference system (CRS) of output.
        async_retriever_kwargs : Optional[Dict], default = None
            Additional keyword arguments to pass to `async_retriever.retrieve_json()`. URL and request_kwds are already set.

        Returns
        -------
        Union[gpd.GeoDataFrame, pd.DataFrame, List[Dict]]
            The retrieved data as a GeoDataFrame, DataFrame, or a list of dictionaries.

        Raises
        ------
        InputValueError
            If the input data_type is not one of 'instruments', 'peaks', 'hwms', or 'sites'.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
        .. [2] [Filtered Sensors API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/Sensor/FilteredSensors)
        .. [3] [Peak Summary API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/PeakSummary/FilteredPeakSummaries)
        .. [4] [Filtered HWM API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/HWM/FilteredHWMs)
        .. [5] [Filtered Sites API Documentation](https://stn.wim.usgs.gov/STNServices/Documentation/Site/FilteredSites)

        See Also
        --------
        get_all_data : Retrieves all data for a given data type.
        get_data_dictionary : Retrieves the data dictionary for a given data type.

        Examples
        --------
        >>> from stnfloodevents import STNFloodEventData
        >>> query_params = {"States": "SC, CA"}
        >>> data = STNFloodEventData.get_filtered_data(data_type="instruments", query_params=query_params)
        >>> data.shape
        (473, 34)
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
        # filtered endpoints
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

        # check if query_params are valid
        if not set(query_params.keys()).issubset(allowed_query_params):
            raise InputValueError("query_params", allowed_query_params)

        if async_retriever_kwargs is None:
            async_retriever_kwargs = {}
        else:
            async_retriever_kwargs.pop("url", None)
            async_retriever_kwargs.pop("request_kwds", None)

        # retrieve data
        data = ar.retrieve_json(
            [urljoin(cls.service_url, endpoint)],
            request_kwds=[{"params": query_params}],
            **async_retriever_kwargs,
        )[0]

        # delists all unit length lists in a dictionary
        data = [cls._delist_dict(d) for d in data]

        # denotes the fields that are considered as x and y coordinates by data type
        x_and_y_columns = {
            "instruments": ("longitude", "latitude"),
            "peaks": ("longitude_dd", "latitude_dd"),
            "hwms": ("longitude", "latitude"),
            "sites": ("longitude_dd", "latitude_dd"),
        }

        if as_list:
            return data
        elif x_and_y_columns[data_type] is None:
            return pd.DataFrame(data)
        else:
            x_column, y_column = x_and_y_columns[data_type]

            return cls._geopandify(data, crs=crs, x_column=x_column, y_column=y_column)


if __name__ == "__main__":
    data_types = ["instruments", "peaks", "hwms", "sites"]

    query_params = [
        {},
        {"States": "SC, CA"},
        {"States": "SC, CA"},
        {"State": "SC,CA"},
    ]

    for data_type, query_param in zip(data_types, query_params):
        try:
            print(f"Getting filtered {data_type} data ...")
            data = STNFloodEventData.data_dictionaries(data_type=data_type, as_dict=False)
        except ar.exceptions.ServiceError:
            print(f"{data_type} data dictionary not available.")
        else:
            print(data.columns, type(data))

        try:
            print(f"Getting filtered {data_type} data ...")
            data = STNFloodEventData.get_filtered_data(
                data_type=data_type, query_params=query_param, as_list=False
            )
        except ar.exceptions.ServiceError:
            print(f"{data_type} filtered data not available.")
        else:
            print(data.columns, type(data))

        try:
            print(f"Getting all {data_type} data ...")
            data = STNFloodEventData.get_all_data(data_type=data_type, as_list=False)
        except ar.exceptions.ServiceError:
            print(f"{data_type} all data not available.")
        else:
            print(data.columns, type(data))
