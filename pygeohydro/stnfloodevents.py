"""
Access USGS Short-Term Network (STN) via Restful API.

References
----------
 .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
"""

import inspect
from io import StringIO
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import requests
from exceptions import InputValueError
from shapely.geometry import Point

timeout = 200


def _check_response(response):
    """
    Check the response from the API and raise an error if the response is not 200.

    Parameters
    ----------
    response : requests.models.Response
        The response from the API.

    Raises
    ------
    requests.exceptions.HTTPError
        If the response status code is not 200.
    """
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


# private decorator that handles requests
def _return_typing(func):
    """
    Handle requests and return data in the specified format.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The decorated function.
    """

    def wrapper(cls, *args, **kwargs):
        response = func(cls, *args, **kwargs)

        # Access the 'returns' argument
        returns = kwargs.get("returns", "GeoDataFrame")

        _check_response(response)

        data = response.json()

        if returns == "List":
            return data

        elif returns == "DataFrame":
            return pd.DataFrame(data)

        elif returns == "GeoDataFrame":
            df = pd.DataFrame(data)

            x_column = "longitude_dd"
            y_column = "latitude_dd"

            if (y_column not in df.columns) | (x_column not in df.columns):
                return df
            else:
                geometry = [Point(xy) for xy in zip(df[x_column], df[y_column])]
                df = df.drop([x_column, y_column], axis=1)

                # needs to use horizontal and/or vertical datums found in the dfs.
                # each point might be different so would need to iterate through each point
                # create 3d points for HWMs

                gdf = gpd.GeoDataFrame(df, crs="EPSG:4329", geometry=geometry)
                return gdf

        else:
            raise InputValueError(returns, ["List", "DataFrame", "GeoDataFrame"])
            # raise ValueError(f"Invalid return type: {returns}")

    return wrapper


def _handle_filter_params(func):
    """
    Handle filter parameters using the inspect module.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The decorated function.
    """

    def wrapper(cls, *args, **kwargs):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        filter_params = {
            k: values[k]
            for k in args
            if (values[k] is not None) and (k != "cls") and (k != "returns")
        }

        returns = kwargs.pop("returns", None)

        return func(cls, returns, **filter_params)

    return wrapper


class STNFloodEventData:
    """
    Python client to retrieve data from the STN Flood Event Data RESTFUL Service API.

    Advantages of using this client are:
        - The user does not need to know the details of the RESTFUL API.
        - Parses the data and returns Python objects (e.g., pandas.DataFrame, geopandas.GeoDataFrame) instead of JSON.
        - Convenience functions are offered for data dictionaries.
        - Geo-references the data where applicable.

    References
    ----------
    .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
    """

    base_url = "https://stn.wim.usgs.gov/STNServices/"

    @classmethod
    def data_dictionaries(
        cls, data_type: str = "instruments", returns: Union[pd.DataFrame, Dict] = "DataFrame"
    ) -> Union[pd.DataFrame, Dict]:
        """
        Retrieve data dictionaries from the STN Flood Event Data API.

        Parameters
        ----------
        data_type : str, optional
            Type of the data to retrieve. It can be 'instruments', 'peaks', 'hwms', or 'sites'. Default is 'instruments'.
        returns : Union[pd.DataFrame, Dict], default = pd.DataFrame

        Returns
        -------
        Union[pd.DataFrame, Dict]
            The retrieved data dictionary as pd.DataFrame or Dict.

        Raises
        ------
        requests.exceptions.HTTPError
            If the response status code is not 200.
        ValueError
            If the return type is not 'DataFrame' or 'Dict'.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
        """
        dtype_dict = {
            "instruments": "Instruments.csv",
            "peaks": "FilteredPeaks.csv",
            "hwms": "FilteredHWMs.csv",
            "sites": "sites.csv",
        }

        data_dictionary_url = "https://stn.wim.usgs.gov/STNWeb/datadictionary"
        response = requests.get(data_dictionary_url + "/" + dtype_dict[data_type], timeout=timeout)

        _check_response(response)

        data = pd.read_csv(StringIO(response.text))

        if "Field" not in data.columns:
            data.iloc[0] = data.columns.tolist()
            data.columns = ["Field", "Definition"]

        data["Definition"] = data["Definition"].apply(lambda x: x.replace("\r\n", "  "))

        # concatenate definitions corresponding to NaN fields until a non-NaN field is encountered
        new_data = {"Field": [], "Definition": []}

        for _, row in data.iterrows():
            if pd.isna(row["Field"]):
                new_data["Definition"][-1] += " " + row["Definition"]
            else:
                new_data["Field"].append(row["Field"])
                new_data["Definition"].append(row["Definition"])

        if returns == "DataFrame":
            return pd.DataFrame(new_data)
        elif returns == "Dict":
            return new_data
        else:
            raise InputValueError(returns, ["DataFrame", "Dict"])
            # raise ValueError(f"Invalid return type: {returns}. Pass 'DataFrame' or 'Dict'.")

    @classmethod
    @_return_typing
    @_handle_filter_params
    def get_sensor_data(
        cls,
        returns: Optional[str] = "GeoDataFrame",
        filter_param1: Optional[str] = None,
        filter_param2: Optional[str] = None,
    ) -> Union[pd.DataFrame, List]:
        """
        Retrieve data from the STN Flood Event Data API.

        Parameters
        ----------
        returns : Optional[str], default = "DataFrame"
            Return object type. Supports "DataFrame" or "List".
        filter_param1 : str, default = None
            RESTFUL API filter parameter 1. Not yet implemented.
        filter_param2 : Optional[str], default = None
            RESTFUL API filter parameter 2. Not yet implemented.

        Raises
        ------
        requests.exceptions.HTTPError
            If the response status code is not 200.

        Returns
        -------
        Union[pd.DataFrame, List]
            The retrieved data as pd.DataFrame or List.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
        """
        return requests.get(
            cls.base_url + "/" + "Instruments.json", timeout=timeout
        )  # , params=params)

    @classmethod
    @_return_typing
    @_handle_filter_params
    def get_hwm_data(
        cls,
        returns: Optional[str] = "GeoDataFrame",
        filter_param1: Optional[str] = None,
        filter_param2: Optional[str] = None,
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame, List]:
        """
        Retrieve data from the STN Flood Event Data API.

        Parameters
        ----------
        returns : str, default = "GeoDataFrame"
            Return object type. Supports "GeoDataFrame", "DataFrame", and "List".
        filter_param1 : Optional[str], default = None
            RESTFUL API filter parameter 1. Not yet implemented.
        filter_param2 : Optional[str], default = None
            RESTFUL API filter parameter 2. Not yet implemented.

        Returns
        -------
        Union[gpd.GeoDataFrame, pd.DataFrame, List]
            The retrieved data as gpd.GeoDataFrame, pd.DataFrame, or List.

        Raises
        ------
        requests.exceptions.HTTPError
            If the response status code is not 200.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
        """
        return requests.get(cls.base_url + "/" + "HWMs.json", timeout=timeout)

    @classmethod
    @_return_typing
    @_handle_filter_params
    def get_peak_data(
        cls,
        returns: Optional[str] = "GeoDataFrame",
        filter_param1: Optional[str] = None,
        filter_param2: Optional[str] = None,
    ) -> Union[pd.DataFrame, List]:
        """
        Retrieve data from the STN Flood Event Data API.

        Parameters
        ----------
        returns : str, default = "DataFrame"
            Return object type. Supports "DataFrame" or "List".
        filter_param1 : Optional[str], default = None
            RESTFUL API filter parameter 1. Not yet implemented.
        filter_param2 : Optional[str], default = None
            RESTFUL API filter parameter 2. Not yet implemented.


        Returns
        -------
        Union[pd.DataFrame, List]
            The retrieved data as pd.DataFrame or List.

        Raises
        ------
        requests.exceptions.HTTPError
            If the response status code is not 200.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
        """
        return requests.get(cls.base_url + "/" + "PeakSummaries.json", timeout=timeout)

    @classmethod
    @_return_typing
    @_handle_filter_params
    def get_site_data(
        cls,
        returns: Optional[str] = "GeoDataFrame",
        filter_param1: Optional[str] = None,
        filter_param2: Optional[str] = None,
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame, List]:
        """
        Retrieve data from the STN Flood Event Data API.

        Parameters
        ----------
        returns : str, default = "GeoDataFrame"
            Return object type. Supports "GeoDataFrame", "DataFrame", and "List".
        filter_param1 : Optional[str], default = None
            RESTFUL API filter parameter 1. Not yet implemented.
        filter_param2 : Optional[str], default = None
            RESTFUL API filter parameter 2. Not yet implemented.

        Returns
        -------
        Union[gpd.GeoDataFrame, pd.DataFrame, List]
            The retrieved data as gpd.GeoDataFrame, pd.DataFrame, or List.

        Raises
        ------
        requests.exceptions.HTTPError
            If the response status code is not 200.

        References
        ----------
        .. [1] [USGS Short-Term Network (STN)](https://stn.wim.usgs.gov/STNWeb/#/)
        """
        return requests.get(cls.base_url + "/" + "Sites.json", timeout=timeout)


if __name__ == "__main__":
    return_type = "Dict"

    instruments_dd, peaks_dd, hwms_dd, sites_dd = (
        STNFloodEventData.data_dictionaries(data_type="instruments", returns=return_type),
        STNFloodEventData.data_dictionaries(data_type="peaks", returns=return_type),
        STNFloodEventData.data_dictionaries(data_type="hwms", returns=return_type),
        STNFloodEventData.data_dictionaries(data_type="sites", returns=return_type),
    )

    try:
        # print(instruments_dd.head(), peaks_dd.head(), hwms_dd.head(), sites_dd.head())
        print(instruments_dd.shape, peaks_dd.shape, hwms_dd.shape, sites_dd.shape)
    except AttributeError:
        print(instruments_dd, peaks_dd, hwms_dd, sites_dd)
        print(len(instruments_dd), len(peaks_dd), len(hwms_dd), len(sites_dd))

    return_type = "GeoDataFrame"
    sensor_data, hwm_data, peak_data, site_data = (
        STNFloodEventData.get_sensor_data(returns=return_type),
        STNFloodEventData.get_hwm_data(returns=return_type),
        STNFloodEventData.get_peak_data(returns=return_type),
        STNFloodEventData.get_site_data(returns=return_type),
    )

    print(type(sensor_data), type(hwm_data), type(peak_data), type(site_data))

    try:
        print(sensor_data.shape, hwm_data.shape, peak_data.shape, site_data.shape)
    except AttributeError:
        print(len(sensor_data), len(hwm_data), len(peak_data), len(site_data))
