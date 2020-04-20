#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

# Package specific imports
import pandas as pd
from shapely.geometry import Polygon

class DataSource:
    """Parent class of data sources."""
    # Attributes every source has. These will be shared across all instances of
    # child classes
    _provider_name = None
    _provider_url = None
    _base_url = None

    def __init__(
        self,
        provider_name,
        provider_url,
        base_url,
        ):
        if not self.__class__._provider_name:
            self.__class__._provider_name = provider_name
            self.__class__._provider_url = provider_url
            self.__class__._base_url = base_url

    def __str__(self):
        """Return provider name and url"""
        return f'{self.__class__._provider_name}\n{self.__class__._provider_url}'

    def _parse_start_end_years_kwargs(
        self,
        **kwargs: Union[str, List[str]],
        ) -> List[pd.core.indexes.datetimes.DatetimeIndex]:
        """Parse passed {start, end, years} to check for their validity.

        Returns
        -------
        List[pandas.core.indexes.datetimes.DatetimeIndex]
        """
        if kwargs.get('years'):
            passed_kwargs = kwargs['years']

            years = passed_kwargs if isinstance(passed_kwargs, list) else [passed_kwargs] 
            
            # This will return a flat list of pandas DatetimeIndicies
            return_dates = []
            for year in years:
                return_dates += pd.date_range(f"{year}0101", f"{year}1231")

            return return_dates
        else:
            try:
                start, end = kwargs['start'], kwargs['end']
                return [pd.date_range(start=start, end=end)]

            except KeyError:
                raise KeyError(f'Please use the key word arguments `start` and `end`, or `years`.')

    @property
    def provider_name(self):
        """Return provider name"""
        return self.__class__._provider_name

    @property
    def provider_url(self):
        """Return url to provider's website"""
        return self.__class__._provider_url

    @classmethod
    def byloc(
        cls,
        lon,
        lat,
        start=None,
        end=None,
        years=None,
        ):
        pass

    @classmethod
    def bygeom(
        cls,
        geometry,
        start=None,
        end=None,
        years=None,
        ):

        if not isinstance(geometry, Polygon):
            raise TypeError("Geometry should be of type Shapely Polygon.")

        return cls._parse_start_end_years_kwargs(start, end, years)

    @classmethod
    def bybox(
        cls,
        feature,
        bbox,
        start=None,
        end=None,
        years=None,
        ):
        pass


class Nwis(DataSource):

    def __init__(self):
        super().__init__(
            "NWIS",
            "https://nwis.waterdata.usgs.gov/nwis",
            "https://waterservices.usgs.gov/nwis/dv",
            )

if __name__ == "__main__":
    nwis = Nwis()

    print(nwis)