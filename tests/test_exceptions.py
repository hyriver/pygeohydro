import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

import pygeohydro as gh
from pygeohydro import (
    DataNotAvailableError,
    InputRangeError,
    InputTypeError,
    InputValueError,
)

has_typeguard = bool(sys.modules.get("typeguard"))

SID_NATURAL = "01031500"
GEOM = Polygon(
    [
        [-69.77, 45.07],
        [-69.31, 45.07],
        [-69.31, 45.45],
        [-69.77, 45.45],
        [-69.77, 45.07],
    ]
)


class TestETAExceptions:
    dates = ("2000-01-01", "2000-01-05")

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_dates(self):
        with pytest.raises(InputTypeError) as ex:
            _ = gh.ssebopeta_bycoords((GEOM.centroid.x, GEOM.centroid.y), dates="2000-01-01")
        assert "pandas.DataFrame" in str(ex.value)

    def test_unsupported_years(self):
        coords = pd.DataFrame(
            [
                ["s1", -72.77, 40.07],
                ["s2", -70.31, 46.07],
                ["s3", -69.31, 45.45],
                ["s4", -69.77, 45.45],
            ],
            columns=["id", "x", "y"],
        )
        with pytest.raises(InputRangeError) as ex:
            _ = gh.ssebopeta_bycoords(coords, dates=[2010, 2014, 2030])
        assert "2022" in str(ex.value)


class TestNLCDExceptions:
    """Test NLCD Exceptions."""

    years = {"cover": [2016, 2019]}
    res = 1e3
    geom = gpd.GeoSeries([GEOM], crs=4326)

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_years_type(self):
        with pytest.raises(InputTypeError) as ex:
            _ = gh.nlcd_bygeom(self.geom, years=2010, resolution=self.res, ssl=False)
        assert "dict" in str(ex.value)

    def test_invalid_region(self):
        with pytest.raises(InputValueError) as ex:
            _ = gh.nlcd_bygeom(
                self.geom, years=self.years, resolution=self.res, region="us", ssl=False
            )
        assert "L48" in str(ex.value)

    def test_invalid_years(self):
        with pytest.raises(InputValueError) as ex:
            _ = gh.nlcd_bygeom(self.geom, years={"cover": 2030}, resolution=self.res, ssl=False)
        assert "2019" in str(ex.value)

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_cover_type(self):
        with pytest.raises(InputTypeError) as ex:
            lulc = gh.nlcd_bygeom(
                self.geom,
                years={"cover": [2016, 2019]},
                resolution=1e3,
                crs="epsg:3542",
                ssl=False,
            )
            _ = gh.cover_statistics(lulc[0])
        assert "DataArray" in str(ex.value)

    def test_invalid_cover_values(self):
        with pytest.raises(InputValueError) as ex:
            lulc = gh.nlcd_bygeom(
                self.geom,
                years={"cover": [2016, 2019]},
                resolution=1e3,
                crs="epsg:3542",
                ssl=False,
            )
            _ = gh.cover_statistics(lulc[0].cover_2016 * 2)
        assert "11" in str(ex.value)


class TestNWISExceptions:
    nwis = gh.NWIS()

    def test_invaild_station(self):
        with pytest.raises(DataNotAvailableError) as ex:
            _ = self.nwis.get_streamflow(SID_NATURAL, ("1900-01-01", "1900-01-31"))
        assert "Discharge" in str(ex.value)
