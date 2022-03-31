import geopandas as gpd
import pytest
from shapely.geometry import Polygon

import pygeohydro as gh
from pygeohydro import DataNotAvailable, InvalidInputRange, InvalidInputType, InvalidInputValue

try:
    import typeguard  # noqa: F401
except ImportError:
    has_typeguard = False
else:
    has_typeguard = True

SID_NATURAL = "01031500"
GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)


class TestETAExceptions:
    "Test ssebopeta Exceptions"
    dates = ("2000-01-01", "2000-01-05")

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_coords(self):
        with pytest.raises(InvalidInputType) as ex:
            _ = gh.ssebopeta_byloc(GEOM.centroid.x, dates=self.dates)
        assert "(lon, lat)" in str(ex.value)

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_dates(self):
        with pytest.raises(InvalidInputType) as ex:
            _ = gh.ssebopeta_byloc((GEOM.centroid.x, GEOM.centroid.y), dates="2000-01-01")
        assert "tuple" in str(ex.value)

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_dates_tuple(self):
        with pytest.raises(InvalidInputType) as ex:
            _ = gh.ssebopeta_byloc((GEOM.centroid.x, GEOM.centroid.y), dates=("2000-01-01"))
        assert "(start, end)" in str(ex.value)

    def test_unsupported_dates(self):
        with pytest.raises(InvalidInputRange) as ex:
            _ = gh.ssebopeta_byloc(
                (GEOM.centroid.x, GEOM.centroid.y), dates=("1990-01-01", "1990-01-05")
            )
        assert "2000" in str(ex.value)

    def test_unsupported_years(self):
        with pytest.raises(InvalidInputRange) as ex:
            _ = gh.ssebopeta_byloc((GEOM.centroid.x, GEOM.centroid.y), dates=[2010, 2014, 2021])
        assert "2020" in str(ex.value)


class TestNLCDExceptions:
    "Test NLCD Exceptions"
    years = {"cover": [2016, 2019]}
    res = 1e3
    geom = gpd.GeoSeries([GEOM], crs="EPSG:4326")

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_years_type(self):
        with pytest.raises(InvalidInputType) as ex:
            _ = gh.nlcd_bygeom(self.geom, years=2010, resolution=self.res, ssl=False)
        assert "dict" in str(ex.value)

    def test_invalid_region(self):
        with pytest.raises(InvalidInputValue) as ex:
            _ = gh.nlcd_bygeom(
                self.geom, years=self.years, resolution=self.res, region="us", ssl=False
            )
        assert "L48" in str(ex.value)

    def test_invalid_years(self):
        with pytest.raises(InvalidInputValue) as ex:
            _ = gh.nlcd_bygeom(self.geom, years={"cover": 2020}, resolution=self.res, ssl=False)
        assert "2019" in str(ex.value)

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_invalid_cover_type(self):
        with pytest.raises(InvalidInputType) as ex:
            lulc = gh.nlcd_bygeom(
                self.geom, years={"cover": [2016, 2019]}, resolution=1e3, crs="epsg:3542", ssl=False
            )
            _ = gh.cover_statistics(lulc[0])
        assert "DataArray" in str(ex.value)

    def test_invalid_cover_values(self):
        with pytest.raises(InvalidInputValue) as ex:
            lulc = gh.nlcd_bygeom(
                self.geom, years={"cover": [2016, 2019]}, resolution=1e3, crs="epsg:3542", ssl=False
            )
            _ = gh.cover_statistics(lulc[0].cover_2016 * 2)
        assert "11" in str(ex.value)


class TestNWISExceptions:
    "Test NWIS"
    nwis = gh.NWIS()

    def test_invaild_station(self):
        with pytest.raises(DataNotAvailable) as ex:
            _ = self.nwis.get_streamflow(SID_NATURAL, ("1900-01-01", "1900-01-31"))
        assert "Discharge" in str(ex.value)
