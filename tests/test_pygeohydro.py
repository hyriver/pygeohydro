"""Tests for PyGeoHydro package."""
import io

from shapely.geometry import Polygon

import pygeohydro as gh
from pygeohydro import NID, NWIS

SMALL = 1e-3
SID_NATURAL = "01031500"
SID_URBAN = "11092450"
DATES = ("2005-01-01", "2005-01-31")
DATES_LONG = ("2000-01-01", "2009-12-31")
GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)


class TestNWIS:
    "Test NWIS"
    nwis: NWIS = NWIS()

    def test_qobs(self):
        qobs = self.nwis.get_streamflow(SID_NATURAL, DATES, mmd=True)
        assert abs(qobs.sum().item() - 27.630) < SMALL

    def test_info(self):
        query = {"sites": ",".join([SID_NATURAL])}
        info = self.nwis.get_info(query, expanded=True)
        assert info.hcdn_2009.item()

    def test_info_box(self):
        query = {"bBox": ",".join(f"{b:.06f}" for b in GEOM.bounds)}
        info_box = self.nwis.get_info(query)
        assert info_box.shape[0] == 36

    def test_param_cd(self):
        codes = self.nwis.get_parameter_codes("%discharge%")
        assert (
            codes.loc[codes.parameter_cd == "00060", "parm_nm"][0]
            == "Discharge, cubic feet per second"
        )


class TestETA:
    "Test ssebopeta"
    dates = ("2000-01-01", "2000-01-05")
    years = [2010, 2014, 2015]

    def test_coords(self):
        eta_p = gh.ssebopeta_byloc((GEOM.centroid.x, GEOM.centroid.y), dates=self.dates)
        assert abs(eta_p.mean().values[0] - 0.575) < SMALL

    def test_geom(self):
        eta_g = gh.ssebopeta_bygeom(GEOM, dates=self.dates)
        assert abs(eta_g.mean().values.item() - 0.576) < SMALL

    def test_get_ssebopeta_urls(self):
        _ = gh.pygeohydro._get_ssebopeta_urls(self.years[0])
        urls_dates = gh.pygeohydro._get_ssebopeta_urls(DATES_LONG)
        urls_years = gh.pygeohydro._get_ssebopeta_urls(self.years)
        assert len(urls_dates) == 3653 and len(urls_years) == 1095


# def test_nlcd():
#     _ = gh.nlcd(GEOM.bounds, resolution=1e3)
#     years = {"impervious": None, "cover": None, "canopy": 2016, "descriptor": None}
#     lulc = gh.nlcd(GEOM, years=years, resolution=1e3, crs="epsg:3542")
#     st = gh.cover_statistics(lulc.cover)
#     assert abs(st["categories"]["Forest"] - 82.548) < SMALL


class TestNID:
    nid: NID = NID()

    def test_bygeom(self):
        dams = self.nid.bygeom(GEOM, "epsg:4326", sql_clause="MAX_STORAGE > 200")
        assert len(dams) == 5

    def test_byids(self):
        names = ["GUILFORD", "PINGREE POND", "FIRST DAVIS POND"]
        dams = self.nid.byids("DAM_NAME", names)
        assert len(dams) == len(names)

    def test_bysql(self):
        dams = self.nid.bysql("DAM_HEIGHT > 50")
        assert len(dams) == 5331


def test_interactive_map():
    nwis_kwds = {"hasDataTypeCd": "dv", "outputDataTypeCd": "dv", "parameterCd": "00060"}
    m = gh.interactive_map((-69.77, 45.07, -69.31, 45.45), nwis_kwds=nwis_kwds)
    assert len(m.to_dict()["children"]) == 4


def test_plot():
    nwis = NWIS()
    qobs = nwis.get_streamflow([SID_NATURAL, SID_URBAN], DATES_LONG)
    gh.plot.signatures(qobs, precipitation=qobs[f"USGS-{SID_NATURAL}"], output="data/gh.plot.png")
    gh.plot.signatures(qobs[f"USGS-{SID_NATURAL}"], precipitation=qobs[f"USGS-{SID_NATURAL}"])
    _, _, levels = gh.plot.cover_legends()
    assert levels[-1] == 100


def test_helpers():
    err = gh.helpers.nwis_errors()
    assert err.shape[0] == 7


def test_show_versions():
    f = io.StringIO()
    gh.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
