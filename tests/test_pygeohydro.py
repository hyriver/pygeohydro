"""Tests for PyGeoHydro package."""
import io

import pytest
from shapely.geometry import Polygon

import pygeohydro as gh
from pygeohydro import NID, NWIS

SID_NATURAL = "01031500"
SID_URBAN = "11092450"
DATES = ("2005-01-01", "2005-01-31")
DATES_LONG = ("2000-01-01", "2009-12-31")
GEOM = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)


def test_nwis():
    nwis = NWIS()
    qobs = nwis.get_streamflow(SID_NATURAL, DATES, mmd=True)
    info = nwis.get_info(nwis.query_byid([SID_NATURAL]), expanded=True)
    info_box = nwis.get_info(nwis.query_bybox(GEOM.bounds))
    assert (
        abs(qobs.sum().item() - 27.630) < 1e-3 and info.hcdn_2009.item() and info_box.shape[0] == 36
    )


def test_ssebopeta():
    dates = ("2000-01-01", "2000-01-05")
    coords = (GEOM.centroid.x, GEOM.centroid.y)
    eta_p = gh.ssebopeta_byloc(coords, dates=dates)
    eta_g = gh.ssebopeta_bygeom(GEOM, dates=dates)
    assert (
        abs(eta_p.mean().values[0] - 0.575) < 1e-3
        and abs(eta_g.mean().values.item() - 0.576) < 1e-3
    )


def test_get_ssebopeta_urls():
    gh.pygeohydro._get_ssebopeta_urls(2010)
    urls_dates = gh.pygeohydro._get_ssebopeta_urls(DATES_LONG)
    urls_years = gh.pygeohydro._get_ssebopeta_urls([2010, 2014, 2015])
    assert len(urls_dates) == 3653 and len(urls_years) == 1095


def test_nlcd():
    gh.nlcd(GEOM.bounds, resolution=1e3)
    years = {"impervious": None, "cover": 2016, "canopy": None}
    lulc = gh.nlcd(GEOM, years=years, resolution=1e3, crs="epsg:3542")
    st = gh.cover_statistics(lulc.cover)
    assert abs(st["categories"]["Forest"] - 82.548) < 1e-3


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


@pytest.mark.parametrize("dv", [True, False])
@pytest.mark.parametrize("iv", [True, False])
def test_interactive_map(dv, iv):
    m = gh.interactive_map(GEOM.bounds, dv=dv, iv=iv)
    if dv and iv:
        assert len(m.to_dict()["children"]) == 11
    elif not dv and not iv:
        assert len(m.to_dict()["children"]) == 37
    else:
        assert len(m.to_dict()["children"]) == 10


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
