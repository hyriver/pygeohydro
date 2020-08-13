"""Tests for Hydrodata package."""
import io
from urllib.error import HTTPError

import pytest
from shapely.geometry import Polygon

import hydrodata as hd
from hydrodata import NWIS

SID_NATURAL = "01031500"
SID_URBAN = "11092450"
DATES = ("2005-01-01", "2005-01-31")
DATES_LONG = ("2000-01-01", "2009-12-31")


@pytest.fixture
def geometry_nat():
    return Polygon(
        [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    )


@pytest.fixture
def geometry_urb():
    return Polygon(
        [
            [-118.72, 34.118],
            [-118.31, 34.118],
            [-118.31, 34.518],
            [-118.72, 34.518],
            [-118.72, 34.118],
        ]
    )


@pytest.mark.flaky(max_runs=3)
def test_nwis(geometry_nat):
    nwis = NWIS()
    qobs = nwis.get_streamflow(SID_NATURAL, DATES, mmd=True)
    info = nwis.get_info(nwis.query_byid(SID_NATURAL), expanded=True)
    info_box = nwis.get_info(nwis.query_bybox(geometry_nat.bounds))
    assert (
        abs(qobs.sum().item() - 27.630) < 1e-3
        and info.hcdn_2009.item()
        and info_box.site_no.tolist() == ["01031300", "01031450", "01031500"]
    )


def test_ssebopeta(geometry_nat):
    dates = ("2000-01-01", "2000-01-05")
    coords = (geometry_nat.centroid.x, geometry_nat.centroid.y)
    eta_p = hd.ssebopeta_byloc(coords, dates=dates)
    eta_g = hd.ssebopeta_bygeom(geometry_nat, dates=dates, fill_holes=True)
    assert (
        abs(eta_p.mean().values[0] - 0.575) < 1e-3
        and abs(eta_g.mean().values.item() - 0.576) < 1e-3
    )


def test_get_ssebopeta_urls():
    hd.hydrodata._get_ssebopeta_urls(2010)
    urls_dates = hd.hydrodata._get_ssebopeta_urls(DATES_LONG)
    urls_years = hd.hydrodata._get_ssebopeta_urls([2010, 2014, 2015])
    assert len(urls_dates) == 3653 and len(urls_years) == 1095


@pytest.mark.flaky(max_runs=3)
def test_nlcd(geometry_nat):
    hd.nlcd(geometry_nat.bounds, resolution=1e3)
    years = {"impervious": None, "cover": 2016, "canopy": None}
    lulc = hd.nlcd(geometry_nat, years=years, resolution=1e3, crs="epsg:3542")
    st = hd.cover_statistics(lulc.cover)
    assert abs(st["categories"]["Forest"] - 82.548) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_plot(geometry_nat, geometry_urb):
    hd.interactive_map(geometry_nat.bounds)
    nwis = NWIS()
    qobs = nwis.get_streamflow([SID_NATURAL, SID_URBAN], DATES_LONG)
    hd.plot.signatures(qobs, precipitation=qobs[f"USGS-{SID_NATURAL}"], output="data/hd.plot.png")
    hd.plot.signatures(qobs[f"USGS-{SID_NATURAL}"], precipitation=qobs[f"USGS-{SID_NATURAL}"])
    _, _, levels = hd.plot.cover_legends()
    assert levels[-1] == 100


@pytest.mark.flaky(max_runs=3)
def test_helpers():
    err = hd.helpers.nwis_errors()
    assert err.shape[0] == 7


def test_show_versions():
    f = io.StringIO()
    hd.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
