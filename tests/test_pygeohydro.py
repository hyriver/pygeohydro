"""Tests for PyGeoHydro package."""
import io
import shutil

import geopandas as gpd
import pandas as pd
from pygeoogc import utils as ogc_utils
from shapely.geometry import Polygon

import pygeohydro as gh
from pygeohydro import NID, NWIS

SMALL = 1e-3
DEF_CRS = "epsg:4326"
ALT_CRS = "epsg:3542"
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

    def test_qobs_dv(self):
        df = self.nwis.get_streamflow(SID_NATURAL, DATES, mmd=True)
        ds = self.nwis.get_streamflow(SID_NATURAL, DATES, mmd=True, to_xarray=True)
        col = f"USGS-{SID_NATURAL}"
        assert (
            abs(df[col].sum().item() - ds.sel(station_id=col).discharge.sum().item()) < SMALL
            and df.attrs[col]["huc_cd"] == ds.sel(station_id=col).huc_cd.item()
        )

    def test_qobs_iv(self):
        iv = self.nwis.get_streamflow(SID_NATURAL, ("2020-01-01", "2020-01-31"), freq="iv")
        dv = self.nwis.get_streamflow(SID_NATURAL, ("2020-01-01", "2020-01-31"), freq="dv")
        assert abs(iv.mean().item() - dv.mean().item()) < 0.054

    def test_info(self):
        query = {"sites": ",".join([SID_NATURAL])}
        info = self.nwis.get_info(query, expanded=True)
        assert abs(info.drain_sqkm.item() - 769.048) < SMALL and info.hcdn_2009.item()

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
        coords = pd.DataFrame(
            [
                ["s1", -72.77, 40.07],
                ["s2", -70.31, 46.07],
                ["s3", -69.31, 45.45],
                ["s4", -69.77, 45.45],
            ],
            columns=["id", "x", "y"],
        )
        ds = gh.ssebopeta_bycoords(coords, dates=self.dates)
        assert abs(ds.eta.sum().item() - 8.625) < SMALL and ds.eta.isnull().sum().item() == 5

    def test_geom(self):
        eta_g = gh.ssebopeta_bygeom(GEOM, dates=self.dates)
        assert abs(eta_g.mean().values.item() - 0.576) < SMALL

    def test_get_ssebopeta_urls(self):
        _ = gh.pygeohydro.helpers.get_ssebopeta_urls(self.years[0])
        urls_dates = gh.pygeohydro.helpers.get_ssebopeta_urls(DATES_LONG)
        urls_years = gh.pygeohydro.helpers.get_ssebopeta_urls(self.years)
        assert len(urls_dates) == 3653 and len(urls_years) == 1095

    def test_loc(self):
        eta = gh.ssebopeta_byloc((GEOM.centroid.x, GEOM.centroid.y), dates=self.dates)
        assert abs(eta.mean() - 0.575) < SMALL


class TestNLCD:
    years = {"cover": [2016]}
    res = 1e3

    def test_geodf(self):
        geom = gpd.GeoSeries([GEOM, GEOM], crs=DEF_CRS)
        lulc = gh.nlcd_bygeom(geom, years=self.years, resolution=self.res, crs=ALT_CRS)
        self.assertion(lulc[0].cover_2016, 84.357)
        self.assertion(lulc[1].cover_2016, 84.357)

    def test_coords(self):
        coords = list(GEOM.exterior.coords)
        lulc = gh.nlcd_bycoords(coords)
        assert lulc.cover_2019.sum() == 211

    def test_nlcd_deprecated(self):
        lulc = gh.nlcd(GEOM, years=self.years, resolution=self.res)
        self.assertion(lulc.cover_2016, 83.4)

    @staticmethod
    def assertion(cover, expected):
        st = gh.cover_statistics(cover)
        assert abs(st["categories"]["Forest"] - expected) < SMALL


class TestNID:
    nid = NID()

    def test_suggestion(self):
        dams, contexts = self.nid.get_suggestions("texas", "huc2")
        assert dams.empty and contexts.loc["HUC2", "value"] == "12"

    def test_filter(self):
        query_list = [
            {"huc6": ["160502", "100500"], "drainageArea": ["[200 500]"]},
            {"nidId": ["CA01222"]},
        ]
        dam_dfs = self.nid.get_byfilter(query_list)
        assert dam_dfs[0].name[0] == "Stillwater Point Dam"

    def test_id(self):
        dams = self.nid.inventory_byid(["514871", "459170", "514868", "463501", "463498"])
        assert abs(dams.damHeight.max() - 120) < SMALL

    def test_geom(self):
        dams_geo = self.nid.get_bygeom(GEOM, DEF_CRS)
        bbox = ogc_utils.match_crs(GEOM.bounds, DEF_CRS, ALT_CRS)
        dams_box = self.nid.get_bygeom(bbox, ALT_CRS)
        assert dams_geo.name.iloc[0] == dams_box.name.iloc[0] == "Little Moose"


class TestWaterQuality:
    wq: gh.WaterQuality = gh.WaterQuality()

    def test_bbox(self):
        stations = self.wq.station_bybbox(
            (-92.8, 44.2, -88.9, 46.0), {"characteristicName": "Caffeine"}
        )
        assert stations.shape[0] == 75

    def test_distance(self):
        stations = self.wq.station_bydistance(-92.8, 44.2, 30, {"characteristicName": "Caffeine"})
        assert stations.shape[0] == 38

    def test_data(self):
        stations = [
            "USGS-435221093001901",
            "MN040-443119093050101",
            "MN040-443602092510501",
            "MN040-443656092474901",
            "MN048-442839093085901",
            "MN048-442849093085401",
            "MN048-443122093050101",
            "MN048-443128092593201",
            "MN048-443129092592701",
            "MN048-443140093042801",
            "MN048-443141093042601",
        ]
        caff = self.wq.data_bystation(stations, {"characteristicName": "Caffeine"})
        assert caff.shape[0] == 12


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
    shutil.rmtree("data")

    assert levels[-1] == 100


def test_helpers():
    err = gh.helpers.nwis_errors()
    assert err.shape[0] == 7


def test_show_versions():
    f = io.StringIO()
    gh.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
