"""Tests for PyGeoHydro package."""

from __future__ import annotations

import io

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import Polygon

import pygeohydro as gh
import pygeoutils as geoutils
import pynhd as nhd
from pygeohydro import NFHL, NID, NLD, NWIS, WBD, EHydro

DEF_CRS = 4326
ALT_CRS = 3542
SID_NATURAL = "01031500"
SID_URBAN = "11092450"
DATES = ("2005-01-01", "2005-01-31")
DATES_LONG = ("2000-01-01", "2009-12-31")
GEOM = Polygon(
    [
        [-69.77, 45.07],
        [-69.31, 45.07],
        [-69.31, 45.45],
        [-69.77, 45.45],
        [-69.77, 45.07],
    ]
)


def assert_close(a: float, b: float) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-3)


class TestNWIS:
    nwis: NWIS = NWIS()

    def test_qobs_dv(self):
        df = self.nwis.get_streamflow(SID_NATURAL, DATES)
        ds = self.nwis.get_streamflow(SID_NATURAL, DATES, to_xarray=True)
        col = f"USGS-{SID_NATURAL}"
        assert_close(df[col].sum().item(), ds.sel(station_id=col).discharge.sum().item())
        assert df.attrs[col]["huc_cd"] == ds.sel(station_id=col).huc_cd.item()

    def test_qobs_mmd(self):
        df = self.nwis.get_streamflow(SID_NATURAL, DATES, mmd=True)
        assert_close(df[f"USGS-{SID_NATURAL}"].sum().item(), 27.6375)

    def test_cst_tz(self):
        q = self.nwis.get_streamflow(["08075000", "11092450"], DATES)
        assert q.index.tz.tzname(None) == "UTC"

    def test_qobs_iv(self):
        iv = self.nwis.get_streamflow(SID_NATURAL, ("2020-01-01", "2020-01-31"), freq="iv")
        dv = self.nwis.get_streamflow(SID_NATURAL, ("2020-01-01", "2020-01-31"), freq="dv")
        assert_close(abs(iv.mean().item() - dv.mean().item()), 0.0539)

    def test_info(self):
        query = {"sites": ",".join([SID_NATURAL])}
        info = self.nwis.get_info(query, expanded=True, nhd_info=True)
        assert_close(info["nhd_areasqkm"].item(), 773.964)
        assert info.hcdn_2009.item()

    def test_info_box(self):
        query = {"bBox": ",".join(f"{b:.06f}" for b in GEOM.bounds)}
        info_box = self.nwis.get_info(query, nhd_info=True)
        assert info_box.shape[0] == 35
        assert info_box["nhd_areasqkm"].isna().sum() == 31

    def test_param_cd(self):
        codes = self.nwis.get_parameter_codes("%discharge%")
        assert (
            codes.loc[codes.parameter_cd == "00060", "parm_nm"].iloc[0]
            == "Discharge, cubic feet per second"
        )

    def test_fillna(self):
        index = pd.date_range("2000-01-01", "2020-12-31", freq="D")
        q = pd.Series(np.ones(index.size), index=index)
        qf = gh.streamflow_fillna(q)
        assert not qf.name
        q.loc[slice("2000-01-01", "2000-01-05")] = np.nan
        qf = gh.streamflow_fillna(q)
        assert np.all(qf == 1)
        qf = gh.streamflow_fillna(q.to_frame("12345678"))
        assert np.all(qf == 1)
        qf = gh.streamflow_fillna(xr.DataArray(q))
        assert np.all(qf == 1)


class TestETA:
    dates = ("2005-10-01", "2005-10-05")
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
        assert_close(ds.eta.sum().item(), 1.858)
        assert ds.eta.isnull().sum().item() == 5

    def test_geom(self):
        eta_g = gh.ssebopeta_bygeom(GEOM, dates=self.dates)
        assert_close(eta_g.mean().values.item(), 0.6822)

    def test_get_ssebopeta_urls(self):
        _ = gh.helpers.get_ssebopeta_urls(self.years[0])
        urls_dates = gh.helpers.get_ssebopeta_urls(DATES_LONG)
        urls_years = gh.helpers.get_ssebopeta_urls(self.years)
        assert len(urls_dates) == 3653
        assert len(urls_years) == 1095


class TestNLCD:
    years = {"cover": [2016]}
    res = 1000

    @staticmethod
    def assertion(cover, expected):
        st = gh.cover_statistics(cover)
        assert_close(st.categories["Forest"], expected)

    def test_geodf(self):
        geom = gpd.GeoSeries([GEOM, GEOM], crs=DEF_CRS)
        lulc = gh.nlcd_bygeom(geom, years=self.years, resolution=self.res, crs=ALT_CRS, ssl=False)
        self.assertion(lulc[0].cover_2016, 73.1459)
        self.assertion(lulc[1].cover_2016, 73.1459)
        assert lulc[0].cover_2016.rio.nodata == 127

    def test_coords(self):
        coords = list(GEOM.exterior.coords)
        lulc = gh.nlcd_bycoords(coords, ssl=False)
        assert lulc.cover_2021.sum() == 211

    def test_consistency(self):
        coords = [(-87.11890, 34.70421), (-88.83390, 40.17190), (-95.68978, 38.23926)]
        lulc_m = gh.nlcd_bycoords(coords, ssl=False)
        lulc_s = gh.nlcd_bycoords(coords[:1], ssl=False)
        assert lulc_m.iloc[0]["cover_2021"] == lulc_s.iloc[0]["cover_2021"] == 24

    def test_roughness(self):
        geom = gpd.GeoSeries([GEOM], crs=DEF_CRS)
        lulc = gh.nlcd_bygeom(geom, years=self.years, resolution=self.res, crs=ALT_CRS, ssl=False)
        roughness = gh.overland_roughness(lulc[0].cover_2016)
        assert_close(roughness.mean().item(), 0.3256)

    def test_area(self):
        geom = gpd.GeoSeries([GEOM], crs=DEF_CRS)
        area = gh.nlcd_area_percent(geom)
        assert_close(area[["urban", "natural"]].sum(axis=1), 100)
        assert_close(area[["natural", "developed", "impervious"]].sum(axis=1), 100)


class TestNID:
    nid = NID()
    ids = ["KY01232", "GA02400", "NE04081", "IL55070", "TN05345"]

    def test_suggestion(self):
        dams, contexts = self.nid.get_suggestions("houston", "city")
        assert dams.empty
        assert contexts["suggestion"].to_list() == ["Houston", "Houston Lake"]

    def test_filter(self):
        query_list = [
            {"drainageArea": ["[200 500]"]},
            {"nidId": ["CA01222"]},
        ]
        dam_dfs = self.nid.get_byfilter(query_list)
        assert dam_dfs[0].loc[dam_dfs[0].name == "Prairie Portage"].id.item() == "496613"

    def test_id(self):
        dams = self.nid.inventory_byid(self.ids)
        assert_close(dams.damHeight.max(), 39)

    def test_geom(self):
        dams_geo = self.nid.get_bygeom(GEOM, DEF_CRS)
        bbox = geoutils.geometry_reproject(GEOM.bounds, DEF_CRS, ALT_CRS)
        dams_box = self.nid.get_bygeom(bbox, ALT_CRS)
        name = "Pingree Pond"
        assert (dams_geo.name == name).any()
        assert (dams_box.name == "Pingree Pond").any()

    def test_nation(self):
        assert self.nid.df.shape == (92392, 83)
        assert self.nid.gdf.shape == (92245, 97)


class TestWaterQuality:
    wq: gh.WaterQuality = gh.WaterQuality()

    def test_bbox(self):
        stations = self.wq.station_bybbox(
            (-92.8, 44.2, -88.9, 46.0), {"characteristicName": "Caffeine"}
        )
        assert stations.shape[0] == 83

    def test_distance(self):
        stations = self.wq.station_bydistance(-92.8, 44.2, 30, {"characteristicName": "Caffeine"})
        assert stations.shape[0] == 44

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


def test_wbd():
    wbd = WBD("huc4")
    hudson = wbd.byids("huc4", ["0202", "0203"])
    assert len(",".join(hudson.states).split(",")) == 8


def test_states_lookup():
    codes = gh.helpers.states_lookup_table()
    ca = codes["06"].counties
    la_cd = ca[ca.str.contains("Los")].index[0]
    assert la_cd == "037"


@pytest.mark.xfail(reason="Hydroshare is unstable.")
def test_camels():
    attrs, qobs = gh.get_camels()
    assert attrs.shape[0] == qobs.station_id.shape[0] == 671


def test_interactive_map():
    nwis_kwds = {
        "hasDataTypeCd": "dv",
        "outputDataTypeCd": "dv",
        "parameterCd": "00060",
    }
    m = gh.interactive_map((-69.77, 45.07, -69.31, 45.45), nwis_kwds=nwis_kwds)
    assert len(m.to_dict()["children"]) == 4


def test_plot():
    _, _, levels = gh.plot.cover_legends()
    assert levels[-1] == 100


def test_nwis_errors():
    err = gh.helpers.nwis_errors()
    assert err.shape[0] == 7


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (None, 56),
        (["TX", "ca"], 2),
        ("contiguous", 48),
        ("continental", 49),
        ("commonwealths", 4),
        ("territories", 5),
    ],
)
def test_us_states(key, expected):
    states = gh.helpers.get_us_states(key)
    assert states.shape[0] == expected


def test_full_huc():
    hu16 = gh.huc_wb_full(16)
    assert hu16.shape[0] == 7266


def test_irrigation():
    irr = gh.irrigation_withdrawals()
    assert_close(irr.TW.mean(), 419996.4992)


def test_soil():
    soil = gh.soil_properties("por")
    assert soil.sizes["x"] == 266301


@pytest.mark.xfail(reason="NLD is unstable.")
def test_nld():
    nld = NLD("levee_stations")
    levees = nld.bygeom((-105.914551, 37.437388, -105.807434, 37.522392))
    assert levees.shape == (1838, 12)


def test_gnatsgo():
    layers = ["Tk0_100a", "Soc20_50"]
    geometry = (-95.624515, 30.121598, -95.448253, 30.264074)
    soil = gh.soil_gnatsgo(layers, geometry, 4326)
    assert_close(soil.tk0_100a.mean().item(), 89.848)


def test_soilgrid():
    layers = "bdod_5"
    geometry = (-95.624515, 30.121598, -95.448253, 30.264074)
    soil = gh.soil_soilgrids(layers, geometry, 4326)
    assert_close(soil.bdod_0_5cm_mean.mean().item(), 1.4459)


def test_soilpolaris():
    layers = "bd_5"
    geometry = (-95.624515, 30.121598, -95.614515, 30.131598)
    soil = gh.soil_polaris(layers, geometry, 4326)
    assert_close(soil.bd_0_5cm_mean.mean().item(), 1.4620)


# def test_sensorthings():
# sensor = gh.SensorThings()
# cond = " and ".join(
#     ("properties/monitoringLocationType eq 'Stream'", "properties/stateFIPS eq 'US:04'")
# )
# odata = sensor.odata_helper(conditionals=cond)
# df = sensor.query_byodata(odata)
# assert df.shape[0] == 72

# df = sensor.sensor_info("USGS-09380000")
# assert df["description"].iloc[0] == "Stream"

# df = sensor.sensor_property("Datastreams", "USGS-09380000")
# assert df["observationType"].unique()[0] == "Instantaneous"


def test_show_versions():
    f = io.StringIO()
    gh.show_versions(file=f)
    assert "SYS INFO" in f.getvalue()


def test_ehydro():
    bound = (-122.53, 45.57, -122.52, 45.59)
    ehydro = EHydro("bathymetry")
    bathy = ehydro.bygeom(bound)
    assert_close(bathy["depthMean"].mean(), 25.39277)
    assert ehydro.survey_grid.shape[0] == 2672


class TestNFHL:
    """Test the Natinoal Flood Hazard Layer (NFHL) class."""

    @pytest.mark.parametrize(
        ("service", "layer", "expected_url", "expected_layer"),
        [
            (
                "NFHL",
                "cross-sections",
                "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer",
                "Cross-Sections (14)",
            ),
            (
                "Prelim_CSLF",
                "floodway change",
                "https://hazards.fema.gov/arcgis/rest/services/CSLF/Prelim_CSLF/MapServer",
                "Floodway Change (2)",
            ),
            (
                "Draft_CSLF",
                "special flood hazard area change",
                "https://hazards.fema.gov/arcgis/rest/services/CSLF/Draft_CSLF/MapServer",
                "Special Flood Hazard Area Change (3)",
            ),
            (
                "Prelim_NFHL",
                "preliminary water lines",
                "https://hazards.fema.gov/arcgis/rest/services/PrelimPending/Prelim_NFHL/MapServer",
                "Preliminary Water Lines (20)",
            ),
            (
                "Pending_NFHL",
                "pending high water marks",
                "https://hazards.fema.gov/arcgis/rest/services/PrelimPending/Pending_NFHL/MapServer",
                "Pending High Water Marks (12)",
            ),
            (
                "Draft_NFHL",
                "draft transect baselines",
                "https://hazards.fema.gov/arcgis/rest/services/AFHI/Draft_FIRM_DB/MapServer",
                "Draft Transect Baselines (13)",
            ),
        ],
    )
    def test_nfhl(self, service, layer, expected_url, expected_layer):
        """Test the NFHL class."""
        nfhl = NFHL(service, layer)
        assert nfhl.service_info.url == expected_url
        assert nfhl.service_info.layer == expected_layer

    def test_nfhl_fail_layer(self):
        """Test the layer argument failures in NFHL init."""
        with pytest.raises(nhd.exceptions.InputValueError):
            NFHL("NFHL", "cross_sections")

    def test_nfhl_fail_service(self):
        """Test the service argument failures in NFHL init."""
        with pytest.raises(gh.exceptions.InputValueError):
            NFHL("NTHL", "cross-sections")

    @pytest.mark.parametrize(
        ("service", "layer", "geom", "expected_gdf_len", "expected_schema"),
        [
            (
                "NFHL",
                "cross-sections",
                (-73.42, 43.48, -72.5, 43.52),
                44,
                [
                    "geometry",
                    "OBJECTID",
                    "DFIRM_ID",
                    "VERSION_ID",
                    "XS_LN_ID",
                    "WTR_NM",
                    "STREAM_STN",
                    "START_ID",
                    "XS_LTR",
                    "XS_LN_TYP",
                    "WSEL_REG",
                    "STRMBED_EL",
                    "LEN_UNIT",
                    "V_DATUM",
                    "PROFXS_TXT",
                    "MODEL_ID",
                    "SEQ",
                    "SOURCE_CIT",
                    "SHAPE.STLength()",
                    "GFID",
                    "GlobalID",
                ],
            ),
        ],
    )
    def test_nfhl_getgeom(self, service, layer, geom, expected_gdf_len, expected_schema):
        """Test the NFHL bygeom method."""
        nfhl = NFHL(service, layer)
        gdf_xs = nfhl.bygeom(geom, geo_crs=4269)
        assert isinstance(gdf_xs, gpd.GeoDataFrame)
        assert len(gdf_xs) >= expected_gdf_len
        assert set(gdf_xs.columns) == set(expected_schema)
