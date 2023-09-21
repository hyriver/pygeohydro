"""Tests for PyGeoHydro package."""
import io
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Polygon

import pygeohydro as gh
import pynhd as nhd
from pygeohydro import NFHL, NID, NWIS, WBD, EHydro
from pygeoogc import utils as ogc_utils

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
    assert np.isclose(a, b, rtol=1e-3).all()


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
        assert info_box.shape[0] == 35 and info_box["nhd_areasqkm"].isna().sum() == 29

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
        assert_close(ds.eta.sum().item(), 8.625)
        assert ds.eta.isnull().sum().item() == 5

    def test_geom(self):
        eta_g = gh.ssebopeta_bygeom(GEOM, dates=self.dates)
        assert_close(eta_g.mean().values.item(), 0.577)

    def test_get_ssebopeta_urls(self):
        _ = gh.pygeohydro.helpers.get_ssebopeta_urls(self.years[0])
        urls_dates = gh.pygeohydro.helpers.get_ssebopeta_urls(DATES_LONG)
        urls_years = gh.pygeohydro.helpers.get_ssebopeta_urls(self.years)
        assert len(urls_dates) == 3653 and len(urls_years) == 1095


class TestNLCD:
    years = {"cover": [2016]}
    res = 1e3

    @staticmethod
    def assertion(cover, expected):
        st = gh.cover_statistics(cover)
        assert_close(st.categories["Forest"], expected)

    def test_geodf(self):
        geom = gpd.GeoSeries([GEOM, GEOM], crs=DEF_CRS)
        lulc = gh.nlcd_bygeom(geom, years=self.years, resolution=self.res, crs=ALT_CRS, ssl=False)
        self.assertion(lulc[0].cover_2016, 83.048)
        self.assertion(lulc[1].cover_2016, 83.048)
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
        assert_close(roughness.mean().item(), 0.3197)

    def test_area(self):
        geom = gpd.GeoSeries([GEOM], crs=DEF_CRS)
        area = gh.nlcd_area_percent(geom)
        assert_close(area[["urban", "natural"]].sum(axis=1), 100)
        assert_close(area[["natural", "developed", "impervious"]].sum(axis=1), 100)


class TestNID:
    nid = NID()
    ids = ("KY01232", "GA02400", "NE04081", "IL55070", "TN05345")

    def test_suggestion(self):
        dams, contexts = self.nid.get_suggestions("houston", "city")
        assert dams.empty and contexts["suggestion"].to_list() == ["Houston", "Houston Lake"]

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
        bbox = ogc_utils.match_crs(GEOM.bounds, DEF_CRS, ALT_CRS)
        dams_box = self.nid.get_bygeom(bbox, ALT_CRS)
        name = "Pingree Pond"
        assert (dams_geo.name == name).any() and (dams_box.name == "Pingree Pond").any()

    def test_nation(self):
        assert self.nid.df.shape == (91807, 79)
        assert self.nid.gdf.shape == (91658, 97)


class TestWaterQuality:
    wq: gh.WaterQuality = gh.WaterQuality()

    def test_bbox(self):
        stations = self.wq.station_bybbox(
            (-92.8, 44.2, -88.9, 46.0), {"characteristicName": "Caffeine"}
        )
        assert stations.shape[0] == 82

    def test_distance(self):
        stations = self.wq.station_bydistance(-92.8, 44.2, 30, {"characteristicName": "Caffeine"})
        assert stations.shape[0] == 40

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
    nwis = NWIS()
    qobs = nwis.get_streamflow([SID_NATURAL, SID_URBAN], DATES_LONG)
    gh.plot.signatures(qobs, precipitation=qobs[f"USGS-{SID_NATURAL}"], output="data/gh.plot.png")
    gh.plot.signatures(qobs[f"USGS-{SID_NATURAL}"], precipitation=qobs[f"USGS-{SID_NATURAL}"])
    _, _, levels = gh.plot.cover_legends()
    shutil.rmtree("data")

    assert levels[-1] == 100


def test_nwis_errors():
    err = gh.helpers.nwis_errors()
    assert err.shape[0] == 7


@pytest.mark.parametrize(
    "key,expected",
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
    assert hu16.shape[0] == 7202


def test_irrigation():
    irr = gh.irrigation_withdrawals()
    assert_close(irr.TW.mean(), 419996.4992)


def test_soil():
    soil = gh.soil_properties("por")
    assert soil.dims["x"] == 266301


def test_gnatsgo():
    layers = ["Tk0_100a", "Soc20_50"]
    geometry = (-95.624515, 30.121598, -95.448253, 30.264074)
    soil = gh.soil_gnatsgo(layers, geometry, 4326)
    assert_close(soil.tk0_100a.mean().compute().item(), 89.848)


def test_sensorthings():
    sensor = gh.SensorThings()
    cond = " and ".join(
        ("properties/monitoringLocationType eq 'Stream'", "properties/stateFIPS eq 'US:04'")
    )
    odata = sensor.odata_helper(conditionals=cond)
    df = sensor.query_byodata(odata)
    assert df.shape[0] == 195

    df = sensor.sensor_info("USGS-09380000")
    assert df["description"].iloc[0] == "Stream"

    df = sensor.sensor_property("Datastreams", "USGS-09380000")
    assert df["observationType"].unique()[0] == "Instantaneous"


def test_show_versions():
    f = io.StringIO()
    gh.show_versions(file=f)
    assert "SYS INFO" in f.getvalue()


class TestSTNFloodEventData:
    stn: gh.STNFloodEventData = gh.STNFloodEventData()

    expected_data_dictionary_schema = ["Field", "Definition"]

    expected_all_data_schemas = {
        "instruments": [
            "instrument_id",
            "sensor_type_id",
            "deployment_type_id",
            "location_description",
            "serial_number",
            "interval",
            "site_id",
            "event_id",
            "inst_collection_id",
            "housing_type_id",
            "sensor_brand_id",
            "vented",
            "instrument_status",
            "data_files",
            "files",
            "last_updated",
            "last_updated_by",
            "housing_serial_number",
        ],
        "peaks": [
            "peak_summary_id",
            "member_id",
            "peak_date",
            "is_peak_estimated",
            "is_peak_time_estimated",
            "peak_stage",
            "is_peak_stage_estimated",
            "is_peak_discharge_estimated",
            "vdatum_id",
            "time_zone",
            "calc_notes",
            "data_file",
            "hwms",
            "height_above_gnd",
            "peak_discharge",
            "aep",
            "aep_lowci",
            "aep_upperci",
            "aep_range",
            "is_hag_estimated",
            "last_updated",
            "last_updated_by",
        ],
        "hwms": [
            "hwm_id",
            "waterbody",
            "site_id",
            "event_id",
            "hwm_type_id",
            "hwm_quality_id",
            "hwm_locationdescription",
            "latitude_dd",
            "longitude_dd",
            "survey_date",
            "elev_ft",
            "vdatum_id",
            "vcollect_method_id",
            "bank",
            "approval_id",
            "marker_id",
            "height_above_gnd",
            "hcollect_method_id",
            "hwm_notes",
            "hwm_environment",
            "flag_date",
            "hdatum_id",
            "flag_member_id",
            "survey_member_id",
            "hwm_label",
            "files",
            "stillwater",
            "peak_summary_id",
            "last_updated",
            "last_updated_by",
            "uncertainty",
            "hwm_uncertainty",
            "geometry",
        ],
        "sites": [
            "site_id",
            "site_no",
            "site_name",
            "site_description",
            "state",
            "county",
            "waterbody",
            "latitude_dd",
            "longitude_dd",
            "hdatum_id",
            "hcollect_method_id",
            "member_id",
            "network_name_site",
            "network_type_site",
            "objective_points",
            "instruments",
            "files",
            "site_housing",
            "hwms",
            "site_notes",
            "access_granted",
            "address",
            "city",
            "is_permanent_housing_installed",
            "safety_notes",
            "zip",
            "other_sid",
            "sensor_not_appropriate",
            "drainage_area_sqmi",
            "last_updated",
            "last_updated_by",
            "landownercontact_id",
            "priority_id",
            "zone",
            "usgs_sid",
            "noaa_sid",
            "geometry",
        ],
    }

    expected_filtered_data_schemas = {
        "instruments": [
            "sensorType",
            "deploymentType",
            "eventName",
            "collectionCondition",
            "housingType",
            "sensorBrand",
            "statusId",
            "timeStamp",
            "site_no",
            "latitude",
            "longitude",
            "siteDescription",
            "networkNames",
            "stateName",
            "countyName",
            "siteWaterbody",
            "siteHDatum",
            "sitePriorityName",
            "siteZone",
            "siteHCollectMethod",
            "sitePermHousing",
            "instrument_id",
            "sensor_type_id",
            "deployment_type_id",
            "location_description",
            "serial_number",
            "interval",
            "site_id",
            "vented",
            "instrument_status",
            "data_files",
            "files",
            "housing_serial_number",
            "geometry",
        ],
        "peaks": [
            "vdatum",
            "member_name",
            "site_id",
            "site_no",
            "latitude_dd",
            "longitude_dd",
            "description",
            "networks",
            "state",
            "county",
            "waterbody",
            "horizontal_datum",
            "priority",
            "horizontal_collection_method",
            "perm_housing_installed",
            "peak_summary_id",
            "peak_date",
            "is_peak_estimated",
            "is_peak_time_estimated",
            "peak_stage",
            "is_peak_stage_estimated",
            "is_peak_discharge_estimated",
            "time_zone",
            "calc_notes",
            "data_file",
            "hwms",
            "peak_discharge",
            "zone",
            "height_above_gnd",
            "is_hag_estimated",
            "aep_upperci",
            "geometry",
        ],
        "hwms": [
            "latitude",
            "longitude",
            "eventName",
            "hwmTypeName",
            "hwmQualityName",
            "verticalDatumName",
            "verticalMethodName",
            "approvalMember",
            "markerName",
            "horizontalMethodName",
            "horizontalDatumName",
            "flagMemberName",
            "surveyMemberName",
            "site_no",
            "siteDescription",
            "sitePriorityName",
            "networkNames",
            "stateName",
            "countyName",
            "siteZone",
            "sitePermHousing",
            "site_latitude",
            "site_longitude",
            "hwm_id",
            "waterbody",
            "site_id",
            "event_id",
            "hwm_type_id",
            "hwm_quality_id",
            "hwm_locationdescription",
            "latitude_dd",
            "longitude_dd",
            "survey_date",
            "elev_ft",
            "vdatum_id",
            "vcollect_method_id",
            "bank",
            "approval_id",
            "marker_id",
            "hcollect_method_id",
            "hwm_notes",
            "hwm_environment",
            "flag_date",
            "stillwater",
            "hdatum_id",
            "flag_member_id",
            "survey_member_id",
            "uncertainty",
            "hwm_label",
            "files",
            "height_above_gnd",
            "hwm_uncertainty",
            "peak_summary_id",
            "geometry",
        ],
        "sites": [
            "networkNames",
            "Events",
            "site_id",
            "site_no",
            "site_name",
            "site_description",
            "address",
            "city",
            "state",
            "zip",
            "other_sid",
            "county",
            "waterbody",
            "latitude_dd",
            "longitude_dd",
            "hdatum_id",
            "zone",
            "is_permanent_housing_installed",
            "usgs_sid",
            "noaa_sid",
            "hcollect_method_id",
            "safety_notes",
            "access_granted",
            "network_name_site",
            "network_type_site",
            "objective_points",
            "instruments",
            "files",
            "site_housing",
            "hwms",
            "RecentOP",
            "priority_id",
            "member_id",
            "landownercontact_id",
            "drainage_area_sqmi",
            "geometry",
        ],
    }

    @pytest.mark.parametrize(
        "data_type, as_dict, async_retriever_kwargs, expected_shape",
        [
            ("instruments", False, None, (26, 2)),
            ("peaks", False, {"raise_status": True}, (41, 2)),
            ("hwms", False, None, (51, 2)),
            ("sites", False, {"disable": True, "expire_after": 2e6}, (16, 2)),
            ("instruments", True, {}, 26),
            ("peaks", True, None, 41),
            ("hwms", True, {"url": "https://www.google.com", "max_workers": 9}, 51),
            ("sites", True, None, 16),
        ],
    )
    def test_data_dictionary_success(
        self, data_type, as_dict, async_retriever_kwargs, expected_shape
    ):
        """Test the data_dictionary method of the STNFloodEventData class for success cases."""
        result = self.stn.data_dictionary(
            data_type, as_dict, async_retriever_kwargs=async_retriever_kwargs
        )

        if as_dict:
            assert isinstance(result, dict)
            assert list(result.keys()) == self.expected_data_dictionary_schema

            for field in self.expected_data_dictionary_schema:
                assert len(result[field]) == expected_shape
        else:
            assert isinstance(result, pd.DataFrame)
            assert result.shape == expected_shape
            assert list(result.columns) == self.expected_data_dictionary_schema
            assert result.dtypes.to_list() == [np.dtype("O"), np.dtype("O")]

    @pytest.mark.parametrize(
        "data_type, as_dict, async_retriever_kwargs, expected_exception",
        [
            ("instrimants", False, None, gh.exceptions.InputValueError),
            ("peaks", True, {"dummy": "dummy"}, TypeError),
            ("hwms", False, {"raise_status": True, "anything": 1}, TypeError),
        ],
    )
    def test_data_dictionary_fail(
        self, data_type, as_dict, async_retriever_kwargs, expected_exception
    ):
        """Test the data_dictionary method of the STNFloodEventData class for failure cases."""
        with pytest.raises(expected_exception):
            self.stn.data_dictionary(
                data_type, as_dict, async_retriever_kwargs=async_retriever_kwargs
            )

    @pytest.mark.parametrize(
        "data_type, as_list, crs, async_retriever_kwargs, expected_shape",
        [
            ("instruments", False, 4329, {"raise_status": False}, (4612, 18)),
            ("peaks", False, None, None, (13159, 22)),
            ("hwms", False, None, {"url": "https://www.google.com"}, (34694, 33)),
            (
                "sites",
                False,
                "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                {},
                (23600, 37),
            ),
            ("sites", False, 4236, {}, (23600, 37)),
            (
                "instruments",
                True,
                4326,
                {"url": "https://www.google.com", "disable": True},
                4612,
            ),
            ("peaks", True, None, {"max_workers": 7, "timeout": 10}, 13159),
            ("hwms", True, 26915, None, 34694),
            ("sites", True, None, None, 23600),
        ],
    )
    def test_get_all_data_success(
        self, data_type, as_list, crs, async_retriever_kwargs, expected_shape
    ):
        """Test the get_all_data method of the STNFloodEventData class for success cases."""
        result = self.stn.get_all_data(
            data_type, as_list=as_list, crs=crs, async_retriever_kwargs=async_retriever_kwargs
        )

        if as_list:
            assert isinstance(result, list)
            assert len(result) >= expected_shape
            assert isinstance(result[0], dict)
            assert all(
                rk in self.expected_all_data_schemas[data_type] for rk in list(result[0].keys())
            )
        else:
            assert isinstance(result, (gpd.GeoDataFrame, pd.DataFrame))

            if isinstance(result, gpd.GeoDataFrame):
                if crs is None:
                    crs = self.stn.service_crs
                assert result.crs == CRS(crs)

            assert result.shape[0] >= expected_shape[0]  # minimum number of rows
            assert result.shape[1] == expected_shape[1]  # exact number of columns

            assert list(result.columns) == self.expected_all_data_schemas[data_type]

    @pytest.mark.parametrize(
        "data_type, as_list, crs, async_retriever_kwargs, expected_exception",
        [
            ("instruments", False, 4329, {"raise_status": False, "anything": 1}, TypeError),
            ("peekks", False, None, None, gh.exceptions.InputValueError),
            ("hwms", False, None, {"url": "https://www.google.com", "any": "yes"}, TypeError),
            ("sites", False, "EBSJ:3829", {}, CRSError),
        ],
    )
    def test_get_all_data_fail(
        self, data_type, as_list, crs, async_retriever_kwargs, expected_exception
    ):
        """Test the get_all_data method of the STNFloodEventData class for failure cases."""
        with pytest.raises(expected_exception):
            self.stn.get_all_data(
                data_type, as_list=as_list, crs=crs, async_retriever_kwargs=async_retriever_kwargs
            )

    @pytest.mark.parametrize(
        "data_type, query_params, as_list, crs, async_retriever_kwargs, expected_shape",
        [
            (
                "instruments",
                {"States": "OR,WA,AK,HI"},
                False,
                4329,
                {"raise_status": False},
                (1, 32),
            ),
            ("peaks", {"States": "CA, FL, SC"}, False, None, None, (885, 31)),
            (
                "hwms",
                {"States": "LA"},
                False,
                None,
                {"url": "https://www.google.com", "request_kwds": {"k": "v"}},
                (1208, 54),
            ),
            ("sites", {"State": "OK, KS, NE, SD, MS, MD, MN, WI"}, False, "EPSG:3829", {}, (1, 36)),
            (
                "instruments",
                {"States": "NE,IL,IA,TX"},
                True,
                4326,
                {"url": "https://www.google.com", "disable": True},
                143,
            ),
            (
                "peaks",
                {"States": "NV, AZ, AR, MO, IN"},
                True,
                None,
                {"max_workers": 7, "timeout": 10},
                205,
            ),
            ("hwms", {"States": "KY,WV,NC,GA,TN,PA"}, True, 26915, None, 6220),
            ("sites", {"State": "NY"}, True, None, None, 712),
            ("instruments", None, True, None, None, 4612),
        ],
    )
    def test_get_filtered_data_success(
        self, data_type, query_params, as_list, crs, async_retriever_kwargs, expected_shape
    ):
        """Test the get_filtered_data method of the STNFloodEventData class for success cases."""
        result = self.stn.get_filtered_data(
            data_type,
            query_params,
            as_list=as_list,
            crs=crs,
            async_retriever_kwargs=async_retriever_kwargs,
        )

        if as_list:
            assert isinstance(result, list)
            assert len(result) >= expected_shape
            assert isinstance(result[0], dict)
            assert all(
                rk in self.expected_filtered_data_schemas[data_type]
                for rk in list(result[0].keys())
            )
        else:
            assert isinstance(result, gpd.GeoDataFrame)
            if crs is None:
                crs = self.stn.service_crs
            assert result.crs == CRS(crs)

            assert result.shape[0] >= expected_shape[0]
            assert result.shape[1] == expected_shape[1]

            assert all(
                rc in self.expected_filtered_data_schemas[data_type] for rc in list(result.columns)
            )

    @pytest.mark.parametrize(
        "data_type, query_params, as_list, crs, async_retriever_kwargs, expected_exception",
        [
            (
                "instruments",
                {"States": "OR,WA,AK,HI"},
                False,
                4329,
                {"raise_status": False, "anything": 1},
                TypeError,
            ),
            (
                "peaks",
                {"Storms": "Sandy, Ivan, Harvey"},
                False,
                None,
                None,
                gh.exceptions.InputValueError,
            ),
            (
                "hwms",
                {"States": "LA"},
                False,
                None,
                {"url": "https://www.google.com", "any": "yes"},
                TypeError,
            ),
            (
                "sitessss",
                {"State": "OK, KS, NE, SD, MS, MD, MN, WI"},
                False,
                "EPSG:3829",
                {},
                gh.exceptions.InputValueError,
            ),
            ("instruments", {}, False, "EPSJ:4326", None, CRSError),
        ],
    )
    def test_get_filtered_data_fail(
        self, data_type, query_params, as_list, crs, async_retriever_kwargs, expected_exception
    ):
        """Test the get_filtered_data method of the STNFloodEventData class for failure cases."""
        with pytest.raises(expected_exception):
            self.stn.get_filtered_data(
                data_type,
                query_params,
                as_list=as_list,
                crs=crs,
                async_retriever_kwargs=async_retriever_kwargs,
            )

    @pytest.mark.parametrize(
        "data_type, query_params, expected_shape",
        [
            (
                "instruments",
                {"States": "OR,WA,AK,HI"},
                (1, 32),
            ),
            ("peaks", {"States": "CA, FL, SC"}, (885, 31)),
            (
                "hwms",
                {"States": "LA"},
                (1208, 54),
            ),
            ("sites", {"State": "OK, KS, NE, SD, MS, MD, MN, WI"}, (1, 36)),
            (
                "instruments",
                {"States": "NE,IL,IA,TX"},
                143,
            ),
            (
                "peaks",
                {"States": "NV, AZ, AR, MO, IN"},
                205,
            ),
            ("hwms", {"States": "KY,WV,NC,GA,TN,PA"}, 6220),
            ("sites", {"State": "NY"}, 712),
            ("instruments", None, 4612),
        ],
    )
    def test_stn_func(self, data_type, query_params, expected_shape):
        """Test the function wrapper of the STNFloodEventData class."""
        result = gh.stn_flood_event(data_type, query_params)
        if isinstance(expected_shape, tuple):
            assert result.shape[0] >= expected_shape[0]
            assert result.shape[1] == expected_shape[1]
        else:
            assert len(result) >= expected_shape
        if query_params is None:
            assert all(rc in self.expected_all_data_schemas[data_type] for rc in result)
        else:
            assert all(rc in self.expected_filtered_data_schemas[data_type] for rc in result)


def test_ehydro():
    bound = (-122.53, 45.57, -122.52, 45.59)
    ehydro = EHydro("bathymetry")
    bathy = ehydro.bygeom(bound)
    assert_close(bathy["depthMean"].mean(), 25.5078)
    assert ehydro.survey_grid.shape[0] == 1022


class TestNFHL:
    """Test the Natinoal Flood Hazard Layer (NFHL) class."""

    @pytest.mark.parametrize(
        "service, layer, expected_url, expected_layer",
        [
            (
                "NFHL",
                "cross-sections",
                "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer",
                "Cross-Sections (14)",
            ),
            (
                "Prelim_CSLF",
                "floodway change",
                "https://hazards.fema.gov/gis/nfhl/rest/services/CSLF/Prelim_CSLF/MapServer",
                "Floodway Change (2)",
            ),
            (
                "Draft_CSLF",
                "special flood hazard area change",
                "https://hazards.fema.gov/gis/nfhl/rest/services/CSLF/Draft_CSLF/MapServer",
                "Special Flood Hazard Area Change (3)",
            ),
            (
                "Prelim_NFHL",
                "preliminary water lines",
                "https://hazards.fema.gov/gis/nfhl/rest/services/PrelimPending/Prelim_NFHL/MapServer",
                "Preliminary Water Lines (17)",
            ),
            (
                "Pending_NFHL",
                "pending high water marks",
                "https://hazards.fema.gov/gis/nfhl/rest/services/PrelimPending/Pending_NFHL/MapServer",
                "Pending High Water Marks (12)",
            ),
            (
                "Draft_NFHL",
                "draft transect baselines",
                "https://hazards.fema.gov/gis/nfhl/rest/services/AFHI/Draft_FIRM_DB/MapServer",
                "Draft Transect Baselines (13)",
            ),
        ],
    )
    def test_nfhl(self, service, layer, expected_url, expected_layer):
        """Test the NFHL class."""
        nfhl = NFHL(service, layer)
        assert isinstance(nfhl.service_info, nhd.core.ServiceInfo)
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
        "service, layer, geom, expected_gdf_len, expected_schema",
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
        gdf_xs = nfhl.bygeom(geom, geo_crs="epsg:4269")
        assert isinstance(gdf_xs, gpd.GeoDataFrame)
        assert len(gdf_xs) >= expected_gdf_len
        assert set(gdf_xs.columns) == set(expected_schema)
