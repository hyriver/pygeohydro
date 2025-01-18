"""Tests for PyGeoHydro package."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
from pyproj.exceptions import CRSError

import pygeohydro as gh


def assert_close(a: float, b: float) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-3)


class TestSTNFloodEventData:
    stn = gh.STNFloodEventData

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
            "last_updated",
            "last_updated_by",
            "other_sid",
            "sensor_not_appropriate",
            "drainage_area_sqmi",
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
        ("data_type", "as_list", "crs", "async_retriever_kwargs", "expected_shape"),
        [
            ("instruments", False, 4329, {"raise_status": False}, (4612, 18)),
            ("peaks", False, None, None, (13159, 22)),
            ("hwms", False, None, {"url": "https://www.google.com"}, (34694, 33)),
            (
                "sites",
                False,
                "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                {},
                (26343, 37),
            ),
            ("sites", False, 4236, {}, (26343, 37)),
            (
                "instruments",
                True,
                4326,
                {"url": "https://www.google.com", "disable": True},
                4612,
            ),
            ("peaks", True, None, {"timeout": 10}, 13159),
            ("hwms", True, 26915, None, 34694),
            ("sites", True, None, None, 26343),
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
        ("data_type", "as_list", "crs", "async_retriever_kwargs", "expected_exception"),
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
        ("data_type", "query_params", "as_list", "crs", "async_retriever_kwargs", "expected_shape"),
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
            ("sites", {"State": "OK, KS, NE, SD, MS, MD, MN, WI"}, False, 3829, {}, (1, 36)),
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
                {"timeout": 10},
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
        (
            "data_type",
            "query_params",
            "as_list",
            "crs",
            "async_retriever_kwargs",
            "expected_exception",
        ),
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
                3829,
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
        ("data_type", "query_params", "expected_shape"),
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
