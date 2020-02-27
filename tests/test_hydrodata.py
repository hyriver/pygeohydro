#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `hydrodata` package."""

import pytest


@pytest.fixture
def get_data():
    """Test all hydrodata functionalities."""
    from hydrodata import Station
    import hydrodata.datasets as hds
    from hydrodata import plot

    lon, lat = -69.32, 45.17
    start, end = "2000-01-01", "2010-01-21"
    wshed = Station(start, end, coords=(lon, lat), data_dir="tests/data")

    dem = hds.dem_bygeom(wshed.geometry)

    clm_loc = hds.deymet_byloc(wshed.lon, wshed.lat, start=wshed.start, end=wshed.end)
    clm_loc["Q (cms)"] = hds.nwis(wshed.station_id, wshed.start, wshed.end)

    variables = ["tmin", "tmax", "prcp"]
    clm_grd = hds.daymet_bygeom(
        wshed.geometry,
        start="2005-01-01",
        end="2005-01-5",
        variables=variables,
        pet=True,
    )
    eta_grd = hds.ssebopeta_bygeom(wshed.geometry, start="2005-01-01", end="2005-01-5")

    stations = wshed.watershed.get_stations()
    stations_upto_150 = wshed.watershed.get_stations(
        navigation="upstreamMain", distance=150
    )

    lulc = hds.NLCD(wshed.geometry)

    p = 0
    plot.signatures(
        {"test": (clm_loc["Q (cms)"], wshed.drainage_area)},
        prcp=clm_loc["prcp (mm/day)"],
        title=wshed.name,
    )
    p = 1
    return (
        dem.isel(x=int(dem.x.shape[0] / 2), y=int(dem.y.shape[0] / 2)).values,
        clm_loc.loc["2008-11-10", "prcp (mm/day)"],
        clm_loc.loc["2008-11-10", "Q (cms)"],
        clm_grd.isel(time=2, x=25, y=20).tmin.values,
        eta_grd.isel(time=2, x=25, y=20).eta.values,
        stations.values[0][3],
        stations_upto_150.values[1][3],
        lulc["cover"]["categories"]["Shrubland"],
        p,
    )


def test_content(get_data):
    """Run the tests"""
    elev, prcp, q, grd, eta, st, st150, cov, p = get_data
    assert (
        abs(elev - 297.0) < 1e-3
        and abs(prcp - 2.0) < 1e-3
        and abs(q - 54.368) < 1e-3
        and abs(grd - (-11.5)) < 1e-3
        and abs(eta - 0.575) < 1e-3
        and st == "USGS-01031300"
        and st150 == "USGS-01031500"
        and abs(cov - 0.142) < 1e-3
        and p == 1
    )
