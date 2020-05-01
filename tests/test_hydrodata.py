#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `hydrodata` package."""

import hydrodata.datasets as hds
from hydrodata import Station, helpers, plot, services, utils


def test_station():
    natural = Station("2000-01-01", "2010-01-21", station_id="01031500", verbose=True)
    natural = Station("2000-01-01", "2010-01-21", station_id="01031500", verbose=True)
    urban = Station(start="2000-01-01", end="2010-01-21", coords=(-118.47, 34.16))
    assert natural.hcdn and not urban.hcdn


def test_nwis():
    df = hds.nwis_streamflow("01031500", "2000-01-01", "2000-01-31")
    assert abs(df.sum().values[0] - 139.8569) < 1e-4


def test_daymet():
    wshed = Station("2000-01-01", "2000-01-12", station_id="01031500")
    variables = ["tmin"]
    st_p = hds.daymet_byloc(
        -118.47, 34.16, start=wshed.start, end=wshed.end, variables=variables, pet=True
    )
    yr_p = hds.daymet_byloc(-118.47, 34.16, years=2010, variables=variables)

    st_g = hds.daymet_bygeom(
        wshed.geometry, start=wshed.start, end=wshed.end, variables=variables, pet=True
    )
    yr_g = hds.daymet_bygeom(wshed.geometry, years=2010, variables=variables)
    assert (
        abs(st_g.isel(time=10, x=5, y=10).pet.values.item() - 0.6823) < 1e-4
        and abs(yr_g.isel(time=10, x=5, y=10).tmin.values.item() - (-18.0)) < 1e-1
        and abs(st_p.iloc[10]["pet (mm/day)"] - 2.3928) < 1e-4
        and abs(yr_p.iloc[10]["tmin (deg c)"] - 11.5) < 1e-1
    )


def test_nldi():
    station_id = "01031500"
    nldi = hds.NLDI

    sid = nldi.starting_comid(station_id)
    ids = nldi.comids(station_id)
    trib = nldi.tributaries(station_id)
    main = nldi.main(station_id)
    st100 = nldi.stations(station_id, navigation="upstreamTributaries", distance=100)
    stm = nldi.stations(station_id, navigation="upstreamMain")
    pp = nldi.pour_points(station_id)
    fl = utils.prepare_nhdplus(
        nldi.flowlines(station_id), 0, 0, purge_non_dendritic=False
    )
    ct = nldi.catchments(station_id)

    assert (
        sid == 1722317
        and ids[-1] == 1723451
        and trib.shape[0] == 432
        and main.shape[0] == 52
        and st100.shape[0] == 3
        and stm.shape[0] == 2
        and pp.shape[0] == 12
        and abs(fl.lengthkm.sum() - 565.755) < 1e-3
        and abs(ct.areasqkm.sum() - 773.9541) < 1e-4
    )


def test_nhdplus_bybox():
    wb = hds.nhdplus_bybox(
        "nhdwaterbody",
        (-69.7718294059999, 45.074243489, -69.314140401, 45.4533586220001),
    )
    assert abs(wb.areasqkm.sum() - 87.084) < 1e-3


def test_ssebopeta():
    wshed = Station("2000-01-01", "2000-01-05", station_id="01031500")
    eta_p = hds.ssebopeta_byloc(*wshed.coords, start=wshed.start, end=wshed.end)
    eta_g = hds.ssebopeta_bygeom(wshed.geometry, start=wshed.start, end=wshed.end)
    assert (
        abs(eta_p.mean().values[0] == 0.5750) < 1e-4
        and abs(eta_g.mean().values.item() - 0.5766) < 1e-4
    )


def test_nlcd():
    wshed = Station("2000-01-01", "2000-01-05", station_id="01031500")
    lulc = hds.nlcd(wshed.geometry, resolution=1)
    st = utils.cover_statistics(lulc.cover)
    assert abs(st["categories"]["Forest"] - 43.0943) < 1e-4


def test_dem():
    wshed = Station("2000-01-01", "2000-01-05", station_id="01031500")
    dem = hds.nationalmap_dem(wshed.geometry, resolution=1)
    assert abs(dem.mean().values.item() - 302.2381) < 1e-4


def test_newdb():
    wshed = Station("2005-01-01", "2005-01-31", "11092450")

    s = services.ArcGISREST(host="maps.lacity.org", site="lahub", verbose=True)
    s.spatialRel = "esriSpatialRelIntersects"
    s.folder = "Utilities"
    s.folder = None
    s.get_fs()
    s.folder
    s.serviceName = "Stormwater_Information"
    s.get_layers()
    s.layer = 10
    s.generate_url()
    url_rest = "https://maps.lacity.org/lahub/rest/services/Stormwater_Information/MapServer/10"
    s = services.ArcGISREST(url_rest, verbose=True)
    s.n_threads = 4
    s.get_featureids(wshed.geometry.bounds)
    s.get_featureids(wshed.geometry)
    s.outFormat = "geojson"
    storm_pipes = s.get_features()
    s.outFormat = "json"
    storm_pipes = s.get_features()

    url_wms = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"
    slope = services.wms_bygeom(
        url_wms,
        "3DEP",
        geometry=wshed.geometry,
        version="1.3.0",
        layers={"slope": "3DEPElevation:Slope Degrees"},
        outFormat="image/tiff",
        width=2000,
        fill_holes=True,
        in_crs="epsg:4326",
        crs="epsg:3857",
    )
    slope = services.wms_bygeom(
        url_wms,
        "3DEP",
        geometry=wshed.geometry,
        version="1.3.0",
        layers={"slope": "3DEPElevation:Slope Degrees"},
        outFormat="image/tiff",
        resolution=1,
    )

    url_wfs = (
        "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"
    )

    wfs = services.WFS(
        url_wfs,
        layer="public_NFHL:Base_Flood_Elevations",
        outFormat="esrigeojson",
        crs="epsg:4269",
    )
    r = wfs.getfeature_bybox(wshed.geometry.bounds, in_crs="epsg:4326")
    flood = utils.json_togeodf(r.json(), "epsg:4269", "epsg:4326")

    assert (
        abs(storm_pipes.length.sum() - 9.6357) < 1e-4
        and abs(slope.mean().values.item() == 118.9719) < 1e-4
        and abs(flood.length.sum() == 0.1916) < 1e-4
    )


def test_plot():
    utils.interactive_map([-70, 44, -69, 46])
    wshed = Station("2000-01-01", "2000-01-05", station_id="01031500")
    qobs = hds.nwis_streamflow(wshed.station_id, wshed.start, wshed.end)
    clm_p = hds.daymet_byloc(
        *wshed.coords, start=wshed.start, end=wshed.end, variables=["prcp"]
    )
    plot.signatures(
        {"Q": (qobs["USGS-01031500"], wshed.drainage_area)}, clm_p["prcp (mm/day)"]
    )
    cmap, norm, levels = plot.cover_legends()
    err = helpers.nwis_errors()
    fc = helpers.nhdplus_fcodes()
    assert levels[-1] == 100 and err.shape[0] == 7 and fc.shape[0] == 115


def test_acc():
    flw = utils.prepare_nhdplus(hds.NLDI.flowlines("11092450"), 1, 1, 1, True, True)

    def routing(qin, q):
        return qin + q

    qsim = utils.vector_accumulation(
        flw[["comid", "tocomid", "lengthkm"]],
        routing,
        "lengthkm",
        ["lengthkm"],
        threading=False,
    )
    flw = flw.merge(qsim, on="comid")
    diff = flw.arbolatesu - flw.acc

    assert diff.abs().sum() < 1e-5


def test_acc_threading():
    flw = utils.prepare_nhdplus(
        hds.NLDI.flowlines("11092450"), 1, 1, purge_non_dendritic=True
    )

    def routing(qin, q):
        return qin + q

    qsim = utils.vector_accumulation(
        flw[["comid", "tocomid", "lengthkm"]],
        routing,
        "lengthkm",
        ["lengthkm"],
        threading=True,
    )
    flw = flw.merge(qsim, on="comid")
    diff = flw.arbolatesu - flw.acc

    assert diff.abs().sum() < 1e-5
