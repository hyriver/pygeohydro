#!/usr/bin/env python
"""Tests for `hydrodata` package."""

import shutil
from urllib.error import HTTPError

import hydrodata.datasets as hds
from hydrodata import (
    NLDI,
    WFS,
    ArcGISREST,
    NationalMap,
    Station,
    WaterData,
    helpers,
    plot,
    services,
    utils,
)


def test_station():
    natural = Station(station_id="01031500", verbose=True)
    natural = Station(station_id="01031500", verbose=True)
    urban = Station(coords=(-118.47, 34.16), dates=("2000-01-01", "2010-01-21"))
    assert natural.hcdn and not urban.hcdn


def test_nwis():
    discharge = hds.nwis_streamflow("01031500", "2000-01-01", "2000-01-31")
    assert abs(discharge.sum().values[0] - 139.857) < 1e-3


def test_daymet():
    wshed = Station(station_id="01031500")
    start, end = "2000-01-01", "2000-01-12"
    variables = ["tmin"]
    st_p = hds.daymet_byloc(
        -118.47, 34.16, start=start, end=end, variables=variables, pet=True
    )
    yr_p = hds.daymet_byloc(-118.47, 34.16, years=2010, variables=variables)

    st_g = hds.daymet_bygeom(
        wshed.geometry, start=start, end=end, variables=variables, pet=True
    )
    yr_g = hds.daymet_bygeom(wshed.geometry, years=2010, variables=variables)
    assert (
        abs(st_g.isel(time=10, x=5, y=10).pet.values.item() - 0.682) < 1e-3
        and abs(yr_g.isel(time=10, x=5, y=10).tmin.values.item() - (-18.0)) < 1e-1
        and abs(st_p.iloc[10]["pet (mm/day)"] - 2.393) < 1e-3
        and abs(yr_p.iloc[10]["tmin (deg c)"] - 11.5) < 1e-1
    )


def test_nldi():
    station_id = "01031500"

    sid = NLDI.starting_comid(station_id)
    ids = NLDI.comids(station_id)
    trib = NLDI.tributaries(station_id)
    main = NLDI.main(station_id)
    st100 = NLDI.stations(station_id, navigation="upstreamTributaries", distance=100)
    stm = NLDI.stations(station_id, navigation="upstreamMain")
    pp = NLDI.pour_points(station_id)
    fl = utils.prepare_nhdplus(
        NLDI.flowlines(station_id), 0, 0, purge_non_dendritic=False
    )
    ct = NLDI.catchments(station_id)

    assert (
        sid == 1722317
        and ids[-1] == 1723451
        and trib.shape[0] == 432
        and main.shape[0] == 52
        and st100.shape[0] == 3
        and stm.shape[0] == 2
        and pp.shape[0] == 12
        and abs(fl.lengthkm.sum() - 565.755) < 1e-3
        and abs(ct.areasqkm.sum() - 773.954) < 1e-3
    )


def test_nhdplus_bybox():
    wd = WaterData("nhdwaterbody")
    wb = wd.features_bybox(
        (-69.7718294059999, 45.074243489, -69.314140401, 45.4533586220001),
    )
    assert abs(wb.areasqkm.sum() - 87.084) < 1e-3


def test_nhdplus_byid():
    wd = WaterData("catchmentsp")
    ct = wd.features_byid("featureid", NLDI().comids("01031500"))
    assert abs(ct.areasqkm.sum() - 773.954) < 1e-3


def test_ssebopeta():
    wshed = Station(station_id="01031500")
    start, end = "2000-01-01", "2000-01-05"
    eta_p = hds.ssebopeta_byloc(*wshed.coords, start=start, end=end)
    eta_g = hds.ssebopeta_bygeom(wshed.geometry, start=start, end=end)
    assert (
        abs(eta_p.mean().values[0] == 0.575) < 1e-3
        and abs(eta_g.mean().values.item() - 0.577) < 1e-3
    )


def test_nlcd():
    wshed = Station(station_id="01031500")
    lulc = hds.nlcd(wshed.geometry, resolution=1)
    st = utils.cover_statistics(lulc.cover)
    assert abs(st["categories"]["Forest"] - 43.094) < 1e-3


def test_nm():
    wshed = Station(station_id="01031500")
    nm = NationalMap(wshed.geometry, resolution=1)
    dem, slope, aspect = nm.get_dem(), nm.get_slope(), nm.get_aspect()
    nm.get_slope(mpm=True)
    assert (
        abs(dem.mean().values.item() - 302.237) < 1e-3
        and abs(slope.mean().values.item() - 4.180) < 1e-3
        and abs(aspect.mean().values.item() - 168.891) < 1e-3
    )


def test_newdb():
    wshed = Station(station_id="11092450")

    s = ArcGISREST(host="maps.lacity.org", site="lahub", verbose=True)
    s.spatialRel = "esriSpatialRelIntersects"
    s.folder = "Utilities"
    s.folder = None
    s.get_fs()
    s.serviceName = "Stormwater_Information"
    s.get_layers()
    s.layer = 10
    s.generate_url()
    url_rest = "https://maps.lacity.org/lahub/rest/services/Stormwater_Information/MapServer/10"
    s = ArcGISREST(url_rest, verbose=True)
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
        geometry=wshed.geometry,
        version="1.3.0",
        layers={"slope": "3DEPElevation:Slope Degrees"},
        outFormat="image/tiff",
        fpath={"slope": "tmp/slope.tiff"},
        width=2000,
        fill_holes=True,
        in_crs="epsg:4326",
        crs="epsg:3857",
    )

    slope = services.wms_bygeom(
        url_wms,
        geometry=wshed.geometry,
        version="1.3.0",
        layers={"slope": "3DEPElevation:Slope Degrees"},
        outFormat="image/tiff",
        resolution=1,
    )

    url_wfs = (
        "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"
    )

    wfs = WFS(
        url_wfs,
        layer="public_NFHL:Base_Flood_Elevations",
        outFormat="esrigeojson",
        crs="epsg:4269",
    )

    r = wfs.getfeature_bybox(wshed.geometry.bounds, in_crs="epsg:4326")
    flood = utils.json_togeodf(r.json(), "epsg:4269", "epsg:4326")

    shutil.rmtree("tmp", ignore_errors=True)

    assert (
        abs(storm_pipes.length.sum() - 9.636) < 1e-3
        and abs(slope.mean().values.item() == 118.972) < 1e-3
        and abs(flood.length.sum() == 0.192) < 1e-3
    )


def test_plot():
    hds.interactive_map([-70, 44, -69, 46])
    wshed = Station(station_id="01031500")
    start, end = "2000-01-01", "2009-12-31"
    qobs = hds.nwis_streamflow(wshed.station_id, start, end)
    clm_p = hds.daymet_byloc(*wshed.coords, start=start, end=end, variables=["prcp"])
    plot.signatures({"Q": qobs["USGS-01031500"]}, prcp=clm_p["prcp (mm/day)"])
    cmap, norm, levels = plot.cover_legends()
    assert levels[-1] == 100


def test_helpers():
    err = helpers.nwis_errors()
    try:
        fc = helpers.nhdplus_fcodes()
        assert err.shape[0] == 7 and fc.shape[0] == 115
    except (HTTPError, AttributeError):
        assert err.shape[0] == 7


def test_acc():
    flw = utils.prepare_nhdplus(NLDI.flowlines("11092450"), 1, 1, 1, True, True)

    def routing(qin, q):
        return qin + q

    qsim = utils.vector_accumulation(
        flw[["comid", "tocomid", "lengthkm"]], routing, "lengthkm", ["lengthkm"],
    )
    flw = flw.merge(qsim, on="comid")
    diff = flw.arbolatesu - flw.acc

    assert diff.abs().sum() < 1e-5


def test_ring():
    ring = {
        "rings": [
            [
                [-97.06138, 32.837],
                [-97.06133, 32.836],
                [-97.06124, 32.834],
                [-97.06127, 32.832],
                [-97.06138, 32.837],
            ],
            [
                [-97.06326, 32.759],
                [-97.06298, 32.755],
                [-97.06153, 32.749],
                [-97.06326, 32.759],
            ],
        ],
        "spatialReference": {"wkid": 4326},
    }
    _ring = utils.arcgis_togeojson(ring)
    res = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [-97.06138, 32.837],
                    [-97.06127, 32.832],
                    [-97.06124, 32.834],
                    [-97.06133, 32.836],
                    [-97.06138, 32.837],
                ]
            ],
            [
                [
                    [-97.06326, 32.759],
                    [-97.06298, 32.755],
                    [-97.06153, 32.749],
                    [-97.06326, 32.759],
                ]
            ],
        ],
    }
    assert _ring == res


def test_point():
    point = {"x": -118.15, "y": 33.80, "z": 10.0, "spatialReference": {"wkid": 4326}}
    _point = utils.arcgis_togeojson(point)
    res = {"type": "Point", "coordinates": [-118.15, 33.8, 10.0]}
    assert _point == res


def test_multipoint():
    mpoint = {
        "hasZ": "true",
        "points": [
            [-97.06138, 32.837, 35.0],
            [-97.06133, 32.836, 35.1],
            [-97.06124, 32.834, 35.2],
        ],
        "spatialReference": {"wkid": 4326},
    }
    _mpoint = utils.arcgis_togeojson(mpoint)
    res = {
        "type": "MultiPoint",
        "coordinates": [
            [-97.06138, 32.837, 35.0],
            [-97.06133, 32.836, 35.1],
            [-97.06124, 32.834, 35.2],
        ],
    }
    assert _mpoint == res


def test_path():
    path = {
        "hasM": "true",
        "paths": [
            [
                [-97.06138, 32.837, 5],
                [-97.06133, 32.836, 6],
                [-97.06124, 32.834, 7],
                [-97.06127, 32.832, 8],
            ],
            [[-97.06326, 32.759], [-97.06298, 32.755]],
        ],
        "spatialReference": {"wkid": 4326},
    }
    _path = utils.arcgis_togeojson(path)
    res = {
        "type": "MultiLineString",
        "coordinates": [
            [
                [-97.06138, 32.837, 5],
                [-97.06133, 32.836, 6],
                [-97.06124, 32.834, 7],
                [-97.06127, 32.832, 8],
            ],
            [[-97.06326, 32.759], [-97.06298, 32.755]],
        ],
    }
    assert _path == res


def test_wbd():
    wbd = ArcGISREST(
        base_url="https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/1"
    )
    wbd.max_nrecords = 5
    wbd.featureids = [str(n) for n in range(1, 21)]
    wbd.outFields = ["HUC2", "NAME", "SHAPE_Area"]
    f = wbd.get_features()
    assert f.shape[0] == len([x for y in wbd.featureids for x in y])
