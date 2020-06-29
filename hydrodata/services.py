"""Base classes and function for REST, WMS, and WMF services."""

from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import defusedxml.cElementTree as ET
import geopandas as gpd
import pandas as pd
import simplejson as json
import xarray as xr
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService
from requests import Response
from shapely.geometry import Polygon
from simplejson import JSONDecodeError

from hydrodata import utils
from hydrodata.connection import RetrySession
from hydrodata.exceptions import (
    InvalidInputType,
    InvalidInputValue,
    MissingInputs,
    ServerError,
    ZeroMatched,
)


class ArcGISREST:
    """Access to an ArcGIS REST sercive.

    Attributes
    ----------
    base_url : str, optional
        The url as a whole rather than in parts, defaults to None. Instead of passing this
        argument you can pass host and site arguments and explor available folders, service,
        and layers, then set them separately.
    outFormat : str, optional
        One of the output formats offered by the selected layer. If not correct
        a list of available formats is shown, defaults to ``geojson``.
    outFields : str or list
        The output fields to be requested. Setting ``*`` as outFields requests
        all the available fields. Defaults to ``*``.
    spatialRel : str, optional
        The spatial relationship to be applied on the input geometry
        while performing the query. If not correct a list of available options is shown.
        Defaults to ``esriSpatialRelIntersects``.
    n_threads : int, optional
        Number of simultanious download, default to 4.
    """

    def __init__(
        self,
        base_url: str,
        outFormat: str = "geojson",
        outFields: Union[List[str], str] = "*",
        spatialRel: str = "esriSpatialRelIntersects",
        n_threads: int = 4,
    ) -> None:

        self.session = RetrySession()
        self.base_url = base_url
        self.test_url()

        self._outFormat = outFormat
        self._outFields = outFields if isinstance(outFields, list) else [outFields]
        self._spatialRel = spatialRel
        self._n_threads = n_threads

    @property
    def outFormat(self) -> str:
        return self._outFormat

    @outFormat.setter
    def outFormat(self, value: str) -> None:
        if self.base_url is not None and value not in self.queryFormats:
            raise InvalidInputValue("outFormat", self.queryFormats)

        self._outFormat = value

    @property
    def outFields(self) -> List[str]:
        return self._outFields

    @outFields.setter
    def outFields(self, value: Union[List[str], str]) -> None:
        if not isinstance(value, (list, str)):
            raise InvalidInputType("outFields", "str or list")

        self._outFields = value if isinstance(value, list) else [value]

    @property
    def spatialRel(self) -> str:
        return self._spatialRel

    @spatialRel.setter
    def spatialRel(self, value: str) -> None:
        valid_spatialRels = [
            "esriSpatialRelIntersects",
            "esriSpatialRelContains",
            "esriSpatialRelCrosses",
            "esriSpatialRelEnvelopeIntersects",
            "esriSpatialRelIndexIntersects",
            "esriSpatialRelOverlaps",
            "esriSpatialRelTouches",
            "esriSpatialRelWithin",
            "esriSpatialRelRelation",
        ]
        if value not in valid_spatialRels:
            raise InvalidInputValue("spatialRel", valid_spatialRels)

        self._spatialRel = value

    def test_url(self) -> None:
        """Test the generated url and get the required parameters from the service"""
        try:
            r = self.session.get(self.base_url, {"f": "json"}).json()
            try:
                self.units = r["units"].replace("esri", "").lower()
            except KeyError:
                self.units = None
            self.maxRecordCount = r["maxRecordCount"]
            self.queryFormats = r["supportedQueryFormats"].replace(" ", "").lower().split(",")
            self.valid_fields = utils.traverse_json(r, ["fields", "name"]) + ["*"]
        except KeyError:
            raise ServerError(self.base_url)

        self._max_nrecords = self.maxRecordCount

    @property
    def n_threads(self) -> int:
        return self._n_threads

    @n_threads.setter
    def n_threads(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise InvalidInputType("n_threads", "positive int")
        self._n_threads = value

    @property
    def max_nrecords(self) -> int:
        return self._max_nrecords

    @max_nrecords.setter
    def max_nrecords(self, value: int) -> None:
        if value > self.maxRecordCount:
            raise ValueError(
                f"The server doesn't accept more than {self.maxRecordCount}"
                + " records per request."
            )
        if value > 0:
            self._max_nrecords = value
        else:
            raise InvalidInputType("max_nrecords", "positive int")

    @property
    def featureids(self) -> List[Tuple[str, ...]]:
        return self._featureids

    @featureids.setter
    def featureids(self, value: Union[List[str], str]) -> None:
        if isinstance(value, (list, str)):
            oids = [value] if isinstance(value, str) else value
            oid_list = list(zip_longest(*[iter(oids)] * self.max_nrecords))
            oid_list[-1] = tuple(i for i in oid_list[-1] if i is not None)
            self._featureids = oid_list
        else:
            raise InvalidInputType("featureids", "str or list")

    def __repr__(self) -> str:
        """Print the service configuration."""
        return (
            "Service configurations:\n"
            + f"URL: {self.base_url}\n"
            + f"Max Record Count: {self.maxRecordCount}\n"
            + f"Supported Query Formats: {self.queryFormats}\n"
            + f"Units: {self.units}"
        )

    def get_featureids(
        self, geom: Union[Polygon, Tuple[float, float, float, float]], geo_crs: str = "epsg:4326",
    ) -> None:
        """Get feature IDs withing a geometry"""

        if not isinstance(geom, (Polygon, tuple)):
            raise InvalidInputType("geometry", "tuple or Polygon")

        geom = utils.match_crs(geom, geo_crs, "epsg:4326")

        if isinstance(geom, tuple):
            if len(geom) != 4:
                raise InvalidInputType("geom (bounding box)", "tuple", "(west, south, east, north)")

            geometryType = "esriGeometryEnvelope"
            bbox = dict(zip(("xmin", "ymin", "xmax", "ymax"), geom))
            bbox_json = {**bbox, "spatialRelference": {"wkid": 4326}}
            geometry = json.dumps(bbox_json)
        elif isinstance(geom, Polygon):
            geometryType = "esriGeometryPolygon"
            geometry_json = {
                "rings": [[[x, y] for x, y in zip(*geom.exterior.coords.xy)]],
                "spatialRelference": {"wkid": 4326},
            }
            geometry = json.dumps(geometry_json)

        payload = {
            "geometryType": geometryType,
            "geometry": geometry,
            "inSR": "4326",
            "spatialRel": self.spatialRel,
            "returnGeometry": "false",
            "returnIdsOnly": "true",
            "f": self.outFormat,
        }
        r = self.session.post(f"{self.base_url}/query", payload)
        try:
            oids = r.json()["objectIds"]
        except (KeyError, TypeError, IndexError, JSONDecodeError):
            raise ZeroMatched(
                "No feature ID were found within the requested "
                + f"region using the spatial relationship {self.spatialRel}."
            )

        self.featureids = oids

    def get_features(self) -> gpd.GeoDataFrame:
        """Get features based on the feature IDs"""

        if not all(f in self.valid_fields for f in self.outFields):
            raise InvalidInputValue("outFields", self.valid_fields)

        def get_geojson(ids: Tuple[str, ...]) -> Union[Response, Tuple[str, ...]]:
            payload = {
                "objectIds": ",".join(str(i) for i in ids),
                "returnGeometry": "true",
                "outSR": "4326",
                "outFields": ",".join(str(i) for i in self.outFields),
                "f": self.outFormat,
            }
            r = self.session.post(f"{self.base_url}/query", payload)
            try:
                return utils.json_togeodf(r.json(), "epsg:4326")
            except TypeError:
                return ids
            except AssertionError:
                raise ZeroMatched(
                    "There was a problem processing GeoJSON, " + "try setting outFormat to json"
                )

        def get_json(ids: Tuple[str, ...]) -> Union[Response, Tuple[str, ...]]:
            payload = {
                "objectIds": ",".join(str(i) for i in ids),
                "returnGeometry": "true",
                "outSR": "4326",
                "outFields": ",".join(str(i) for i in self.outFields),
                "f": self.outFormat,
            }
            r = self.session.post(f"{self.base_url}/query", payload)
            try:
                return utils.json_togeodf(r.json(), "epsg:4326")
            except TypeError:
                return ids

        getter = get_json if self.outFormat == "json" else get_geojson

        feature_list = utils.threading(getter, self.featureids, max_workers=self.n_threads,)

        # Find the failed batches and retry
        fails = [ids for ids in feature_list if isinstance(ids, tuple)]
        success = [ids for ids in feature_list if isinstance(ids, gpd.GeoDataFrame)]

        if len(fails) > 0:
            failed = [tuple(x) for y in fails for x in y]
            retry = utils.threading(getter, failed, max_workers=self.n_threads,)
            success += [ids for ids in retry if isinstance(ids, gpd.GeoDataFrame)]

        if len(success) == 0:
            raise ZeroMatched("No valid feature was found.")

        data = gpd.GeoDataFrame(pd.concat(success)).reset_index(drop=True)
        data.crs = "epsg:4326"
        return data


def wms_bygeom(
    url: str,
    geometry: Polygon,
    geo_crs: str = "epsg:4326",
    width: Optional[int] = None,
    resolution: Optional[float] = None,
    layers: Optional[Dict[str, str]] = None,
    outFormat: Optional[str] = None,
    fpath: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
    version: str = "1.3.0",
    crs: str = "epsg:4326",
    fill_holes: bool = False,
    validation: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """Data from a WMS service within a geometry

    Parameters
    ----------
    url : str
        The base url for the WMS service e.g., https://www.mrlc.gov/geoserver/mrlc_download/wms
    geometry : Polygon
        A shapely Polygon for getting the data.
    geo_crs : str, optional
        The spatial reference system of the input geometry, defaults to
        epsg:4326.
    width : int
        The width of the output image in pixels. The height is computed
        automatically from the geometry's bounding box aspect ratio. Either width
        or resolution should be provided.
    resolution : float
        The data resolution in arc-seconds. The width and height are computed in pixel
        based on the geometry bounds and the given resolution. Either width or
        resolution should be provided.
    layers : dict
        The layer from the service to be downloaded, defaults to None which throws
        an error and includes all the avialable layers offered by the service. The
        argument should be a dict with keys as the variable name in the output
        dataframe and values as the complete name of the layer in the service.
    outFormat : str
        The data format to request for data from the service, defaults to None which
         throws an error and includes all the avialable format offered by the service.
    fpath : dict, optional
        The path to save the downloaded images, defaults to None which will only return
        the data as ``xarray.Dataset`` and doesn't save the files. The argument should be
        a dict with keys as the variable name in the output dataframe and values as
        the path to save to the file.
    version : str, optional
        The WMS service version which should be either 1.1.1 or 1.3.0, defaults to 1.3.0.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    fill_holes : bool, optional
        Wether to fill the holes in the geometry's interior, defaults to False.
    validation : bool
        Validate the input arguments from the WFS service, defaults to True. Set this
        to False if you are sure all the WMS settings such as layer and crs are correct
        to avoid sending extra requestes.

    Returns
    -------
    xarray.Dataset
    """
    for _ in range(3):
        try:
            wms = WebMapService(url, version=version)
            break
        except ConnectionError:
            continue

    if validation:
        valid_layers = {wms[layer].name: wms[layer].title for layer in list(wms.contents)}
        if layers is None:
            raise MissingInputs(
                "The layers argument is missing."
                + " The following layers are available:\n"
                + "\n".join(f"{name}: {title}" for name, title in valid_layers.items())
            )

        if isinstance(layers, dict):
            _layers = layers
            if any(str(layer) not in valid_layers.keys() for layer in _layers.values()):
                raise InvalidInputValue("layer", (f"{n}: {t}\n" for n, t in valid_layers.items()))
        else:
            raise InvalidInputType("layers", "dict", "{var_name : layer_name}")

        valid_outFormats = wms.getOperationByName("GetMap").formatOptions
        if outFormat is None:
            raise MissingInputs(
                "The outFormat argument is missing."
                + " The following output formats are available:\n"
                + ", ".join(fmt for fmt in valid_outFormats)
            )

        if outFormat not in valid_outFormats:
            raise InvalidInputValue("outFormat", valid_outFormats)

        valid_crss = {
            layer: [s.lower() for s in wms[layer].crsOptions] for layer in _layers.values()
        }
        if any(crs not in valid_crss[layer] for layer in _layers.values()):
            _valid_crss = (f"{lyr}: {', '.join(c for c in cs)}\n" for lyr, cs in valid_crss.items())
            raise InvalidInputValue("CRS", _valid_crss)

    if not isinstance(geometry, Polygon):
        raise InvalidInputType("geometry", "Shapley's Polygon")

    if fill_holes:
        geometry = Polygon(geometry.exterior)

    geometry = utils.match_crs(geometry, geo_crs, crs)

    if (width is None and resolution is None) or (width is not None and resolution is not None):
        raise MissingInputs("Either width or resolution should be provided.")

    west, south, east, north = geometry.bounds
    _width = int((east - west) * 3600 / resolution) if width is None else width
    height = int(abs(north - south) / abs(east - west) * _width)

    if isinstance(fpath, dict):
        _fpath = fpath
        if not all(layer in _layers.keys() for layer in _fpath.keys()):
            raise ValueError("Keys of ``fpath`` and ``layers`` dictionaries should be the same.")
        utils.check_dir(_fpath.values())
    elif fpath is None:
        _fpath = {k: None for k in _layers.keys()}
    else:
        raise InvalidInputType("fpath", "dict", "{var_name : path}")

    mask, transform = utils.geom_mask(geometry, _width, height)

    def _wms(inp):
        name, layer = inp
        img = wms.getmap(
            layers=[layer], srs=crs, bbox=geometry.bounds, size=(_width, height), format=outFormat,
        )
        return (name, img.read())

    resp = utils.threading(_wms, _layers.items(), max_workers=len(_layers),)

    data = utils.create_dataset(
        resp[0][1], mask, transform, _width, height, resp[0][0], _fpath[resp[0][0]]
    )

    if len(resp) > 1:
        for name, r in resp:
            da = utils.create_dataset(r, mask, transform, _width, height, name, _fpath[name])
            data = xr.merge([data, da])
    return data


class WFS:
    """Data from any WFS service within a geometry or by featureid

    Attributes
    ----------
    url : str
        The base url for the WFS service, for examples:
        https://hazards.fema.gov/nfhl/services/public/NFHL/MapServer/WFSServer
    layer : str
        The layer from the service to be downloaded, defaults to None which throws
        an error and includes all the avialable layers offered by the service.
    outFormat : str
        The data format to request for data from the service, defaults to None which
         throws an error and includes all the avialable format offered by the service.
    version : str, optional
        The WFS service version which should be either 1.1.1, 1.3.0, or 2.0.0.
        Defaults to 2.0.0.
    crs: str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    validation : bool
        Validate the input arguments from the WFS service, defaults to True. Set this
        to False if you are sure all the WFS settings such as layer and crs are correct
        to avoid sending extra requestes.
    """

    def __init__(
        self,
        url: str,
        layer: Optional[str] = None,
        outFormat: Optional[str] = None,
        version: str = "2.0.0",
        crs: str = "epsg:4326",
        validation: bool = True,
    ) -> None:
        self.url = url
        self.layer = layer
        self.outFormat = outFormat
        self.version = version
        self.crs = crs

        if validation:
            for _ in range(3):
                try:
                    wfs = WebFeatureService(url, version=version)
                    break
                except ConnectionError:
                    continue

            valid_layers = list(wfs.contents)
            valid_layers_lower = [layer.lower() for layer in valid_layers]
            if layer is None:
                raise MissingInputs(
                    "The layer argument is missing."
                    + " The following layers are available:\n"
                    + ", ".join(layer for layer in valid_layers)
                )

            if layer.lower() not in valid_layers_lower:
                raise InvalidInputValue("layers", valid_layers)

            valid_outFormats = wfs.getOperationByName("GetFeature").parameters["outputFormat"][
                "values"
            ]
            valid_outFormats = [f.lower() for f in valid_outFormats]
            if outFormat is None:
                raise MissingInputs(
                    "The outFormat argument is missing."
                    + " The following output formats are available:\n"
                    + ", ".join(fmt for fmt in valid_outFormats)
                )

            if outFormat.lower() not in valid_outFormats:
                raise InvalidInputValue("outFormat", valid_outFormats)

            valid_crss = [f"{s.authority.lower()}:{s.code}" for s in wfs[layer].crsOptions]
            if crs.lower() not in valid_crss:
                raise InvalidInputValue("crs", valid_crss)

            self.session = RetrySession()

    def __repr__(self) -> str:
        """Print the services properties."""
        return (
            "Connected to the WFS service with the following properties:\n"
            + f"URL: {self.url}\n"
            + f"Version: {self.version}\n"
            + f"Layer: {self.layer}\n"
            + f"Output Format: {self.outFormat}\n"
            + f"Output CRS: {self.crs}"
        )

    def get_validnames(self) -> Response:
        """Get valid column names for a layer"""

        max_features = "count" if self.version == "2.0.0" else "maxFeatures"

        payload = {
            "service": "wfs",
            "version": self.version,
            "outputFormat": self.outFormat,
            "request": "GetFeature",
            "typeName": self.layer,
            max_features: "1",
        }

        r = self.session.get(self.url, payload)

        if r.headers["Content-Type"] == "application/xml":
            root = ET.fromstring(r.text)
            raise ZeroMatched(root[0][0].text.strip())
        return r

    def getfeature_bybox(
        self, bbox: Tuple[float, float, float, float], box_crs: str = "epsg:4326"
    ) -> Response:
        """Data from any WMS service within a geometry

        Parameters
        ----------
        bbox : tuple
            A bounding box for getting the data: [west, south, east, north]
        box_crs : str, optional
            The spatial reference system of the input bbox, defaults to
            epsg:4326.

        Returns
        -------
        requests.Response
        """

        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise InvalidInputType("bbox", "tuple", "(west, south, east, north)")

        bbox = utils.match_crs(bbox, box_crs, self.crs)

        payload = {
            "service": "wfs",
            "version": self.version,
            "outputFormat": self.outFormat,
            "request": "GetFeature",
            "typeName": self.layer,
            "bbox": ",".join(str(c) for c in bbox) + f",{self.crs}",
        }

        r = self.session.get(self.url, payload)

        if r.headers["Content-Type"] == "application/xml":
            root = ET.fromstring(r.text)
            raise ZeroMatched(root[0][0].text.strip())
        return r

    def getfeature_byid(
        self, featurename: str, featureids: Union[List[str], str], filter_spec: str = "1.1",
    ) -> Response:
        """Get features based on feature IDs

        Parameters
        ----------
        featurename : str
            The name of the column for searching for feature IDs
        featureids : str or list
            The feature ID(s)
        filter_spec : str
            The OGC filter spec, defaults to "1.1". Supported vesions are
            1.1 and 2.0.

        Returns
        -------
        requests.Response
        """

        featureids = featureids if isinstance(featureids, list) else [featureids]

        if len(featureids) == 0:
            raise InvalidInputType("featureids", "int or str or list")

        fspecs = ["2.0", "1.1"]
        if filter_spec not in fspecs:
            raise InvalidInputValue("filter_spec", fspecs)

        def filter_xml1(pname, pid):
            fstart = '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc"><ogc:Or>'
            fend = "</ogc:Or></ogc:Filter>"
            return (
                fstart
                + "".join(
                    [
                        f"<ogc:PropertyIsEqualTo><ogc:PropertyName>{pname}"
                        + f"</ogc:PropertyName><ogc:Literal>{p}"
                        + "</ogc:Literal></ogc:PropertyIsEqualTo>"
                        for p in pid
                    ]
                )
                + fend
            )

        def filter_xml2(pname, pid):
            fstart = '<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0"><fes:Or>'
            fend = "</fes:Or></fes:Filter>"
            return (
                fstart
                + "".join(
                    [
                        f"<fes:PropertyIsEqualTo><fes:ValueReference>{pname}"
                        + f"</fes:ValueReference><fes:Literal>{p}"
                        + "</fes:Literal></fes:PropertyIsEqualTo>"
                        for p in pid
                    ]
                )
                + fend
            )

        fxml = filter_xml1 if filter_spec == "1.1" else filter_xml2

        payload = {
            "service": "wfs",
            "version": self.version,
            "outputFormat": self.outFormat,
            "request": "GetFeature",
            "typeName": self.layer,
            "srsName": self.crs,
            "filter": fxml(featurename, featureids),
        }

        r = self.session.post(self.url, payload)

        if r.headers["Content-Type"] == "application/xml":
            root = ET.fromstring(r.text)
            raise ZeroMatched(root[0][0].text.strip())

        return r
