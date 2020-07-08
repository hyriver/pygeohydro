"""Base classes and function for REST, WMS, and WMF services."""
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from warnings import warn

import geopandas as gpd
import xarray as xr
from defusedxml import cElementTree as ET
from owslib.map.wms111 import WebMapService_1_1_1
from owslib.map.wms130 import WebMapService_1_3_0
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
    """Access to an ArcGIS REST service.

    Parameters
    ----------
    base_url : str, optional
        The ArcGIS RESTful service url.
    outFormat : str, optional
        One of the output formats offered by the selected layer. If not correct
        a list of available formats is shown, defaults to ``geojson``.
    outFields : str or list
        The output fields to be requested. Setting ``*`` as outFields requests
        all the available fields which is the default behaviour.
    n_threads : int, optional
        Number of simultaneous download, default to 4.
    """

    def __init__(
        self,
        base_url: str,
        outFormat: str = "geojson",
        outFields: Union[List[str], str] = "*",
        n_threads: int = 4,
    ) -> None:

        self.session = RetrySession()
        self.base_url = base_url
        self.test_url()

        self._outFormat = outFormat
        self._outFields = outFields if isinstance(outFields, list) else [outFields]
        self._n_threads = n_threads
        self.nfeatures = 0

    @property
    def outFormat(self) -> str:
        return self._outFormat

    @outFormat.setter
    def outFormat(self, value: str) -> None:
        if self.base_url is not None and value.lower() not in self.queryFormats:
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

    def test_url(self) -> None:
        """Test the generated url and get the required parameters from the service."""
        try:
            r = self.session.get(self.base_url, {"f": "json"}).json()
            try:
                self.units = r["units"].replace("esri", "").lower()
            except KeyError:
                self.units = None
            self.maxRecordCount = r["maxRecordCount"]
            self.queryFormats = r["supportedQueryFormats"].replace(" ", "").lower().split(",")
            self.valid_fields = list(
                set(
                    utils.traverse_json(r, ["fields", "name"])
                    + utils.traverse_json(r, ["fields", "alias"])
                    + ["*"]
                )
            )
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
    def featureids(self, value: Union[List[int], int]) -> None:
        if not isinstance(value, (list, int)):
            raise InvalidInputType("featureids", "int or list")

        oids = [str(value)] if isinstance(value, int) else [str(v) for v in value]

        self.nfeatures = len(oids)
        if self.nfeatures == 0:
            ZeroMatched("No feature ID were found within the requested region.")

        oid_list = list(zip_longest(*[iter(oids)] * self.max_nrecords))
        oid_list[-1] = tuple(i for i in oid_list[-1] if i is not None)
        self._featureids = oid_list

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
        """Get feature IDs withing a geometry."""

        geom = utils.match_crs(geom, geo_crs, "epsg:4326")

        if isinstance(geom, tuple):
            geom_query = utils.ESRIGeomQuery(geom, 4326).bbox()
        else:
            geom_query = utils.ESRIGeomQuery(geom, 4326).polygon()  # type: ignore

        payload = {
            **geom_query,
            "spatialRel": "esriSpatialRelIntersects",
            "returnGeometry": "false",
            "returnIdsOnly": "true",
            "f": self.outFormat,
        }
        r = self.session.post(f"{self.base_url}/query", payload)

        try:
            self.featureids = r.json()["objectIds"]
        except (KeyError, TypeError, IndexError, JSONDecodeError):
            raise ZeroMatched("No feature ID were found within the requested region.")

    def get_features(self) -> gpd.GeoDataFrame:
        """Get features based on the feature IDs."""

        if not all(f in self.valid_fields for f in self.outFields):
            raise InvalidInputValue("outFields", self.valid_fields)

        payload = {
            "returnGeometry": "true",
            "outSR": "4326",
            "outFields": ",".join(self.outFields),
            "f": self.outFormat,
        }

        def getter(ids: Tuple[str, ...]) -> Union[Response, Tuple[str, ...]]:
            payload.update({"objectIds": ",".join(ids)})
            r = self.session.post(f"{self.base_url}/query", payload)
            r_json = r.json()
            try:
                if "error" in r_json:
                    return ids

                return r_json
            except AssertionError:
                if self.outFormat == "geojson":
                    raise ZeroMatched(
                        "There was a problem processing the request with geojson outFormat. "
                        + "Your can set the outFormat to json and retry."
                    )

                raise ZeroMatched("No matching data was found on the server.")

        feature_list = utils.threading(getter, self.featureids, max_workers=self.n_threads)

        # Split the list based on type which are tuple and dict
        feature_types = defaultdict(list)
        for f in feature_list:
            feature_types[type(f)].append(f)

        features = feature_types[dict]

        if len(feature_types[tuple]) > 0:
            failed = [tuple(x) for y in feature_types[tuple] for x in y]
            retry = utils.threading(getter, failed, max_workers=self.n_threads * 2)
            fixed = [resp for resp in retry if not isinstance(resp, tuple)]

            nfailed = len(failed) - len(fixed)
            if nfailed > 0:
                warn(
                    f"From {self.nfeatures} requetsed features, {nfailed} were not available on the server."
                )

            features += fixed

        if len(features) == 0:
            raise ZeroMatched("No valid feature was found.")

        data = utils.json_togeodf(features[0], "epsg:4326")

        return data.append([utils.json_togeodf(f, "epsg:4326") for f in features[1:]])


def wms_bygeom(
    url: str,
    layers: Dict[str, str],
    outFormat: str,
    geometry: Union[Polygon, Tuple[float, float, float, float]],
    resolution: float,
    geo_crs: str = "epsg:4326",
    crs: str = "epsg:4326",
    version: str = "1.3.0",
    fill_holes: bool = False,
    fpath: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """Data from a WMS service within a geometry or bounding box.

    Parameters
    ----------
    url : str
        The base url for the WMS service e.g., https://www.mrlc.gov/geoserver/mrlc_download/wms
    layers : dict
        The layer from the service to be downloaded. The argument should be a
        dict with keys as the variable name in the output dataframe and values
        as the complete name of the layer in the service. You can pass an empty
        string to get a list of available layers.
    outFormat : str
        The data format to request for data from the service. You can pass an empty
        string to get a list of available output formats.
    geometry : Polygon or tuple
        A shapely Polygon or bounding box for getting the data.
    resolution : float
        The output resolution in meters. The width and height of output are computed in pixel
        based on the geometry bounds and the given resolution.
    geo_crs : str, optional
        The spatial reference system of the input geometry, defaults to
        epsg:4326.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    version : str, optional
        The WMS service version which should be either 1.1.1 or 1.3.0, defaults to 1.3.0.
    fill_holes : bool, optional
        Wether to fill the holes in the Polygon's interior, defaults to False.
    fpath : dict, optional
        The path to save the downloaded images, defaults to None which will only return
        the data as ``xarray.Dataset`` and doesn't save the files. The argument should be
        a dict with keys as the variable name in the output dataframe and values as
        the path to save to the file.

    Returns
    -------
    xarray.Dataset
        Requested layer data within a geometry or bounding box
    """

    wms = WebMapService(url, version=version)

    validate_wms(wms, layers, outFormat, crs)

    if isinstance(geometry, Polygon):
        if fill_holes:
            geometry = Polygon(geometry.exterior)
        bounds = utils.match_crs(geometry.bounds, geo_crs, crs)
    else:
        bounds = utils.match_crs(geometry, geo_crs, crs)

    width, height = utils.bbox_resolution(bounds, resolution, crs)

    if fpath is not None and not isinstance(fpath, dict):
        raise InvalidInputType("fpath", "dict", "{var_name : path}")

    if fpath is None:
        fpath = {k: None for k in layers.keys()}
    else:
        if layers.keys() != fpath.keys():
            raise ValueError("Keys of fpath and layers dictionaries should be the same.")
        utils.check_dir(fpath.values())

    def _wms(inp):
        name, layer = inp
        img = wms.getmap(
            layers=[layer], srs=crs, bbox=bounds, size=(width, height), format=outFormat,
        )
        return (name, img.read())

    resp = utils.threading(_wms, layers.items(), max_workers=len(layers))

    _geometry = utils.match_crs(geometry, geo_crs, crs)
    data = utils.create_dataset(resp[0][1], _geometry, resp[0][0], fpath[resp[0][0]])

    if len(resp) > 1:
        for name, r in resp:
            da = utils.create_dataset(r, _geometry, name, fpath[name])
            data = xr.merge([data, da])

    return data


def validate_wms(
    wms: Union[WebMapService_1_3_0, WebMapService_1_1_1],
    layers: Union[str, Dict[str, str]],
    outFormat: str,
    crs: str,
):
    """Validate query inputs for a WMS request."""

    valid_layers = {wms[layer].name: wms[layer].title for layer in list(wms.contents)}

    if not isinstance(layers, dict):
        raise InvalidInputType("layers", "dict", "{var_name : layer_name}")

    if any(layer not in valid_layers.keys() for layer in layers.values()):
        raise InvalidInputValue("layer", (f"{n} for {t}" for n, t in valid_layers.items()))

    valid_outFormats = wms.getOperationByName("GetMap").formatOptions
    if outFormat is None or outFormat not in valid_outFormats:
        raise InvalidInputValue("outFormat", valid_outFormats)

    valid_crss = {layer: [s.lower() for s in wms[layer].crsOptions] for layer in layers.values()}
    if any(crs not in valid_crss[layer] for layer in layers.values()):
        _valid_crss = (f"{lyr}: {', '.join(cs)}\n" for lyr, cs in valid_crss.items())
        raise InvalidInputValue("CRS", _valid_crss)


class WFS:
    """Data from any WFS service within a geometry or by featureid.

    Parameters
    ----------
    url : str
        The base url for the WFS service, for examples:
        https://hazards.fema.gov/nfhl/services/public/NFHL/MapServer/WFSServer
    layer : str
        The layer from the service to be downloaded, defaults to None which throws
        an error and includes all the available layers offered by the service.
    outFormat : str
        The data format to request for data from the service, defaults to None which
         throws an error and includes all the available format offered by the service.
    version : str, optional
        The WFS service version which should be either 1.1.1, 1.3.0, or 2.0.0.
        Defaults to 2.0.0.
    crs: str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    validation : bool
        Validate the input arguments from the WFS service, defaults to True. Set this
        to False if you are sure all the WFS settings such as layer and crs are correct
        to avoid sending extra requests.
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
        self.session = RetrySession()

        if validation:
            self.validate_wfs()

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

    def validate_wfs(self):
        wfs = WebFeatureService(self.url, version=self.version)

        valid_layers = list(wfs.contents)
        if self.layer is None:
            raise MissingInputs(
                "The layer argument is missing."
                + " The following layers are available:\n"
                + ", ".join(valid_layers)
            )

        if self.layer not in valid_layers:
            raise InvalidInputValue("layers", valid_layers)

        valid_outFormats = wfs.getOperationByName("GetFeature").parameters["outputFormat"]["values"]
        valid_outFormats = [v.lower() for v in valid_outFormats]
        if self.outFormat is None:
            raise MissingInputs(
                "The outFormat argument is missing."
                + " The following output formats are available:\n"
                + ", ".join(valid_outFormats)
            )

        if self.outFormat not in valid_outFormats:
            raise InvalidInputValue("outFormat", valid_outFormats)

        valid_crss = [f"{s.authority.lower()}:{s.code}" for s in wfs[self.layer].crsOptions]
        if self.crs.lower() not in valid_crss:
            raise InvalidInputValue("crs", valid_crss)

    def get_validnames(self) -> Response:
        """Get valid column names for a layer."""

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
        """Data from any WMS service within a geometry.

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
            WFS query response within a bounding box.
        """

        utils.check_bbox(bbox)
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
        """Get features based on feature IDs.

        Parameters
        ----------
        featurename : str
            The name of the column for searching for feature IDs
        featureids : str or list
            The feature ID(s)
        filter_spec : str
            The OGC filter spec, defaults to "1.1". Supported versions are
            1.1 and 2.0.

        Returns
        -------
        requests.Response
            WMS query response
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


class ServiceURL:
    """Base URLs of the supported services."""

    @property
    def restful(self):
        return RESTfulURLs()

    @property
    def wms(self):
        return WMSURLs()

    @property
    def wfs(self):
        return WFSURLs()

    @property
    def http(self):
        return HTTPURLs()


class RESTfulURLs(NamedTuple):
    """A list of RESTful services URLs."""

    nwis: str = "https://waterservices.usgs.gov/nwis"
    nldi: str = "https://labs.waterdata.usgs.gov/api/nldi/linked-data"
    daymet_point: str = "https://daymet.ornl.gov/single-pixel/api/data"
    daymet_grid: str = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328"
    wbd: str = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"
    fws: str = "https://www.fws.gov/wetlands/arcgis/rest/services"
    fema: str = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer"


class WMSURLs(NamedTuple):
    """A list of WMS services URLs."""

    mrlc: str = "https://www.mrlc.gov/geoserver/mrlc_download/wms"
    fema: str = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHLWMS/MapServer/WMSServer"
    nm_3dep: str = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"
    fws: str = "https://www.fws.gov/wetlands/arcgis/services/Wetlands_Raster/ImageServer/WMSServer"


class WFSURLs(NamedTuple):
    """A list of WFS services URLs."""

    waterdata: str = "https://labs.waterdata.usgs.gov/geoserver/wmadata/ows"
    fema: str = "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"


class HTTPURLs(NamedTuple):
    """A list of HTTP services URLs."""

    ssebopeta: str = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/daily/downloads"
