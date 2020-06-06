#!/usr/bin/env python
"""Base classes and function for REST, WMS, and WMF services."""

from itertools import zip_longest
from warnings import warn

import defusedxml.cElementTree as ET
import geopandas as gpd
import pandas as pd
import pyproj
import shapely.ops as ops
import simplejson as json
import xarray as xr
from lxml import html
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService
from pqdm.threads import pqdm
from requests.exceptions import ConnectionError, RetryError
from shapely.geometry import Polygon, box

from hydrodata import utils


class ArcGISServer:
    """Base class for web services based on ArcGIS REST."""

    def __init__(
        self,
        host,
        site,
        folder=None,
        serviceName=None,
        layer=None,
        outFormat="geojson",
        spatialRel="esriSpatialRelIntersects",
    ):
        """Form the base url and get the service information.

        Notes
        -----
        The general url is in the following form ofr RESTful services:
        https://<host>/<site>/rest/services/<folder>/<serviceName>/<serviceType>/<layer>/
        https://<host>/<site>/rest/services/<serviceName>/<serviceType>/<layer>/
        and for OGC interfaces:
        https://<host>/<site>/services/<serviceName>/<serviceType>/<OGCType>/
        For more information visit: `ArcGIS <https://developers.arcgis.com/rest/services-reference/get-started-with-the-services-directory.htm>`_
        An example is:
        https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer

        Parameters
        ----------
        host : string
            The host part of the URL, e.g., elevation.nationalmap.gov
        site : string
            The site part of the URL, e.g., arcgis
        folder : string
            One of the available folders offered by the host. If not correct
            a list of available folders is shown.
        serviceName : string
            One of the available services offered by the host. If not correct
            a list of available services is shown.
        layer : string
            One of the layer of the requested service name. If not correct
            a list of available layers is shown.
        outFormat : string
            One of the output formats offered by the selected layer. If not correct
            a list of available formats is shown.
        spatialRel : string
            The spatial relationship to be applied on the input geometry
            while performing the query. If not correct
            a list of available options is shown.
        """
        if host is not None and site is not None:
            self.root = f"https://{host}/{site}/rest/services"
        else:
            self.root = None

        self._folder = folder
        self._serviceName = serviceName
        self._layer = layer
        self._outFormat = outFormat
        self._spatialRel = spatialRel
        self.session = utils.retry_requests()

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        if value is not None:
            valids, _ = self.get_fs()
            if value not in valids:
                msg = (
                    f"The given folder, {value}, is not valid. "
                    + "Valid folders are "
                    + ", ".join(str(v) for v in valids)
                )
                raise ValueError(msg)
        self._folder = value

    @property
    def serviceName(self):
        return self._serviceName

    @serviceName.setter
    def serviceName(self, value):
        if value is not None:
            _, valids = self.get_fs(self.folder)
            if value not in valids:
                msg = (
                    f"The given serviceName, {value}, is not valid. "
                    + "Valid serviceNames are "
                    + ", ".join(str(v) for v in valids)
                )
                raise ValueError(msg)
        self._serviceName = value

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, value):
        if value is not None:
            v_dict, _ = self.get_layers()
            valids = list(v_dict.keys())
            if value not in valids or not isinstance(value, int):
                msg = (
                    f"The given layer, {value}, is not valid. "
                    + "Valid layers are integers "
                    + ", ".join(str(v) for v in valids)
                )
                raise ValueError(msg)
        self._layer = value

    @property
    def outFormat(self):
        return self._outFormat

    @outFormat.setter
    def outFormat(self, value):
        if self.base_url is not None:
            valids = self.queryFormats
            if value not in valids:
                msg = (
                    f"The given outFormat, {value}, is not valid. "
                    + "Valid outFormats are "
                    + ", ".join(str(v) for v in valids)
                )
                raise ValueError(msg)
        self._outFormat = value

    @property
    def spatialRel(self):
        return self._spatialRel

    @spatialRel.setter
    def spatialRel(self, value):
        spatialRels = [
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
        if value is not None and value.lower() not in [s.lower() for s in spatialRels]:
            msg = (
                f"The given spatialRel, {value}, is not valid. "
                + "Valid spatialRels are "
                + ", ".join(str(v) for v in spatialRels)
            )
            raise ValueError(msg)
        self._spatialRel = value

    def get_fs(self, folder=None):
        """Get folders and services of the geoserver's a folder/root url"""
        url = self.root if folder is None else f"{self.root}/{folder}"
        info = utils.get_url(self.session, url, {"f": "json"}).json()
        try:
            folders = info["folders"]
        except KeyError:
            warn(f"The url doesn't have any folders: {url}")
            folders = None

        try:
            services = {
                f"/{s}".split("/")[-1]: t
                for s, t in zip(
                    utils.traverse_json(info, ["services", "name"]),
                    utils.traverse_json(info, ["services", "type"]),
                )
            }
        except KeyError:
            warn(f"The url doesn'thave any services: {url}")
            services = None
        return folders, services

    def get_layers(self, url=None, serviceName=None):
        """Find the available sublayers and their parent layers

        Parameters
        ----------
        url : string, optional
            The url that contains the layers
        serviceName : string, optional
            The geoserver serviceName, defaults to the serviceName instance variable of the class

        Returns
        -------
        dict
            Two dictionaries sublayers and parent_layers
        """
        if url is None:
            _, services = self.get_fs(self.folder)
            serviceName = self.serviceName if serviceName is None else serviceName
            if serviceName is None:
                raise ValueError(
                    "serviceName should be either passed as an argument "
                    + "or be set as a class instance variable"
                )
            try:
                if self.folder is None:
                    url = f"{self.root}/{serviceName}/{services[serviceName]}"
                else:
                    url = f"{self.root}/{self.folder}/{serviceName}/{services[serviceName]}"
            except KeyError:
                raise KeyError(
                    "The serviceName was not found. Check if folder is set correctly."
                )

        info = utils.get_url(self.session, url, {"f": "json"}).json()
        try:
            layers = {
                i: n
                for i, n in zip(
                    utils.traverse_json(info, ["layers", "id"]),
                    utils.traverse_json(info, ["layers", "name"]),
                )
                if n is not None
            }
            layers.update({-1: "HEAD LAYER"})
        except TypeError:
            raise TypeError(f"The url doesn't have layers, {url}")

        if len(layers) < 2:
            raise ValueError(f"The url doesn't have layers, {url}")

        parent_layers = {
            i: p
            for i, p in zip(
                utils.traverse_json(info, ["layers", "id"]),
                utils.traverse_json(info, ["layers", "parentLayerId"]),
            )
        }

        def get_parents(lid):
            lid = [parent_layers[lid]]
            while lid[-1] > -1:
                lid.append(parent_layers[lid[-1]])
            return lid

        parent_layers = {
            i: [layers[layer] for layer in get_parents(i)[:-1]]
            for i in utils.traverse_json(info, ["layers", "id"])
        }
        sublayers = [
            i
            for i, s in zip(
                utils.traverse_json(info, ["layers", "id"]),
                utils.traverse_json(info, ["layers", "subLayerIds"]),
            )
            if s is None
        ]
        sublayers = {i: layers[i] for i in sublayers}
        return sublayers, parent_layers


class ArcGISREST(ArcGISServer):
    """For getting data by geometry from an Arc GIS REST sercive."""

    def __init__(
        self,
        base_url=None,
        host=None,
        site=None,
        folder=None,
        serviceName=None,
        layer=None,
        n_threads=4,
        outFormat="json",
        spatialRel="esriSpatialRelIntersects",
        verbose=False,
    ):
        super().__init__(host, site, folder, serviceName, layer, outFormat, spatialRel)
        self.verbose = verbose

        if self.outFormat not in ["json", "geojson"]:
            raise ValueError("Only json and geojson are supported for outFormat.")

        self._n_threads = min(n_threads, 8)

        if base_url is None:
            self.generate_url()
        else:
            self.base_url = base_url

    @property
    def base_url(self):
        return self._base_url

    @base_url.setter
    def base_url(self, value):
        self._base_url = value
        layer = value.split("/")[-1]
        if layer.isdigit():
            url = value.replace(f"/{layer}", "")
            sublayers, _ = self.get_layers(url=url)
            self.layer_name = sublayers[int(layer)]
        else:
            self.layer_name = value.split("/")[-2]
        self.test_url()

    @property
    def n_threads(self):
        return self._n_threads

    @n_threads.setter
    def n_threads(self, value):
        self._n_threads = min(value, 8)
        if value > 8:
            warn(f"No. of threads was reduced to 8 from {value}.")

    def generate_url(self):
        """Generate the base_url based on the class properties"""
        if self.serviceName is None:
            warn(
                "The base_url set to None since serviceName is not set:\n"
                + "URL's general form is https://<host>/<site>/rest/"
                + "services/<folder>/<serviceName>/<serviceType>/<layer>/\n"
                + "Use get_fs(<folder>) or get_layers(<serviceName> or "
                + "<url>) to get the available folders, services and layers."
            )
        else:
            _, services = self.get_fs(self.folder)
            if self.layer is None:
                layer_suffix = ""
                self.layer_name = self.serviceName
            else:
                sublayers, _ = self.get_layers()
                self.layer_name = sublayers[self.layer]
                layer_suffix = f"/{self.layer}"

            if self.folder is None:
                try:
                    self.base_url = f"{self.root}/{self.serviceName}/{services[self.serviceName]}{layer_suffix}"
                except KeyError:
                    raise KeyError(
                        f"The requetsed service is not available on the server:\n{self.base_url}"
                    )
            else:
                try:
                    self.base_url = f"{self.root}/{self.folder}/{self.serviceName}/{services[self.serviceName]}{layer_suffix}"
                except KeyError:
                    raise KeyError(
                        f"The requetsed service is not available on the server:\n{self.base_url}"
                    )

            self.test_url()

    def test_url(self):
        """Test the generated url and get the required parameters from the service"""
        try:
            r = utils.get_url(self.session, self.base_url, {"f": "json"}).json()
            try:
                self.units = r["units"].replace("esri", "").lower()
            except KeyError:
                self.units = None
            self.maxRecordCount = r["maxRecordCount"]
            self.queryFormats = (
                r["supportedQueryFormats"].replace(" ", "").lower().split(",")
            )
        except RetryError:
            try:
                r = utils.get_url(self.session, self.base_url)
                tree = html.fromstring(r.content)
                info = tree.xpath('//div[@class="rbody"]//text()')
                info = [i.strip() for i in info if i.strip() != ""]
                try:
                    units = info[info.index("Units:") + 1]
                    self.units = units.replace("esri", "").lower().strip()
                except ValueError:
                    self.units = None
                self.maxRecordCount = int(info[info.index("MaxRecordCount:") + 1])
                queryFormats = info[info.index("Supported Query Formats:") + 1]
                self.queryFormats = queryFormats.lower().replace(" ", "").split(",")
            except ValueError:
                raise KeyError(f"The service url is not valid:\n{self.base_url}")
        except KeyError:
            raise KeyError(f"The service url is not valid:\n{self.base_url}")

        if self.verbose:
            print(self.__repr__())

    def __repr__(self):
        """Print the services properties."""
        msg = (
            f"The following url was generated successfully:\n{self.base_url}\n"
            + f"Max Record Count: {self.maxRecordCount}\n"
            + f"Supported Query Formats: {self.queryFormats}"
        )
        if self.units is not None:
            msg += f"Units: {self.units}"
        return msg

    def get_featureids(self, geom):
        """Get feature IDs withing a geometry"""
        if self.base_url is None:
            raise ValueError(
                "The base_url is not set yet, use "
                + "self.generate_url(<layer>) to form the url"
            )

        if isinstance(geom, list) or isinstance(geom, tuple):
            if len(geom) != 4:
                raise TypeError(
                    "The bounding box should be a list or tuple of form [west, south, east, north]"
                )
            geometryType = "esriGeometryEnvelope"
            bbox = dict(zip(["xmin", "ymin", "xmax", "ymax"], geom))
            geom_json = {**bbox, "spatialRelference": {"wkid": 4326}}
            geometry = json.dumps(geom_json)
        elif isinstance(geom, Polygon):
            geometryType = "esriGeometryPolygon"
            geom_json = {
                "rings": [[[x, y] for x, y in zip(*geom.exterior.coords.xy)]],
                "spatialRelference": {"wkid": 4326},
            }
            geometry = json.dumps(geom_json)
        else:
            raise ValueError("The geometry should be either a bbox (list) or a Polygon")

        payload = {
            "geometryType": geometryType,
            "geometry": geometry,
            "inSR": "4326",
            "spatialRel": self.spatialRel,
            "returnGeometry": "false",
            "returnIdsOnly": "true",
            "f": self.outFormat,
        }
        r = utils.post_url(self.session, f"{self.base_url}/query", payload)
        try:
            oids = r.json()["objectIds"]
            oid_list = list(zip_longest(*[iter(oids)] * self.maxRecordCount))
            oid_list[-1] = [i for i in oid_list[-1] if i is not None]
        except (KeyError, TypeError, IndexError):
            warn(
                "No feature ID were found within the requested "
                + f"region using the spatial relationship {self.spatialRel}."
            )
            raise

        self.splitted_ids = oid_list

    def get_features(self):
        """Get features based on the feature IDs"""

        def get_geojson(ids):
            payload = {
                "objectIds": ",".join(str(i) for i in ids),
                "returnGeometry": "true",
                "outSR": "4326",
                "outFields": "*",
                "f": self.outFormat,
            }
            r = utils.post_url(self.session, f"{self.base_url}/query", payload)
            try:
                return utils.json_togeodf(r.json(), "epsg:4326")
            except TypeError:
                return ids
            except AssertionError:
                raise AssertionError(
                    "There was a problem processing GeoJSON, "
                    + "try setting outFormat to json"
                )

        def get_json(ids):
            payload = {
                "objectIds": ",".join(str(i) for i in ids),
                "returnGeometry": "true",
                "outSR": "4326",
                "outFields": "*",
                "f": self.outFormat,
            }
            r = utils.post_url(self.session, f"{self.base_url}/query", payload)
            try:
                return utils.json_togeodf(r.json(), "epsg:4326")
            except TypeError:
                return ids

        if self.outFormat == "json":
            getter = get_json
        else:
            getter = get_geojson

        feature_list = pqdm(
            self.splitted_ids,
            getter,
            n_jobs=self.n_threads,
            desc=f"{self.layer_name}",
            disable=not self.verbose,
        )

        # Find the failed batches and retry
        fails = [
            ids
            for ids in feature_list
            if isinstance(ids, tuple) or isinstance(ids, list)
        ]
        success = [ids for ids in feature_list if isinstance(ids, gpd.GeoDataFrame)]

        if len(fails) > 0:
            fails = ([x] for y in fails for x in y)
            retry = pqdm(
                fails,
                getter,
                n_jobs=min(self.n_threads * 2, 8),
                desc="Retry failed batches",
                disable=not self.verbose,
            )
            success += [ids for ids in retry if isinstance(ids, gpd.GeoDataFrame)]

        if len(success) == 0:
            raise ValueError("No valid feature were found.")

        data = gpd.GeoDataFrame(pd.concat(success))
        data.crs = "epsg:4326"
        return data


def wms_bygeom(
    url,
    service_name,
    geometry,
    width=None,
    resolution=None,
    layers=None,
    outFormat=None,
    nodata=None,
    fpath=None,
    version="1.3.0",
    in_crs="epsg:4326",
    crs="epsg:4326",
    fill_holes=False,
    verbose=False,
):
    """Data from a WMS service within a geometry

    Parameters
    ----------
    url : string
        The base url for the WMS service. Some examples:
        https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer
        https://www.mrlc.gov/geoserver/mrlc_download/wms
    service_name : string
        Name of the service to appear in the progress bar
    geometry : Polygon
        A shapely Polygon for getting the data
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
    outFormat : string
        The data format to request for data from the service, defaults to None which
         throws an error and includes all the avialable format offered by the service.
    fpath : dict, optional
        The path to save the downloaded images, defaults to None which will only return
        the data as ``xarray.Dataset`` and doesn't save the files. The argument should be
        a dict with keys as the variable name in the output dataframe and values as
        the path to save to the file.
    version : string, optional
        The WMS service version which should be either 1.1.1 or 1.3.0, defaults to 1.3.0.
    in_crs : string, optional
        The spatial reference system of the input geometry, defaults to
        epsg:4326.
    crs : string, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.
    fill_holes : bool, optional
        Wether to fill the holes in the geometry's interior, defaults to False.
    verbose : bool, optional
        Whether to show more information during runtime, defaults to False

    Returns
    -------
    xarray.Dataset
    """
    for i in range(3):
        try:
            wms = WebMapService(url, version=version)
            break
        except ConnectionError:
            continue

    valid_layers = {wms[layer].name: wms[layer].title for layer in list(wms.contents)}
    if layers is None:
        raise ValueError(
            "The layers argument is missing."
            + " The following layers are available:\n"
            + "\n".join(f"{name}: {title}" for name, title in valid_layers.items())
        )
    elif not isinstance(layers, dict):
        raise ValueError(
            "The layers argument should be of type dict: {var_name : layer_name}"
        )
    elif any(str(layer) not in valid_layers.keys() for layer in layers.values()):
        raise ValueError(
            "The given layers argument is invalid."
            + " Valid layers are:\n"
            + "\n".join(f"{name}: {title}" for name, title in valid_layers.items())
        )

    valid_outFormats = wms.getOperationByName("GetMap").formatOptions
    if outFormat is None:
        raise ValueError(
            "The outFormat argument is missing."
            + " The following output formats are available:\n"
            + ", ".join(fmt for fmt in valid_outFormats)
        )
    elif outFormat not in valid_outFormats:
        raise ValueError(
            "The outFormat argument is invalid."
            + " Valid output formats are:\n"
            + ", ".join(fmt for fmt in valid_outFormats)
        )

    valid_crss = {
        layer: [s.lower() for s in wms[layer].crsOptions] for layer in layers.values()
    }
    if any(crs not in valid_crss[layer] for layer in layers.values()):
        raise ValueError(
            "The crs argument is invalid."
            + "\n".join(
                [
                    f" Valid CRSs for {layer} layer are:\n" + ", ".join(c for c in crss)
                    for layer, crss in valid_crss.items()
                ]
            )
        )

    if not isinstance(geometry, Polygon):
        raise ValueError("Geometry should be of type shapley's Polygon")
    elif fill_holes:
        geometry = Polygon(geometry.exterior)

    if in_crs != crs:
        prj = pyproj.Transformer.from_crs(in_crs, crs, always_xy=True)
        geometry = ops.transform(prj.transform, geometry)

    west, south, east, north = geometry.bounds
    if width is None and resolution is not None:
        width = int((east - west) * 3600 / resolution)
    elif width is None and resolution is None:
        raise ValueError("Width or resolution should be provided.")
    elif width is not None and resolution is not None:
        raise ValueError("Either width or resolution should be provided, not both.")

    height = int(abs(north - south) / abs(east - west) * width)

    if fpath is None:
        fpath = {k: f"/tmp/{k}.tiff" for k in layers.keys()}
    elif fpath is not None and not isinstance(fpath, dict):
        raise ValueError("The fpath argument should be of type dict: {var_name : path}")

    [utils.check_dir(f) for f in fpath.values()]

    mask, transform = utils.geom_mask(geometry, width, height)
    name_list = list(layers.keys())
    layer_list = list(layers.values())

    def _wms(inp):
        name, layer = inp
        img = wms.getmap(
            layers=[layer],
            srs=crs,
            bbox=geometry.bounds,
            size=(width, height),
            format=outFormat,
        )
        return (name, img.read())

    resp = pqdm(
        zip(name_list, layer_list),
        _wms,
        n_jobs=4,
        desc=f"{service_name}",
        disable=not verbose,
    )

    data = utils.create_dataset(
        resp[0][1], mask, transform, width, height, resp[0][0], fpath[resp[0][0]]
    )

    if len(resp) > 1:
        for name, r in resp:
            ds = utils.create_dataset(
                r, mask, transform, width, height, name, fpath[name]
            )
            data = xr.merge([data, ds])
    return data


class WFS:
    """Data from any WMS service within a geometry or by featureid"""

    def __init__(
        self, url, layer=None, outFormat=None, version="2.0.0", crs="epsg:4326"
    ):
        """Initialize WFS

        Parameters
        ----------
        url : string
            The base url for the WMS service. Some examples:
            https://hazards.fema.gov/nfhl/services/public/NFHL/MapServer/WFSServer
        layer : string
            The layer from the service to be downloaded, defaults to None which throws
            an error and includes all the avialable layers offered by the service.
        outFormat : string
            The data format to request for data from the service, defaults to None which
             throws an error and includes all the avialable format offered by the service.
        version : string, optional
            The WMS service version which should be either 1.1.1 or 1.3.0, defaults to 1.3.0.
        crs: string, optional
            The spatial reference system to be used for requesting the data, defaults to
            epsg:4326.

        Returns
        -------
        requests.Response
        """
        self.url = url
        self.layer = layer
        self.outFormat = outFormat
        self.version = version
        self.crs = crs

        for i in range(3):
            try:
                wfs = WebFeatureService(url, version=version)
                break
            except ConnectionError:
                continue

        valid_layers = list(wfs.contents)
        valid_layers_lower = [layer.lower() for layer in valid_layers]
        if layer is None:
            raise ValueError(
                "The layer argument is missing."
                + " The following layers are available:\n"
                + ", ".join(layer for layer in valid_layers)
            )
        elif layer.lower() not in valid_layers_lower:
            raise ValueError(
                "The given layers argument is invalid."
                + " Valid layers are:\n"
                + ", ".join(layer for layer in valid_layers)
            )

        valid_outFormats = wfs.getOperationByName("GetFeature").parameters[
            "outputFormat"
        ]["values"]
        valid_outFormats = [f.lower() for f in valid_outFormats]
        if outFormat is None:
            raise ValueError(
                "The outFormat argument is missing."
                + " The following output formats are available:\n"
                + ", ".join(fmt for fmt in valid_outFormats)
            )
        elif outFormat.lower() not in valid_outFormats:
            raise ValueError(
                "The outFormat argument is invalid."
                + " Valid output formats are:\n"
                + ", ".join(fmt for fmt in valid_outFormats)
            )

        valid_crss = [f"{s.authority.lower()}:{s.code}" for s in wfs[layer].crsOptions]
        if crs.lower() not in valid_crss:
            raise ValueError(
                "The crs argument is invalid."
                + "\n".join(
                    [
                        f" Valid CRSs for {layer} layer are:\n"
                        + ", ".join(c for c in valid_crss)
                    ]
                )
            )

    def __repr__(self):
        """Print the services properties."""
        return (
            "Connected to the WFS service with the following properties:\n"
            + f"URL: {self.url}\n"
            + f"Version: {self.version}\n"
            + f"Layer: {self.layer}\n"
            + f"Output Format: {self.outFormat}\n"
            + f"Output CRS: {self.crs}"
        )

    def getfeature_bybox(self, bbox, in_crs="epsg:4326"):
        """Data from any WMS service within a geometry

        Parameters
        ----------
        bbox : list or tuple
            A bounding box for getting the data: [west, south, east, north]
        in_crs : string, optional
            The spatial reference system of the input region, defaults to
            epsg:4326.

        Returns
        -------
        xarray.Dataset
        """
        if not isinstance(bbox, list) and not isinstance(bbox, tuple):
            raise ValueError("The bbox should be of type list or tuple.")
        if len(bbox) != 4:
            raise ValueError("The bbox length should be 4")
        if in_crs != self.crs:
            prj = pyproj.Transformer.from_crs(in_crs, self.crs, always_xy=True)
            bbox = ops.transform(prj.transform, box(*bbox))
            bbox = bbox.bounds
        bbox = ",".join(str(c) for c in bbox) + f",{self.crs}"

        payload = {
            "service": "wfs",
            "version": self.version,
            "outputFormat": self.outFormat,
            "request": "GetFeature",
            "typeName": self.layer,
            "bbox": bbox,
        }

        r = utils.get_url(utils.retry_requests(), self.url, payload)

        if r.headers["Content-Type"] == "application/xml":
            root = ET.fromstring(r.text)
            raise ValueError(root[0][0].text.strip())
        return r

    def getfeature_byid(self, featurename, featureids):
        """Get features based on feature IDs

        Parameters
        ----------
        featurename : string
            The name of the column for searching for feature IDs
        featureids : int, string, or list
            The feature ID(s)

        Returns
        -------
        requests.Response
        """

        if not isinstance(featureids, list):
            featureids = [featureids]

        if len(featureids) == 0:
            raise ValueError("The feature ID list is empty!")

        featureids = [str(i) for i in featureids]

        def filter_xml(pname, pid):
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

        payload = {
            "service": "wfs",
            "version": self.version,
            "outputFormat": self.outFormat,
            "request": "GetFeature",
            "typeName": self.layer,
            "srsName": self.crs,
            "filter": filter_xml(featurename, featureids),
        }

        r = utils.post_url(utils.retry_requests(), self.url, payload)
        if r.headers["Content-Type"] == "application/xml":
            root = ET.fromstring(r.text)
            raise ValueError(root[0][0].text.strip())

        return r
