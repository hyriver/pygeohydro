"""Some helper function for PyGeoHydro."""
import logging
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Tuple, Union

import async_retriever as ar
import defusedxml.ElementTree as etree
import numpy as np
import pandas as pd
import pygeoutils as geoutils
import xarray as xr
from pygeoogc import WMS, ServiceURL
from shapely.geometry import MultiPolygon, Polygon

from .exceptions import InvalidInputRange, InvalidInputType, InvalidInputValue

__all__ = ["nlcd_helper", "nwis_errors"]

DEF_CRS = "epsg:4326"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False


def nlcd_helper() -> Dict[str, Any]:
    """Get legends and properties of the NLCD cover dataset.

    Notes
    -----
    The following references have been used:
        - https://github.com/jzmiller1/nlcd
        - https://www.mrlc.gov/data-services-page
        - https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend

    Returns
    -------
    dict
        Years where data is available and cover classes and categories, and roughness estimations.
    """
    base_url = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/metadata"
    base_path = "eainfo/detailed/attr/attrdomv/edom"

    def _get_xml(layer):
        root = etree.fromstring(ar.retrieve([f"{base_url}/{layer}.xml"], "text")[0])
        return root, root.findall(f"{base_path}/edomv"), root.findall(f"{base_path}/edomvd")

    root, edomv, edomvd = _get_xml("NLCD_2019_Land_Cover_Science_Product_L48_20210604")
    cover_classes = {}
    for t, v in zip(edomv, edomvd):
        cover_classes[t.text] = v.text

    clist = [i.split() for i in root.find("eainfo/overview/eadetcit").text.split("\n")[2:]]
    colors = {
        int(c): (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0) for c, r, g, b in clist
    }

    _, edomv, edomvd = _get_xml("nlcd_2019_impervious_descriptor_l48_20210604")
    descriptors = {}
    for t, v in zip(edomv, edomvd):
        tag = t.text.split(" - ")
        descriptors[tag[0]] = v.text if tag[-1].isnumeric() else f"{tag[-1]}: {v.text}"

    cyear = [2019, 2016, 2013, 2011, 2008, 2006, 2004, 2001]
    nlcd_meta = {
        "cover_years": cyear,
        "impervious_years": cyear,
        "descriptor_years": cyear,
        "canopy_years": [2016, 2011],
        "classes": cover_classes,
        "categories": {
            "Background": ("127",),
            "Unclassified": ("0",),
            "Water": ("11", "12"),
            "Developed": ("21", "22", "23", "24"),
            "Barren": ("31",),
            "Forest": ("41", "42", "43", "45", "46"),
            "Shrubland": ("51", "52"),
            "Herbaceous": ("71", "72", "73", "74"),
            "Planted/Cultivated": ("81", "82"),
            "Wetlands": ("90", "95"),
        },
        "descriptors": descriptors,
        "roughness": {
            "11": 0.001,
            "12": 0.022,
            "21": 0.0404,
            "22": 0.0678,
            "23": 0.0678,
            "24": 0.0404,
            "31": 0.0113,
            "41": 0.36,
            "42": 0.32,
            "43": 0.4,
            "45": 0.4,
            "46": 0.24,
            "51": 0.24,
            "52": 0.4,
            "71": 0.368,
            "72": np.nan,
            "81": 0.325,
            "82": 0.16,
            "90": 0.086,
            "95": 0.1825,
        },
        "colors": colors,
    }

    return nlcd_meta


def nwis_errors() -> pd.DataFrame:
    """Get error code lookup table for USGS sites that have daily values."""
    return pd.read_html("https://waterservices.usgs.gov/rest/DV-Service.html")[0]


def get_ssebopeta_urls(
    dates: Union[Tuple[str, str], Union[int, List[int]]]
) -> List[Tuple[pd.DatetimeIndex, str]]:
    """Get list of URLs for SSEBop dataset within a period or years."""
    if not isinstance(dates, (tuple, list, int)):
        raise InvalidInputType(
            "dates", "tuple, list, or int", "(start, end), year, or [years, ...]"
        )

    if isinstance(dates, tuple):
        if len(dates) != 2:
            raise InvalidInputType("dates", "(start, end)")
        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])
        if start < pd.to_datetime("2000-01-01") or end > pd.to_datetime("2020-12-31"):
            raise InvalidInputRange("SSEBop", ("2000", "2020"))
        date_range = pd.date_range(start, end)
    else:
        years = dates if isinstance(dates, list) else [dates]
        seebop_yrs = np.arange(2000, 2021)

        if any(y not in seebop_yrs for y in years):
            raise InvalidInputRange("SSEBop", ("2000", "2020"))

        d_list = [pd.date_range(f"{y}0101", f"{y}1231") for y in years]
        date_range = d_list[0] if len(d_list) == 1 else d_list[0].union_many(d_list[1:])

    base_url = ServiceURL().http.ssebopeta
    f_list = [
        (d, f"{base_url}/det{d.strftime('%Y%j')}.modisSSEBopETactual.zip") for d in date_range
    ]

    return f_list


def get_nlcd_layers(
    layers: Union[str, List[str]],
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    resolution: float,
    geo_crs: str = DEF_CRS,
    crs: str = DEF_CRS,
) -> xr.Dataset:
    """Get data from NLCD database (2016).

    Download land use/land cover data from NLCD (2016) database within
    a given geometry in epsg:4326.

    Parameters
    ----------
    layers : str, list
        The NLCD layers to be extracted.
    geometry : Polygon, MultiPolygon, or tuple of length 4
        The geometry or bounding box (west, south, east, north) for extracting the data.
    resolution : float
        The data resolution in meters. The width and height of the output are computed in pixel
        based on the geometry bounds and the given resolution.
    geo_crs : str, optional
        The CRS of the input geometry, defaults to epsg:4326.
    crs : str, optional
        The spatial reference system to be used for requesting the data, defaults to
        epsg:4326.

    Returns
    -------
     xarray.DataArray
         NLCD within a geometry
    """
    if resolution < 30:
        logger.warning("NLCD resolution is 30 m, so finer resolutions are not recommended.")

    _geometry = geoutils.geo2polygon(geometry, geo_crs, crs)
    wms = WMS(ServiceURL().wms.mrlc, layers=layers, outformat="image/geotiff", crs=crs)
    r_dict = wms.getmap_bybox(
        _geometry.bounds, resolution, box_crs=crs, kwargs={"styles": "raster"}
    )

    ds = geoutils.gtiff2xarray(r_dict, _geometry, crs)
    attrs = ds.attrs
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    units = OrderedDict(
        (("impervious", "%"), ("cover", "classes"), ("canopy", "%"), ("descriptor", "classes"))
    )
    for lyr in layers:
        name = [n for n in units if n in lyr.lower()][-1]
        lyr_name = f"{name}_{lyr.split('_')[1]}"
        ds = ds.rename({lyr: lyr_name})
        ds[lyr_name].attrs["units"] = units[name]
    ds.attrs = attrs
    return ds


def nlcd_layers(years: Mapping[str, List[int]], region: str) -> List[str]:
    """Get NLCD layers for the provided years dictionary."""
    valid_regions = ["L48", "HI", "PR", "AK"]
    region = region.upper()
    if region not in valid_regions:
        raise InvalidInputValue("region", valid_regions)

    nlcd_meta = nlcd_helper()

    names = ["impervious", "cover", "canopy", "descriptor"]
    avail_years = {n: nlcd_meta[f"{n}_years"] for n in names}

    if any(
        yr not in avail_years[lyr] or lyr not in names for lyr, yrs in years.items() for yr in yrs
    ):
        vals = [f"\n{lyr}: {', '.join(str(y) for y in yr)}" for lyr, yr in avail_years.items()]
        raise InvalidInputValue("years", vals)

    layer_map = {
        "canopy": lambda _: "Tree_Canopy",
        "cover": lambda _: "Land_Cover_Science_Product",
        "impervious": lambda _: "Impervious",
        "descriptor": lambda x: "Impervious_Descriptor" if x == "AK" else "Impervious_descriptor",
    }

    return [
        f"NLCD_{yr}_{layer_map[lyr](region)}_{region}" for lyr, yrs in years.items() for yr in yrs
    ]
