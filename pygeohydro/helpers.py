"""Some helper function for PyGeoHydro."""
import logging
import sys
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import async_retriever as ar
import numpy as np
import pandas as pd
from defusedxml import ElementTree
from pygeoogc import ServiceURL

from .exceptions import InvalidInputRange, InvalidInputType

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

    def _get_xml(
        layer: str,
    ) -> Tuple[Any, Any, Any]:
        root = ElementTree.fromstring(ar.retrieve_text([f"{base_url}/{layer}.xml"], ssl=False)[0])
        return root, root.findall(f"{base_path}/edomv"), root.findall(f"{base_path}/edomvd")

    root, edomv, edomvd = _get_xml("NLCD_2019_Land_Cover_Science_Product_L48_20210604")
    cover_classes = {}
    for t, v in zip(edomv, edomvd):
        cover_classes[t.text] = v.text

    clist = [i.split() for i in root.find("eainfo/overview/eadetcit").text.split("\n")[2:]]
    colors = {
        int(c): (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1) for c, r, g, b in clist
    }
    colors[0] = (*colors[0][:3], 0)

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
) -> List[Tuple[pd.Timestamp, str]]:
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
        date_range = d_list.pop(0)
        while d_list:
            date_range = date_range.union(d_list.pop(0))

    base_url = ServiceURL().http.ssebopeta

    f_list = [
        (d, f"{base_url}/det{d.strftime('%Y%j')}.modisSSEBopETactual.zip") for d in date_range
    ]

    return f_list


class Stats(NamedTuple):
    """Statistics for NLCD."""

    classes: Dict[str, float]
    categories: Dict[str, float]
