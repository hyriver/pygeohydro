"""Some helper function for PyGeoHydro."""
from typing import Any, Dict

import async_retriever as ar
import defusedxml.ElementTree as etree
import numpy as np
import pandas as pd


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

    root, edomv, edomvd = _get_xml("nlcd_2019_land_cover_l48_20210604")
    cover_classes = {}
    for t, v in zip(edomv, edomvd):
        cover_classes[t.text] = v.text

    clist = [i.split() for i in root.find("eainfo/overview/eadetcit").text.split("\n")[2:]]
    colors = {int(c): (float(r), float(g), float(b)) for c, r, g, b in clist}

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
