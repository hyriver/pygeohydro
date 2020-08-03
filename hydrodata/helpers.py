"""Some helper function for Hydrodata."""
from typing import Any, Dict

import numpy as np
import pandas as pd
from defusedxml import cElementTree as ET
from pygeoogc import RetrySession


def nlcd_helper() -> Dict[str, Any]:
    """Get legends and properties of the NLCD cover dataset.

    Notes
    -----
    The following references have been used:
        - https://github.com/jzmiller1/nlcd
        - https://www.mrlc.gov/data-services-page
        - https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend
    """
    url = (
        "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/metadata/"
        + "NLCD_2016_Land_Cover_Science_product_L48.xml"
    )
    r = RetrySession().get(url)

    root = ET.fromstring(r.content)

    clist = root[4][1][1].text.split("\n")[2:]
    _colors = [i.split() for i in clist]
    colors = {int(c): (float(r), float(g), float(b)) for c, r, g, b in _colors}

    classes = {
        root[4][0][3][i][0][0].text: root[4][0][3][i][0][1].text.split("-")[0].strip()
        for i in range(3, len(root[4][0][3]))
    }

    nlcd_meta = {
        "impervious_years": [2016, 2011, 2006, 2001],
        "canopy_years": [2016, 2011],
        "cover_years": [2016, 2013, 2011, 2008, 2006, 2004, 2001],
        "classes": classes,
        "categories": {
            "Unclassified": ("0"),
            "Water": ("11", "12"),
            "Developed": ("21", "22", "23", "24"),
            "Barren": ("31",),
            "Forest": ("41", "42", "43", "45", "46"),
            "Shrubland": ("51", "52"),
            "Herbaceous": ("71", "72", "73", "74"),
            "Planted/Cultivated": ("81", "82"),
            "Wetlands": ("90", "95"),
        },
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
