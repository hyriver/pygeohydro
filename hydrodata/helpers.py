#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some helper function for Hydrodata"""

import xml.etree.cElementTree as ET

import pandas as pd
from hydrodata import utils


def nlcd_helper():
    """Helper for NLCD cover data

    Notes
    -----
    The following references have been used:
        * https://github.com/jzmiller1/nlcd
        * https://www.mrlc.gov/data-services-page
        * https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend
    """
    url = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/metadata/NLCD_2016_Land_Cover_Science_product_L48.xml"
    r = utils.get_url(utils.retry_requests(), url)

    root = ET.fromstring(r.content)

    colors = root[4][1][1].text.split("\n")[2:]
    colors = [i.split() for i in colors]
    colors = dict((int(c), (float(r), float(g), float(b))) for c, r, g, b in colors)

    classes = dict(
        (root[4][0][3][i][0][0].text, root[4][0][3][i][0][1].text.split("-")[0].strip())
        for i in range(3, len(root[4][0][3]))
    )

    nlcd_meta = dict(
        impervious_years=[2016, 2011, 2006, 2001],
        canopy_years=[2016, 2011],
        cover_years=[2016, 2013, 2011, 2008, 2006, 2004, 2001],
        classes=classes,
        categories={
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
        roughness={
            "11": 0.0250,
            "12": 0.0220,
            "21": 0.0400,
            "22": 0.1000,
            "23": 0.0800,
            "24": 0.1500,
            "31": 0.0275,
            "41": 0.1600,
            "42": 0.1800,
            "43": 0.1700,
            "45": 0.1000,
            "46": 0.0350,
            "51": 0.1600,
            "52": 0.1000,
            "71": 0.0350,
            "72": 0.0350,
            "81": 0.0325,
            "82": 0.0375,
            "90": 0.1200,
            "95": 0.0700,
        },
        colors=colors,
    )

    return nlcd_meta


def nhdplus_fcodes():
    """Get NHDPlus FCode lookup table"""
    url = (
        "https://nhd.usgs.gov/userGuide/Robohelpfiles/NHD_User_Guide"
        + "/Feature_Catalog/Hydrography_Dataset/Complete_FCode_List.htm"
    )
    return pd.concat(pd.read_html(url, header=0)).set_index("FCode")


def nwis_errors():
    """Get USGS daily values site web service's error code lookup table"""
    return pd.read_html("https://waterservices.usgs.gov/rest/DV-Service.html")[0]
