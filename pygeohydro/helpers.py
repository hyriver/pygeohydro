"""Some helper function for PyGeoHydro."""
from __future__ import annotations

import io
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, Tuple, Union

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import ujson as json
from defusedxml import ElementTree

import async_retriever as ar
from pygeohydro import us_abbrs
from pygeohydro.exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
)
from pygeoogc import ServiceURL

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

    GTYPE = Union[Polygon, MultiPolygon, Tuple[float, float, float, float]]
    CRSTYPE = Union[int, str, pyproj.CRS]
__all__ = ["nlcd_helper", "nwis_errors", "states_lookup_table", "get_us_states"]


def nlcd_helper() -> dict[str, Any]:
    """Get legends and properties of the NLCD cover dataset.

    Notes
    -----
    The following references have been used:
        - https://github.com/jzmiller1/nlcd
        - https://www.mrlc.gov/data-services-page
        - https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend
        - https://doi.org/10.1111/jfr3.12347

    Returns
    -------
    dict
        Years where data is available and cover classes and categories, and roughness estimations.
    """
    base_url = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/metadata"
    base_path = "eainfo/detailed/attr/attrdomv/edom"

    def _get_xml(
        layer: str,
    ) -> tuple[Any, Any, Any]:
        root = ElementTree.fromstring(ar.retrieve_text([f"{base_url}/{layer}.xml"], ssl=False)[0])
        return (
            root,
            root.findall(f"{base_path}/edomv"),
            root.findall(f"{base_path}/edomvd"),
        )

    root, edomv, edomvd = _get_xml("NLCD_2019_Land_Cover_Science_Product_L48_20210604")
    cover_classes = {}
    for t, v in zip(edomv, edomvd):
        cover_classes[t.text] = v.text

    clist = [i.split() for i in root.find("eainfo/overview/eadetcit").text.split("\n")[2:]]
    colors = {
        int(c): (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0) for c, r, g, b in clist
    }
    colors[0] = (*colors[0][:3], 0.0)

    _, edomv, edomvd = _get_xml("nlcd_2019_impervious_descriptor_l48_20210604")
    descriptors = {}
    for t, v in zip(edomv, edomvd):
        tag = t.text.split(" - ")
        descriptors[tag[0]] = v.text if tag[-1].isnumeric() else f"{tag[-1]}: {v.text}"

    cyear = [2021, 2019, 2016, 2013, 2011, 2008, 2006, 2004, 2001]
    nlcd_meta = {
        "cover_years": cyear,
        "impervious_years": cyear,
        "descriptor_years": cyear,
        "canopy_years": list(range(2011, 2022)),
        "classes": cover_classes,
        "categories": {
            "Background": ("127",),
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


def get_ssebopeta_urls(dates: tuple[str, str] | int | list[int]) -> list[tuple[pd.Timestamp, str]]:
    """Get list of URLs for SSEBop dataset within a period or years."""
    if not isinstance(dates, (tuple, list, int)):
        raise InputTypeError("dates", "tuple, list, or int", "(start, end), year, or [years, ...]")

    year = datetime.now().year - 1
    if isinstance(dates, tuple):
        if len(dates) != 2:
            raise InputTypeError("dates", "(start, end)")
        start = pd.to_datetime(dates[0])
        end = pd.to_datetime(dates[1])
        if start < pd.to_datetime("2000-01-01") or end > pd.to_datetime(f"{year}-12-31"):
            raise InputRangeError("SSEBop", ("2000", str(year)))
        date_range = pd.date_range(start, end)
    else:
        years = dates if isinstance(dates, list) else [dates]
        seebop_yrs = np.arange(2000, year)

        if any(y not in seebop_yrs for y in years):
            raise InputRangeError("SSEBop", ("2000", str(year)))

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

    classes: dict[str, float]
    categories: dict[str, float]


def _get_state_codes(subset_key: str | list[str]) -> list[str]:
    """Get state codes for a subset of the US."""
    keys = [subset_key] if isinstance(subset_key, str) else subset_key
    state_cd = []

    state_keys = [k.upper() for k in keys if len(k) == 2]
    states = us_abbrs.STATES
    if any(k not in states for k in state_keys):
        raise InputValueError("subset_key", states)
    if state_keys:
        state_cd += state_keys

    other_keys = [k for k in keys if len(k) > 2]
    if "conus" in other_keys:
        other_keys.remove("conus")
        other_keys.append("contiguous")
    valid_keys = ["contiguous", "continental", "territories", "commonwealths"]
    if any(k not in valid_keys for k in other_keys):
        raise InputValueError("subset_key", valid_keys + ["conus"])
    if other_keys:
        state_cd += tlz.concat(getattr(us_abbrs, k.upper()) for k in other_keys)
    return state_cd


def get_us_states(subset_key: str | list[str] | None = None) -> gpd.GeoDataFrame:
    """Get US states as a GeoDataFrame from Census' TIGERLine 2022 database.

    Parameters
    ----------
    subset_key : str or list of str, optional
        Key to subset the geometries instead of returning all states, by default
        all states are returned. Valid keys are:

        - ``contiguous`` or ``conus``
        - ``continental``
        - ``commonwealths``
        - ``territories``
        - Two letter state codes, e.g., ``["TX", "CA", "FL", ...]``

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of requested US states.
    """
    url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
    us_states = gpd.read_file(io.BytesIO(ar.retrieve_binary([url])[0]))
    if subset_key is not None:
        state_cd = _get_state_codes(subset_key)
        return us_states[us_states.STUSPS.isin(state_cd)].copy()
    return us_states


class StateCounties(NamedTuple):
    """State and county codes and names."""

    name: str
    code: str | None
    counties: pd.Series


def states_lookup_table() -> dict[str, StateCounties]:
    """Get codes and names of US states and their counties.

    Notes
    -----
    This function is based on a file prepared by developers of
    an R package called `dataRetrieval <https://github.com/USGS-R/dataRetrieval>`__.

    Returns
    -------
    pandas.DataFrame
        State codes and name as a dataframe.
    """
    urls = [
        "https://www2.census.gov/geo/docs/reference/state.txt",
        "/".join(
            [
                "https://raw.githubusercontent.com/USGS-R/dataRetrieval",
                "main/inst/extdata/state_county.json",
            ]
        ),
    ]
    resp = ar.retrieve_text(urls)

    codes = pd.read_csv(io.StringIO(resp[0]), sep="|")
    codes["STATE"] = codes["STATE"].astype(str).str.zfill(2)
    codes = codes.set_index("STATE")

    def _county2series(cd: dict[str, dict[str, str]]) -> pd.Series:
        return pd.DataFrame.from_dict(cd, orient="index")["name"]

    def _state_cd(state: str) -> str | None:
        try:
            return codes.loc[state, "STUSAB"]  # type: ignore[no-any-return]
        except KeyError:
            return None

    states = {
        c: StateCounties(s["name"], _state_cd(c), _county2series(s["county_cd"]))
        for c, s in json.loads(resp[1])["US"]["state_cd"].items()
    }
    return states
