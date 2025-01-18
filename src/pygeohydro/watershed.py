"""Accessing watershed boundary-level data through web services."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import pandas as pd
import xarray as xr

import async_retriever as ar
import pygeoogc as ogc
from pygeohydro.exceptions import InputValueError
from pygeoogc import ServiceURL
from pynhd import AGRBase
from pynhd.core import ScienceBase

if TYPE_CHECKING:
    from pyproj import CRS

    CRSType = int | str | CRS

__all__ = [
    "WBD",
    "huc_wb_full",
    "irrigation_withdrawals",
]


class WBD(AGRBase):
    """Access Watershed Boundary Dataset (WBD).

    Notes
    -----
    This web service offers Hydrologic Unit (HU) polygon boundaries for
    the United States, Puerto Rico, and the U.S. Virgin Islands.
    For more info visit: https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer

    Parameters
    ----------
    layer : str, optional
        A valid service layer. Valid layers are:

        - ``wbdline``
        - ``huc2``
        - ``huc4``
        - ``huc6``
        - ``huc8``
        - ``huc10``
        - ``huc12``
        - ``huc14``
        - ``huc16``

    outfields : str or list, optional
        Target field name(s), default to "*" i.e., all the fields.
    crs : str, int, or pyproj.CRS, optional
        Target spatial reference, default to ``EPSG:4326``.
    """

    def __init__(self, layer: str, outfields: str | list[str] = "*", crs: CRSType = 4326):
        self.valid_layers = {
            "wbdline": "wbdline",
            "huc2": "2-digit hu (region)",
            "huc4": "4-digit hu (subregion)",
            "huc6": "6-digit hu (basin)",
            "huc8": "8-digit hu  (subbasin)",
            "huc10": "10-digit hu (watershed)",
            "huc12": "12-digit hu (subwatershed)",
            "huc14": "14-digit hu",
            "huc16": "16-digit hu",
        }
        _layer = self.valid_layers.get(layer)
        if _layer is None:
            raise InputValueError("layer", list(self.valid_layers))
        super().__init__(ServiceURL().restful.wbd, _layer, outfields, crs)


def huc_wb_full(huc_lvl: int) -> gpd.GeoDataFrame:
    """Get the full watershed boundary for a given HUC level.

    Notes
    -----
    This function is designed for cases where the full watershed boundary is needed
    for a given HUC level. If only a subset of the HUCs is needed, then use
    the ``pygeohydro.WBD`` class. The full dataset is downloaded from the National Maps'
    `WBD staged products <https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/WBD/HU2/Shape/>`__.

    Parameters
    ----------
    huc_lvl : int
        HUC level, must be even numbers between 2 and 16.

    Returns
    -------
    geopandas.GeoDataFrame
        The full watershed boundary for the given HUC level.
    """
    valid_hucs = [2, 4, 6, 8, 10, 12, 14, 16]
    if huc_lvl not in valid_hucs:
        raise InputValueError("huc_lvl", list(map(str, valid_hucs)))

    base_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape"

    urls = [f"{base_url}/WBD_{h2:02}_HU2_Shape.zip" for h2 in range(1, 23)]
    fnames = [Path("cache", Path(url).name) for url in urls]
    fnames = ogc.streaming_download(urls, fnames=fnames)
    fnames = [f for f in fnames if f is not None]
    keys = (p.stem.split("_")[1] for p in fnames)
    engine = "pyogrio" if importlib.util.find_spec("pyogrio") else "fiona"
    huc = (
        gpd.GeoDataFrame(
            pd.concat(
                (gpd.read_file(f"{p}!Shape/WBDHU{huc_lvl}.shp", engine=engine) for p in fnames),
                keys=keys,
            )
        )
        .reset_index()
        .rename(columns={"level_0": "huc2"})
        .drop(columns="level_1")
    )
    huc = cast("gpd.GeoDataFrame", huc)
    return huc


def irrigation_withdrawals() -> xr.Dataset:
    """Get monthly water use for irrigation at HUC12-level for CONUS.

    Notes
    -----
    Dataset is retrieved from https://doi.org/10.5066/P9FDLY8P.
    """
    item = ScienceBase.get_file_urls("5ff7acf4d34ea5387df03d73")
    urls = item.loc[item.index.str.contains(".csv"), "url"]
    resp = ar.retrieve_text(urls.tolist())
    irr = {}
    for name, r in zip(urls.index, resp):
        df = pd.read_csv(
            io.StringIO(r),
            usecols=lambda s: "m3" in s or "huc12t" in s,  # type: ignore
        )
        df["huc12t"] = df["huc12t"].str.strip("'")
        df = df.rename(columns={"huc12t": "huc12"}).set_index("huc12")
        df = df.rename(columns={c: str(c)[:3].capitalize() for c in df})
        irr[name[-6:-4]] = df.copy()
    ds = xr.Dataset(irr).rename({"dim_1": "month"})
    long_names = {
        "GW": "groundwater_withdrawal",
        "SW": "surface_water_withdrawal",
        "TW": "total_withdrawal",
        "CU": "consumptive_use",
    }
    for v, n in long_names.items():
        ds[v].attrs["long_name"] = n
        ds[v].attrs["units"] = "m3"
    ds.attrs["description"] = " ".join(
        (
            "Estimated Monthly Water Use for Irrigation by",
            "12-Digit Hydrologic Unit in the Conterminous United States for 2015",
        )
    )
    ds.attrs["source"] = "https://doi.org/10.5066/P9FDLY8P"
    return ds
