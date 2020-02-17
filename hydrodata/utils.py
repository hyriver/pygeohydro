#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some utilities for Hydrodata"""

from tqdm import tqdm
import urllib
from pathlib import Path


class DownloadProgressBar(tqdm):
    """A tqdm-based class for download progress."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Inspired from a tqdm example.

        Parameters
        ----------
        b : int, optional
            Number of blocks transferred so far [default: 1].
        bsize : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, out_dir):
    """Progress bar for downloading a file."""
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(
            url, filename=Path(out_dir, url.split("/")[-1]), reporthook=t.update_to
        )


def download_extract(url, out_dir):
    """Download and extract a `.7z` file."""
    import py7zr

    file = Path(out_dir).joinpath(url.split("/")[-1])
    if file.exists():
        py7zr.unpack_7zarchive(str(file), str(out_dir))
        print(f"Successfully extracted {file}.")
    else:
        download_url(url, out_dir)
        py7zr.unpack_7zarchive(str(file), str(out_dir))
        print(f"Successfully downloaded and extracted {str(file)}.")


def get_nhd(gis_dir):
    """Download and extract NHDPlus V2.1 database."""
    gis_dir = Path(gis_dir)

    if not gis_dir.is_dir():
        try:
            import os

            os.mkdir(gis_dir)
        except OSError:
            print(f"{gis_dir} directory cannot be created")

    print(f"Downloading USGS gage information data to {str(gis_dir)} >>>")
    base = "https://s3.amazonaws.com/edap-nhdplus/NHDPlusV21/" + "Data/NationalData/"
    dbname = [
        "NHDPlusV21_NationalData_GageInfo_05.7z",
        "NHDPlusV21_NationalData_GageLoc_05.7z",
    ]

    for db in dbname:
        utils.download_extract(base + db, gis_dir)
    return gis_dir.joinpath("NHDPlusNationalData")


def retry_requests(retries=3,
                   backoff_factor=0.5,
                   status_to_retry=(500, 502, 504),
                   prefixes=("http://", "https://")):
    """Configures the passed-in session to retry on failed requests.
    
    The fails can be due to connection errors, specific HTTP response
    codes and 30X redirections. The original code is taken from:
    https://github.com/bustawin/retry-requests
    
    Paramters
    ---------
    retries: int
        The number of maximum retries before raising an exception.
    backoff_factor: float
        A factor used to compute the waiting time between retries.
    status_to_retry: tuple of ints
        A tuple of status codes that trigger the reply behaviour.

    Returns
    -------
        A session object with the retry setup.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3 import Retry

    session = requests.Session()

    r = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_to_retry,
        method_whitelist=False
    )
    adapter = HTTPAdapter(max_retries=r)
    for prefix in prefixes:
        session.mount(prefix, adapter)

    return session


def get_data(stations):
    from hydrodata import Dataloader
    
    default = dict(start=None,
                   end=None,
                   station_id=None,
                   coords=None,
                   data_dir='./data',
                   rain_snow=False,
                   phenology=False,
                   width=2000,
                   climate=False,
                   nlcd=False,
                   yreas={'impervious': 2016, 'cover': 2016, 'canopy': 2016})

    params = list(stations.keys())

    if "station_id" in params and "coords" in params:
        if stations["station_id"] is not None and stations["coords"] is not None:
            raise KeyError('Either coords or station_id should be provided.')

    for k in list(default.keys()):
        if k not in params:
            stations[k] = default[k]

    station = Dataloader(start=stations["start"],
                         end=stations["end"],
                         station_id=stations["station_id"],
                         coords=stations["coords"],
                         data_dir=stations["data_dir"],
                         rain_snow=stations["rain_snow"],
                         phenology=stations["phenology"],
                         width=stations["width"])

    if stations["climate"]:
        station.get_climate()
        
    if stations["nlcd"]:
        station.get_nlcd(stations["years"])
    
    return station.data_dir


def batch(stations):
    """Process queries in batch in parallel.
    
    Parameters
    ----------
    stations : list of dict
        A list of dictionary containing the input variables:
        stations = [{
                     "start" : 'YYYY-MM-DD', [Requaired]
                     "end" : 'YYYY-MM-DD', [Requaired]
                     "station_id" : '<ID>', OR "coords" : (<lon>, <lat>), [Requaired]
                     "data_dir" : '<path/to/store/data>',  [Optional] Default : ./data
                     "climate" : True or Flase, [Optional] Default : False
                     "nlcd" : True or False, [Optional] Default : False
                     "years" : {'impervious': <YYYY>, 'cover': <YYYY>, 'canopy': <YYYY>}, [Optional] Default is 2016
                     "rain_snow" : True or False, [Optional] Default : False
                     "phenology" : True or False, [Optional] Default : False
                     "width" : 2000, [Optional] Default : 200
                     },
                    ...]
        
        
    """
    import multiprocessing
    import psutil

    # get number of physical cores
    max_procs = psutil.cpu_count(logical=False)

    pool = multiprocessing.Pool(processes=min(len(stations), max_procs))
    print(f"Processing queries in batch with {pool._processes} processors ...")

    data_dirs = pool.map(get_data, stations)
    pool.close()

    print("All the jobs finished successfully.")
    return data_dirs


def open_workspace(data_dir):
    """Open a hydrodata workspace using the root of data directory."""
    import xarray as xr
    from hydrodata import Dataloader
    import json
    import geopandas as gpd

    dirs = Path(data_dir).glob('*')
    stations = {}
    for d in dirs:
        if d.is_dir() and d.name.isdigit():
            wshed_file = d.joinpath("watershed.json")
            try:
                with open(wshed_file, "r") as fp:
                    watershed = json.load(fp)
                    gdf = None
                    for dictionary in watershed['featurecollection']:
                        if dictionary.get('name', '') == 'globalwatershed':
                            gdf = gpd.GeoDataFrame.from_features(dictionary['feature'])
                        
                    if gdf is None:
                        raise LookupError(f"Could not find 'globalwatershed' in the {wshed_file}.")
                    
                    wshed_params = watershed['parameters']
                    geometry = gdf.geometry.values[0]
            except FileNotFoundError:
                raise FileNotFoundError(f"{wshed_file} file cannot be found in {d}.")

            climates = []
            for clm in d.glob('*.nc'):
                climates.append(xr.open_dataset(clm))

            if len(climates) == 0:
                raise FileNotFoundError(f"No climate data file (*.nc) exits in {d}.")
            
            stations[d.name] = {'wshed_params' : wshed_params,
                                'geometry' : geometry,
                                'climates' : climates}
    if len(stations) == 0:
        print(f"No data was found in {data_dir}")
        return
    else:
        return stations
                