#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some utilities for Hydrodata"""

from tqdm import tqdm
import urllib
from pathlib import Path
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException


class DownloadProgressBar(tqdm):
    """A tqdm-based class for download progress."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Insnp.pired from a tqdm example.

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


def get_nhd(data_dir):
    """Download and extract NHDPlus V2.1 database."""
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        try:
            import os

            os.mkdir(data_dir)
        except OSError:
            print(f"{data_dir} directory cannot be created")

    print(f"Downloading USGS gage information data to {str(data_dir)} >>>")
    base = "https://s3.amazonaws.com/edap-nhdplus/NHDPlusV21/" + "Data/NationalData/"
    dbname = [
        "NHDPlusV21_NationalData_GageInfo_05.7z",
        "NHDPlusV21_NationalData_GageLoc_05.7z",
    ]

    for db in dbname:
        download_extract(base + db, data_dir)
    return data_dir.joinpath("NHDPlusNationalData")


def retry_requests(
    retries=3,
    backoff_factor=0.5,
    status_to_retry=(500, 502, 504),
    prefixes=("http://", "https://"),
):
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
        method_whitelist=False,
    )
    adapter = HTTPAdapter(max_retries=r)
    for prefix in prefixes:
        session.mount(prefix, adapter)
    session.hooks = {"response": lambda r, *args, **kwargs: r.raise_for_status()}

    return session


def get_data(stations):
    from hydrodata import Dataloader

    default = dict(
        start=None,
        end=None,
        station_id=None,
        coords=None,
        data_dir="./data",
        rain_snow=False,
        phenology=False,
        width=2000,
        climate=False,
        nlcd=False,
        yreas={"impervious": 2016, "cover": 2016, "canopy": 2016},
    )

    params = list(stations.keys())

    if "station_id" in params and "coords" in params:
        if stations["station_id"] is not None and stations["coords"] is not None:
            raise KeyError("Either coords or station_id should be provided.")

    for k in list(default.keys()):
        if k not in params:
            stations[k] = default[k]

    station = Dataloader(
        start=stations["start"],
        end=stations["end"],
        station_id=stations["station_id"],
        coords=stations["coords"],
        data_dir=stations["data_dir"],
        rain_snow=stations["rain_snow"],
        phenology=stations["phenology"],
        width=stations["width"],
    )

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
        stations = [{"start" : 'YYYY-MM-DD', [Requaired]
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
    from concurrent import futures
    import psutil

    with futures.ThreadPoolExecutor() as executor:
        data_dirs = list(executor.map(get_data, stations))

    print("All the jobs finished successfully.")
    return data_dirs


def open_workspace(data_dir):
    """Open a hydrodata workspace using the root of data directory."""
    import xarray as xr
    from hydrodata import Dataloader
    import json
    import geopandas as gpd

    dirs = Path(data_dir).glob("*")
    stations = {}
    for d in dirs:
        if d.is_dir() and d.name.isdigit():
            wshed_file = d.joinpath("watershed.json")
            try:
                with open(wshed_file, "r") as fp:
                    watershed = json.load(fp)
                    gdf = None
                    for dictionary in watershed["featurecollection"]:
                        if dictionary.get("name", "") == "globalwatershed":
                            gdf = gpd.GeoDataFrame.from_features(dictionary["feature"])

                    if gdf is None:
                        raise LookupError(
                            f"Could not find 'globalwatershed' in the {wshed_file}."
                        )

                    wshed_params = watershed["parameters"]
                    geometry = gdf.geometry.values[0]
            except FileNotFoundError:
                raise FileNotFoundError(f"{wshed_file} file cannot be found in {d}.")

            climates = []
            for clm in d.glob("*.nc"):
                climates.append(xr.open_dataset(clm))

            if len(climates) == 0:
                raise FileNotFoundError(f"No climate data file (*.nc) exits in {d}.")

            stations[d.name] = {
                "wshed_params": wshed_params,
                "geometry": geometry,
                "climates": climates,
            }
    if len(stations) == 0:
        print(f"No data was found in {data_dir}")
        return
    else:
        return stations


def get_location(lon, lat, retry=3):
    """Get the state code and county name from US Censue database.
    
    Parameters
    ----------
    lon : float
        Longitude
    lan : float
        Latitude
        
    Returns
    -------
    state : string
        The state code
    county : string
        The county name
    retry : int
        Number of retries if request fails. The default is 3.
    """
    import time
    import geocoder
    
    retry = int(retry)

    try:
        g = geocoder.uscensus([lat, lon], method="reverse")
        state = g.geojson["features"][0]["properties"]["raw"]["States"][0]["STUSAB"]
        county = g.geojson['features'][0]['properties']['county']

        if state is None or county is None:
            time.sleep(0.5)
            get_location(lon, lat, retry=retry-1)

        return state, county
    except KeyError:
        if retry <= 0:
            raise KeyError("The location should be inside the US.")
        else:
            time.sleep(0.5)
            get_location(lon, lat, retry=retry-1)


def daymet_dates(start, end):
    """Correct dates for Daymet when leap years.
    
    Daymet doesn't account for leap years and removes
    Dec 31 when it's leap year. This function returns all
    the dates in the Daymet database within the provided year.
    """
    import pandas as pd

    period = pd.date_range(start, end)
    nl = period[~period.is_leap_year]
    lp = period[
        (period.is_leap_year) & (~period.strftime("%Y-%m-%d").str.endswith("12-31"))
    ]
    period = period[(period.isin(nl)) | (period.isin(lp))]
    years = [period[period.year == y] for y in period.year.unique()]
    return [(y[0], y[-1]) for y in years]


def get_elevation(lon, lat):
    """Get elevation from USGS 3DEP service for a coordinate.
    
    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude
        
    Returns
    -------
    elevation : float
        Elevation in meter
    """
    url = 'https://nationalmap.gov/epqs/pqs.php?'
    session = retry_requests()
    lon = lon if isinstance(lon, list) else [lon]
    lat = lat if isinstance(lat, list) else [lat]
    coords = [(i, j) for i, j in zip(lon, lat)]
    try:
        payload = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters'
        }
        r = session.get(url, params=payload)
    except ConnectionError or Timeout or RequestException:
        raise
    elevation = r.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    if elevation == -1000000:
        raise ValueError(f'The altitude of the requested coordinate ({lon}, {lat}) cannot be found.')
    else:
        return elevation


def get_elevation_bybbox(bbox, coords, data_dir=None):
    """Get elevation from DEM data for a list of coordinates.
    
    The elevations are extracted from SRTM3 (90-m resolution) data.
    This function is intended for getting elevations for ds data.
    
    Parameters
    ----------
    bbox : list
        Bounding box with coordinates in [west, south, east, north] format.
    coords : list of tuples
        A list of coordinates in (lon, lat) foramt.
    data_dir : string or Path
        The path to the directory for saving the DEM file in `tif` format.
        The file name is the name of the county where the center of the bbox
        is located. If None the DEM file is not saved. The default is None.
        
    Returns
    -------
    elevations : array
        A numpy array of elevations in meter
    output : Path
        Path to the downloaded DEM file only if data_dir is not None.
    """
    import elevation
    import rasterio
    import numpy as np

    lon, lat = (bbox[0] + bbox[2])*0.5, (bbox[1] + bbox[3])*0.5
    _, county = get_location(lon, lat)
    output = county.replace(' ', '_') + '.tif'

    root = '.' if data_dir is None else data_dir

    output = Path(root, output).absolute()
    if not output.exists():
        elevation.clip(bounds=bbox, output=str(output), product='SRTM3')
        elevation.clean()

    with rasterio.open(output) as src:
        elevations = np.array([e[0] for e in src.sample(coords)], dtype=np.float32)

    if data_dir is None:
        output.unlink()
        return elevations
    else:
        return elevations, output


def pet_fao(df, lon, lat):
    """Compute Potential EvapoTranspiration using Daymet dataset.
    
    The method is based on `FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`.
    The following variables are required:
    tmin (deg c), tmax (deg c), lat, lon, vp (Pa), srad (W/m^2), dayl (s)
    """
    import numpy as np
    import pandas as pd
    
    keys = [v for v in df.columns]
    reqs = ['tmin (deg c)', 'tmax (deg c)', 'vp (Pa)', 'srad (W/m^2)', 'dayl (s)']

    missing = [r for r in reqs if r not in keys]
    if len(missing) > 0:
        msg = "These required variables are not in the dataset: "
        msg += ", ".join(x for x in missing)
        msg += f'\nRequired variables are {", ".join(x for x in reqs)}'
        raise KeyError(msg)
    
    dtype = df.dtypes[0]
    df['tmean (deg c)'] = 0.5*(df['tmax (deg c)'] + df['tmin (deg c)'])
    Delta = 4098*(0.6108*np.exp(17.27*df['tmean (deg c)']/(df['tmean (deg c)'] + 237.3), dtype=dtype))/((df['tmean (deg c)'] + 237.3)**2)
    elevation = get_elevation(lon, lat)

    P = 101.3*((293.0 - 0.0065*elevation)/293.0)**5.26
    gamma = P*0.665e-3

    G = 0.0  # recommended for daily data
    df['vp (Pa)'] = df['vp (Pa)']*1e-3

    e_max = 0.6108*np.exp(17.27*df['tmax (deg c)']/(df['tmax (deg c)'] + 237.3), dtype=dtype)
    e_min = 0.6108*np.exp(17.27*df['tmin (deg c)']/(df['tmin (deg c)'] + 237.3), dtype=dtype)
    e_s = (e_max + e_min)*0.5
    e_def = e_s - df['vp (Pa)']

    u_2 = 2.0  # recommended when no data is available

    jday = df.index.dayofyear
    R_s = df['srad (W/m^2)']*df['dayl (s)']*1e-6

    alb = 0.23

    jp = 2.0*np.pi*jday/365.0
    d_r = 1.0 + 0.033*np.cos(jp, dtype=dtype)
    delta = 0.409*np.sin(jp - 1.39, dtype=dtype)
    phi = lat*np.pi/180.0
    w_s = np.arccos(-np.tan(phi, dtype=dtype)*np.tan(delta, dtype=dtype))
    R_a = 24.0*60.0/np.pi*0.082*d_r*(w_s*np.sin(phi, dtype=dtype)*np.sin(delta, dtype=dtype)
                                  + np.cos(phi, dtype=dtype)*np.cos(delta, dtype=dtype)*np.sin(w_s, dtype=dtype))
    R_so = (0.75 + 2e-5*elevation)*R_a
    R_ns = (1.0 - alb)*R_s
    R_nl = 4.903e-9*(((df['tmax (deg c)'] + 273.16)**4
                      + (df['tmin (deg c)'] + 273.16)**4)*0.5)*(0.34 - 0.14*np.sqrt(df['vp (Pa)']))*((1.35*R_s/R_so) - 0.35)
    R_n = R_ns - R_nl

    df['pet (mm/day)'] = (0.408*Delta*(R_n - G) + gamma*900.0/(df['tmean (deg c)'] + 273.0)*u_2*e_def) \
                         / (Delta + gamma*(1 + 0.34*u_2))
    df['vp (Pa)'] = df['vp (Pa)']*1.0e3
    
    return df


def pet_fao_gridded(ds):
    """Compute Potential EvapoTranspiration using Daymet dataset.
    
    The method is based on `FAO 56 paper <http://www.fao.org/docrep/X0490E/X0490E00.htm>`.
    The following variables are required:
    tmin (deg c), tmax (deg c), lat, lon, vp (Pa), srad (W/m2), dayl (s/day)
    """
    import numpy as np
    import pandas as pd
    
    keys = [v for v in ds.keys()]
    reqs = ['tmin', 'tmax', 'lat', 'lon', 'vp', 'srad', 'dayl']

    missing = [r for r in reqs if r not in keys]
    if len(missing) > 0:
        msg = "These required variables are not in the dataset: "
        msg += ", ".join(x for x in missing)
        msg += f'\nRequired variables are {", ".join(x for x in reqs)}'
        raise KeyError(msg)
    
    dtype = ds.tmin.dtype
    dates = ds['time']
    ds['tmean'] = 0.5*(ds['tmax'] + ds['tmin'])
    ds['tmean'].attrs['units'] = 'degree C'
    ds['delta'] = 4098*(0.6108*np.exp(17.27*ds['tmean']/(ds['tmean'] + 237.3), dtype=dtype))/((ds['tmean'] + 237.3)**2)
    
    no_elev = False
    if 'elevation' not in keys:
        coords = [(i, j) for i, j in zip(ds.sel(time=ds['time'][0]).lon.values.flatten(),
                                         ds.sel(time=ds['time'][0]).lat.values.flatten())]
        no_elev = True
        margine = 0.1
        bbox = [ds.lon.min().values - margine,
                ds.lat.min().values - margine,
                ds.lon.max().values + margine,
                ds.lat.max().values + margine]
        elevation = get_elevation_bybbox(bbox, coords).reshape(ds.dims['y'], ds.dims['x'])
        ds['elevation'] = ({'y' : ds.dims['y'], 'x' : ds.dims['x']}, elevation)

    P = 101.3*((293.0 - 0.0065*ds['elevation'])/293.0)**5.26
    ds['gamma'] = P*0.665e-3

    G = 0.0  # recommended for daily data
    ds['vp'] *= 1e-3

    e_max = 0.6108*np.exp(17.27*ds['tmax']/(ds['tmax'] + 237.3), dtype=dtype)
    e_min = 0.6108*np.exp(17.27*ds['tmin']/(ds['tmin'] + 237.3), dtype=dtype)
    e_s = (e_max + e_min)*0.5
    ds['e_def'] = e_s - ds['vp']

    u_2 = 2.0  # recommended when no data is available

    lat = ds.sel(time=ds['time'][0]).lat
    ds['time'] = pd.to_datetime(ds.time.values).dayofyear.astype(dtype)
    R_s = ds['srad']*ds['dayl']*1e-6

    alb = 0.23

    jp = 2.0*np.pi*ds['time']/365.0
    d_r = 1.0 + 0.033*np.cos(jp, dtype=dtype)
    delta = 0.409*np.sin(jp - 1.39, dtype=dtype)
    phi = lat*np.pi/180.0
    w_s = np.arccos(-np.tan(phi, dtype=dtype)*np.tan(delta, dtype=dtype))
    R_a = 24.0*60.0/np.pi*0.082*d_r*(w_s*np.sin(phi, dtype=dtype)*np.sin(delta, dtype=dtype)
                                  + np.cos(phi, dtype=dtype)*np.cos(delta, dtype=dtype)*np.sin(w_s, dtype=dtype))
    R_so = (0.75 + 2e-5*ds['elevation'])*R_a
    R_ns = (1.0 - alb)*R_s
    R_nl = 4.903e-9*(((ds['tmax'] + 273.16)**4
                      + (ds['tmin'] + 273.16)**4)*0.5)*(0.34 - 0.14*np.sqrt(ds['vp']))*((1.35*R_s/R_so) - 0.35)
    ds['R_n'] = R_ns - R_nl

    ds['pet'] = (0.408*ds['delta']*(ds['R_n'] - G) + ds['gamma']*900.0/(ds['tmean'] + 273.0)*u_2*ds['e_def']) \
                     / (ds['delta'] + ds['gamma']*(1 + 0.34*u_2))
    ds['pet'].attrs['units'] = 'mm/day'

    ds['time'] = dates
    ds['vp'] *= 1.0e3
    if no_elev:
        ds = ds.drop_vars(['delta', 'gamma', 'e_def', 'R_n', 'elevation'])
    else:
        ds['elevation'].attrs['units'] = 'm'
        ds = ds.drop_vars(['delta', 'gamma', 'e_def', 'R_n'])
    
    return ds