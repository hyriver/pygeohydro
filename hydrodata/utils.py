#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Some utilities for Hydrodata"""

from tqdm import tqdm
import urllib
from pathlib import Path


def retry(num_attempts=3, exception_class=Exception, log=None, sleeptime=1):
    """Retry a function after some exception.
    
    From https://codereview.stackexchange.com/questions/188539/python-code-to-retry-function
    """
    import functools
    import logging
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(num_attempts):
                try:
                    return func(*args, **kwargs)
                except exception_class as e:
                    if i == num_attempts - 1:
                        raise
                    else:
                        if log:
                            logging.error(f"Failed with error {e}, trying again")
                        time.sleep(sleeptime)

        return wrapper

    return decorator


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
