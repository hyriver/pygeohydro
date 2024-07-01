"""Utility functions for printing version information.

The original script is from
`xarray <https://github.com/pydata/xarray/blob/main/xarray/util/print_versions.py>`__
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import locale
import os
import platform
import struct
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TextIO

__all__ = ["show_versions"]


def netcdf_and_hdf5_versions() -> list[tuple[str, str | None]]:
    libhdf5_version = None
    libnetcdf_version = None

    if importlib.util.find_spec("netCDF4"):
        import netCDF4

        libhdf5_version = netCDF4.__hdf5libversion__
        libnetcdf_version = netCDF4.__netcdf4libversion__
    elif importlib.util.find_spec("h5py"):
        import h5py

        libhdf5_version = h5py.version.hdf5_version

    return [("libhdf5", libhdf5_version), ("libnetcdf", libnetcdf_version)]


def get_sys_info():
    """Return system information as a dict."""
    blob = []

    # get full commit hash
    commit = None
    if Path(".git").is_dir():
        with contextlib.suppress(Exception):
            pipe = subprocess.Popen(  # noqa: S603
                'git log --format="%H" -n 1'.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, _ = pipe.communicate()

            if pipe.returncode == 0:
                commit = so
                with contextlib.suppress(ValueError):
                    commit = so.decode("utf-8")
                    commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    with contextlib.suppress(Exception):
        (sysname, _, release, _, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", str(sysname)),
                ("OS-release", str(release)),
                ("machine", str(machine)),
                ("processor", str(processor)),
                ("byteorder", str(sys.byteorder)),
                ("LC_ALL", os.environ.get("LC_ALL", "None")),
                ("LANG", os.environ.get("LANG", "None")),
                ("LOCALE", str(locale.getlocale())),
            ]
        )
    return blob


def show_versions(file: TextIO = sys.stdout) -> None:
    """Print versions of all the dependencies.

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    deps = [
        # HyRiver packages
        "async-retriever",
        "pygeoogc",
        "pygeoutils",
        "py3dep",
        "pynhd",
        "pygridmet",
        "pydaymet",
        "hydrosignatures",
        "pynldas2",
        "pygeohydro",
        #  async-retriever deps
        "aiohttp",
        "aiohttp-client-cache",
        "aiosqlite",
        "cytoolz",
        "ujson",
        #  pygeoogc deps
        "defusedxml",
        "joblib",
        "multidict",
        "owslib",
        "pyproj",
        "requests",
        "requests-cache",
        "shapely",
        "url-normalize",
        "urllib3",
        "yarl",
        #  pygeoutils deps
        "geopandas",
        "netcdf4",
        "numpy",
        "rasterio",
        "rioxarray",
        "scipy",
        "shapely",
        "ujson",
        "xarray",
        #  py3dep deps
        "click",
        "pyflwdir",
        #  pynhd deps
        "networkx",
        "pyarrow",
        #  pygeohydro deps
        "folium",
        "h5netcdf",
        "matplotlib",
        "pandas",
        #  optional
        "numba",
        "bottleneck",
        "py7zr",
        "pyogrio",
    ]
    pad = len(max(deps, key=len)) + 1

    deps_blob = {}
    for modname in deps:
        try:
            deps_blob[modname] = get_version(modname)
        except PackageNotFoundError:
            deps_blob[modname] = "N/A"
        except (NotImplementedError, AttributeError):
            deps_blob[modname] = "installed"

    print("\nSYS INFO", file=file)
    print("--------", file=file)

    for k, stat in get_sys_info():
        print(f"{k}: {stat}", file=file)

    header = f"\n{'PACKAGE':<{pad}}  VERSION"
    print(header, file=file)
    print("-" * len(header), file=file)
    for k, stat in deps_blob.items():
        print(f"{k:<{pad}}  {stat}", file=file)
    print("-" * len(header), file=file)
