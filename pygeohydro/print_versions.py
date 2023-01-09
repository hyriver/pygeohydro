"""Utility functions for printing version information.

The original script is from
`xarray <https://github.com/pydata/xarray/blob/master/xarray/util/print_versions.py>`__
"""
from __future__ import annotations

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


def get_sys_info() -> list[tuple[str, str | None]]:
    """Return system information as a dict.

    From https://github.com/numpy/numpy/blob/master/setup.py#L64-L89

    Returns
    -------
    list
        System information such as python version.
    """
    blob = []

    def _minimal_ext_cmd(cmd: list[str]) -> bytes:
        # construct minimal environment
        env = {}
        for k in ("SYSTEMROOT", "PATH", "HOME"):
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    commit = None
    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        commit = out.strip().decode("ascii")
    except (subprocess.SubprocessError, OSError):
        pass

    blob.append(("commit", commit))

    (sysname, _, release, _, machine, processor) = platform.uname()
    blob.extend(
        [
            ("python", sys.version),
            ("python-bits", f"{struct.calcsize('P') * 8}"),
            ("OS", f"{sysname}"),
            ("OS-release", f"{release}"),
            ("machine", f"{machine}"),
            ("processor", f"{processor}"),
            ("byteorder", f"{sys.byteorder}"),
            ("LC_ALL", f'{os.environ.get("LC_ALL", "None")}'),
            ("LANG", f'{os.environ.get("LANG", "None")}'),
            ("LOCALE", ".".join(str(i) for i in locale.getlocale())),
        ],
    )
    blob.extend(netcdf_and_hdf5_versions())

    return blob


def show_versions(file: TextIO = sys.stdout) -> None:
    """Print versions of all the dependencies.

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    deps = [
        #  async_retriever
        "async-retriever",
        "aiodns",
        "aiohttp",
        "aiohttp-client-cache",
        "aiosqlite",
        "brotli",
        "cytoolz",
        "ujson",
        #  pygeoogc
        "pygeoogc",
        "defusedxml",
        "owslib",
        "yaml",
        "pyproj",
        "requests",
        "requests-cache",
        "shapely",
        "urllib3",
        #  pygeoutils
        "pygeoutils",
        "dask",
        "geopandas",
        "netCDF4",
        "numpy",
        "rasterio",
        "xarray",
        "rioxarray",
        #  py3dep
        "py3dep",
        "click",
        "scipy",
        "richdem",
        #  pynhd
        "pynhd",
        "networkx",
        "pandas",
        "pyarrow",
        #  pygeohydro
        "pygeohydro",
        "folium",
        "lxml",
        "matplotlib",
        #  pydaymet
        "pydaymet",
        #  hydrosignatures
        "hydrosignatures",
        #  pynldas2
        "pynldas2",
        "h5netcdf",
        #  misc
        "numba",
        "bottleneck",
        "pygeos",
        "tables",
        #  test
        "pytest",
        "pytest-cov",
        "xdist",
    ]
    pad = len(max(deps, key=len)) + 1

    deps_blob = {}
    for modname in sorted(deps):
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
