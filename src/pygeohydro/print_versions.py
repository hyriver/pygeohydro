"""Utility functions for printing version information.

The original script is from
`xarray <https://github.com/pydata/xarray/blob/main/xarray/util/print_versions.py>`__
"""

from __future__ import annotations

import contextlib
import locale
import os
import platform
import struct
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TextIO

__all__ = ["show_versions"]


def _get_sys_info():
    """Return system information as a dict."""
    blob = []

    # get full commit hash
    commit = None
    if Path(".git").is_dir():
        with contextlib.suppress(Exception):
            pipe = subprocess.Popen(
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


def _get_package_version(modname: str) -> str:
    try:
        _ = distribution(modname)
        try:
            return get_version(modname)
        except (NotImplementedError, AttributeError):
            return "installed"
    except PackageNotFoundError:
        return "N/A"


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
        # async-retriever
        "aiodns",
        "aiofiles",
        "aiohttp",
        "aiohttp-client-cache",
        "aiosqlite",
        "brotli",
        "cytoolz",
        "ujson",
        # hydrosignatures
        "numpy",
        "pandas",
        "scipy",
        "xarray",
        "numba",
        "numbagg",
        # py3dep
        "click",
        "geopandas",
        "rasterio",
        "rioxarray",
        "shapely",
        # pydaymet/pygridmet/pynldas2
        "netcdf4",
        "pyproj",
        # pygeohydro
        "defusedxml",
        "folium",
        "h5netcdf",
        "matplotlib",
        "planetary-computer",
        "pystac-client",
        # pygeoogc
        "joblib",
        "multidict",
        "owslib",
        "requests",
        "requests-cache",
        "typing-extensions",
        "url-normalize",
        "urllib3",
        "yarl",
        # pygeoutils
        # (no unique dependencies not already listed)
        # pynhd
        "networkx",
        "pyarrow",
        "py7zr",
        # performance
        "flox",
        "opt-einsum"
    ]
    deps_blob = {modname: _get_package_version(modname) for modname in deps}

    print("\nSYS INFO", file=file)
    print("--------", file=file)
    for k, stat in _get_sys_info():
        print(f"{k}: {stat}", file=file)

    pad = len(max(deps, key=len)) + 1
    header = f"\n{'PACKAGE':<{pad}}  VERSION"
    print(header, file=file)
    print("-" * len(header), file=file)
    for k, stat in deps_blob.items():
        print(f"{k:<{pad}}  {stat}", file=file)
    print("-" * len(header), file=file)
