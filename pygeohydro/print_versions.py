"""Utility functions for printing version information.

The original script is from
`xarray <https://github.com/pydata/xarray/blob/master/xarray/util/print_versions.py>`__
"""
import importlib
import locale
import os
import platform
import struct
import subprocess
import sys
from types import ModuleType
from typing import List, Optional, TextIO, Tuple

__all__ = ["show_versions"]


def netcdf_and_hdf5_versions() -> List[Tuple[str, Optional[str]]]:
    libhdf5_version = None
    libnetcdf_version = None
    try:
        import netCDF4

        libhdf5_version = netCDF4.__hdf5libversion__
        libnetcdf_version = netCDF4.__netcdf4libversion__
    except (ImportError, AttributeError):
        try:
            import h5py

            libhdf5_version = h5py.version.hdf5_version
        except (ImportError, AttributeError):
            pass
    return [("libhdf5", libhdf5_version), ("libnetcdf", libnetcdf_version)]


def get_sys_info() -> List[Tuple[str, Optional[str]]]:
    """Return system information as a dict.

    From https://github.com/numpy/numpy/blob/master/setup.py#L64-L89

    Returns
    -------
    list
        System information such as python version.
    """
    blob = []

    def _minimal_ext_cmd(cmd: List[str]) -> bytes:
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
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


def _get_mod(modname: str) -> ModuleType:
    if modname in sys.modules:
        return sys.modules[modname]
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError:
        return importlib.import_module(modname.replace("-", "_"))


def show_versions(file: TextIO = sys.stdout) -> None:
    """Print versions of all the dependencies.

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    deps = [
        #  async_retriever
        ("async-retriever", lambda mod: mod.__version__),
        ("aiodns", lambda mod: mod.__version__),
        ("aiohttp", lambda mod: mod.__version__),
        ("aiohttp-client-cache", lambda mod: mod.__version__),
        ("aiosqlite", lambda mod: mod.__version__),
        ("brotli", lambda mod: mod.__version__),
        ("cchardet", lambda mod: mod.__version__),
        ("cytoolz", lambda mod: mod.__version__),
        ("ujson", lambda mod: mod.__version__),
        #  pygeoogc
        ("pygeoogc", lambda mod: mod.__version__),
        ("defusedxml", lambda mod: mod.__version__),
        ("owslib", lambda mod: mod.__version__),
        ("pydantic", lambda mod: mod.version.VERSION),
        ("yaml", lambda mod: mod.__version__),
        ("pyproj", lambda mod: mod.__version__),
        ("requests", lambda mod: mod.__version__),
        ("requests-cache", lambda mod: mod.__version__),
        ("shapely", lambda mod: mod.__version__),
        ("urllib3", lambda mod: mod.__version__),
        #  pygeoutils
        ("pygeoutils", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),
        ("geopandas", lambda mod: mod.__version__),
        ("netCDF4", lambda mod: mod.__version__),
        ("numpy", lambda mod: mod.__version__),
        ("rasterio", lambda mod: mod.__version__),
        ("xarray", lambda mod: mod.__version__),
        ("rioxarray", lambda mod: mod.__version__),
        #  py3dep
        ("py3dep", lambda mod: mod.__version__),
        ("click", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        ("richdem", lambda mod: mod.pkg_resources.require("richdem")[0].version),
        #  pynhd
        ("pynhd", lambda mod: mod.__version__),
        ("networkx", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("pyarrow", lambda mod: mod.__version__),
        #  pygeohydro
        ("pygeohydro", lambda mod: mod.__version__),
        ("folium", lambda mod: mod.__version__),
        ("lxml", lambda mod: mod.__version__),
        ("matplotlib", lambda mod: mod.__version__),
        #  pydaymet
        ("pydaymet", lambda mod: mod.__version__),
        #  misc
        ("bottleneck", lambda mod: mod.__version__),
        ("pygeos", lambda mod: mod.__version__),
        ("tables", lambda mod: mod.__version__),
        #  test
        ("pytest", lambda mod: mod.__version__),
        ("pytest-cov", lambda mod: mod.__version__),
        ("xdist", lambda mod: mod.__version__),
    ]

    deps_blob: List[Tuple[str, Optional[str]]] = []
    for (modname, ver_f) in deps:
        try:
            mod = _get_mod(modname)
        except ModuleNotFoundError:
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
            except (NotImplementedError, AttributeError):
                ver = "installed"
            deps_blob.append((modname, ver))

    print("\nINSTALLED VERSIONS", file=file)
    print("------------------", file=file)

    for k, stat in get_sys_info():
        print(f"{k}: {stat}", file=file)

    print("", file=file)
    for k, stat in sorted(deps_blob):
        print(f"{k}: {stat}", file=file)
