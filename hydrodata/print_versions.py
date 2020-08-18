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


def get_sys_info():
    """Return system information as a dict.

    From https://github.com/numpy/numpy/blob/master/setup.py#L64-L89
    """
    blob = []

    def _minimal_ext_cmd(cmd):
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

    (sysname, _nodename, release, _version, machine, processor) = platform.uname()
    blob.extend(
        [
            ("python", sys.version),
            ("python-bits", struct.calcsize("P") * 8),
            ("OS", f"{sysname}"),
            ("OS-release", f"{release}"),
            ("machine", f"{machine}"),
            ("processor", f"{processor}"),
            ("byteorder", f"{sys.byteorder}"),
            ("LC_ALL", f'{os.environ.get("LC_ALL", "None")}'),
            ("LANG", f'{os.environ.get("LANG", "None")}'),
            ("LOCALE", ".".join(locale.getlocale())),
        ]
    )

    return blob


def netcdf_and_hdf5_versions():
    """Get netcdf and hdf5 versions."""
    libhdf5_version = None
    libnetcdf_version = None
    try:
        import netCDF4

        libhdf5_version = netCDF4.__hdf5libversion__
        libnetcdf_version = netCDF4.__netcdf4libversion__
    except ImportError:
        try:
            import h5py

            libhdf5_version = h5py.version.hdf5_version
        except ImportError:
            pass
    return [("libhdf5", libhdf5_version), ("libnetcdf", libnetcdf_version)]


def show_versions(file=sys.stdout):
    """Print the versions of hydrodata stack and its dependencies.

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except OSError as e:
        print(f"Error collecting netcdf / hdf5 version: {e}")

    deps = [
        # (MODULE_NAME, f(mod) -> mod version)
        # hydrodata stack
        ("hydrodata", lambda mod: mod.__version__),
        ("pygeoogc", lambda mod: mod.__version__),
        ("pygeoutils", lambda mod: mod.__version__),
        ("py3dep", lambda mod: mod.__version__),
        ("pynhd", lambda mod: mod.__version__),
        ("pydaymet", lambda mod: mod.__version__),
        # dependencies
        ("xarray", lambda mod: mod.__version__),
        ("rasterio", lambda mod: mod.__version__),
        ("geopandas", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("orjson", lambda mod: mod.__version__),
        ("shapely", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        ("netCDF4", lambda mod: mod.__version__),
        ("defusedxml", lambda mod: mod.__version__),
        ("owslib", lambda mod: mod.__version__),
        ("pyproj", lambda mod: mod.__version__),
        ("requests", lambda mod: mod.__version__),
        ("matplotlib", lambda mod: mod.__version__),
        # optionals
        ("bottleneck", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),
        ("cartopy", lambda mod: mod.__version__),
        ("seaborn", lambda mod: mod.__version__),
        # setup/test
        ("setuptools", lambda mod: mod.__version__),
        ("pip", lambda mod: mod.__version__),
        ("conda", lambda mod: mod.__version__),
        ("pytest", lambda mod: mod.__version__),
        # Misc.
        ("IPython", lambda mod: mod.__version__),
        ("sphinx", lambda mod: mod.__version__),
    ]

    deps_blob = []
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except NotImplementedError:
                deps_blob.append((modname, "installed"))

    print("\nINSTALLED VERSIONS", file=file)
    print("------------------", file=file)

    for k, stat in sys_info:
        print(f"{k}: {stat}", file=file)

    print("", file=file)
    for k, stat in deps_blob:
        print(f"{k}: {stat}", file=file)
