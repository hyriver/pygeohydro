.. highlight:: shell

============
Installation
============

Dependencies
------------

Dependencies of Hydrodata are as listed below:

- `Python 3.6+ <https://www.python.org/downloads>`_
- `PyNHD <https://github.com/cheginit/pynhd>`__
- `Py3DEP <https://github.com/cheginit/py3dep>`__
- `PyDaymet <https://github.com/cheginit/pydaymet>`__
- `Folium <https://python-visualization.github.io/folium/>`_
- `LXML <https://lxml.de/>`_

The following optional libraries are recommended to improve performance of ``xarray``:

- `Dask <https://dask.org>`_
- `NetCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`_
- `Bottleneck <https://pypi.org/project/Bottleneck>`_

Additionally, `CartoPy`_ can be installed to support more projections when plotting
geospatial data with ``matplotlib``. This library is specifically
useful for plotting `Daymet`_ data.

Stable release
--------------

You can install Hydrodata using ``pip``:

.. code-block:: console

    $ pip install hydrodata

Please note that installationa with ``pip`` fails if ``libgdal`` is not installed on your system.
You should install this package manually beforehand. For example, on Ubuntu-based distros
the required package is ``libgdal-dev``. If this package is installed on your system
you should be able to run ``gdal-config --version`` successfully.

Alternatively, Hydrodata and all its dependencies can be installed from ``conda-forge``
using `Conda`_:

.. code-block:: console

    $ conda install -c conda-forge hydrodata

From sources
------------

The sources for Hydrodata can be downloaded from its Github `repo`_.

You can either clone this public repository:

.. code-block:: console

    $ git clone https://github.com/cheginit/hydrodata.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/cheginit/hydrodata/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install .

Please refer to the note for installation with ``pip`` above, regarding the
``libgdal`` requirement.

.. _CartoPy: http://scitools.org.uk/cartopy
.. _Daymet: https://daymet.ornl.gov/overview
.. _Conda: https://docs.conda.io/en/latest
.. _repo: https://github.com/cheginit/hydrodata
.. _tarball: https://github.com/cheginit/hydrodata/tarball/master
