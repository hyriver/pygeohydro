.. highlight:: shell

============
Installation
============


Dependencies
------------
+===================================================+==========================================================================+=============================================+
|                    Dependencies                   |                             Sub-dependencies                             |                   Optional                  |
+===================================================+==========================================================================+=============================================+
| `Python 3.6+ <https://www.python.org/downloads>`_ |      `Tables <https://www.pytables.org/usersguide/tutorials.html>`_      | `CartoPy <http://scitools.org.uk/cartopy>`_ |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|          `NumPy <http://www.numpy.org/>`_         |                        `Dask <https://dask.org/>`_                       |   `NetworkX <https://networkx.github.io>`_  |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|        `Pandas <http://pandas.pydata.org>`_       | `NetCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`_ |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|   `Xarra <https://xarray.pydata.org/en/stable>`_  |            `Bottleneck <https://pypi.org/project/Bottleneck>`_           |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|   `Requests <https://requests.readthedocs.io>`_   |                     `SciPy <https://www.scipy.org>`_                     |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|   `OWSLib <https://geopython.github.io/OWSLib>`_  |                         `Lxml <https://lxml.de>`_                        |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|        `GeoPandas <https://geopandas.org>`_       |             `Descartes <https://pypi.org/project/descartes>`_            |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|  `RasterIO <https://github.com/mapbox/rasterio>`_ |               `GeoJSON <https://pypi.org/project/geojson>`_              |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|       `Matplotlib <http://matplotlib.org>`_       |                                                                          |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+
|    `Shapely <https://shapely.readthedocs.io>`_    |                                                                          |                                             |
+---------------------------------------------------+--------------------------------------------------------------------------+---------------------------------------------+

Stable release
--------------

After installing dependencies, to install Hydrodata, run this command in your terminal:

.. code-block:: console

    pip install hydrodata

This is the preferred method to install Hydrodata, as it will always install the most recent stable release.

Alternatively, Hydrodata and all its dependencies can be installed using `Conda <https://conda.io/en/latest/>`_ as follows:

.. code-block:: console

    wget https://raw.githubusercontent.com/cheginit/hydrodata/master/environment.yml
    conda env create -f environment.yml
    conda activate hydrodata

From sources
------------

The sources for Hydrodata can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/cheginit/hydrodata

Or download the `tarball`_:

.. code-block:: console

    curl -OJL https://github.com/cheginit/hydrodata/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    conda env create -f environment.yml
    conda activate hydrodata
    python setup.py install


.. _Github repo: https://github.com/cheginit/hydrodata
.. _tarball: https://github.com/cheginit/hydrodata/tarball/master
