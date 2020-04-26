.. highlight:: shell

============
Installation
============

The following dependencies must be eirther installed manually or using the provided YAML file `here <https://raw.githubusercontent.com/cheginit/hydrodata/master/environment.yml>`_:

Dependencies
------------

- `Python 3.6+ <https://www.python.org/downloads>`_ 
- `NumPy <http://www.numpy.org>`_ 
- `Pandas <http://pandas.pydata.org>`_
- `Xarray <https://xarray.pydata.org>`_
- `Requests <https://requests.readthedocs.io>`_
- `OWSLib <https://geopython.github.io/OWSLib>`_
- `GeoPandas <https://geopandas.org>`_
- `RasterIO <https://github.com/mapbox/rasterio>`_
- `Matplotlib <http://matplotlib.org>`_
- `Shapely <https://shapely.readthedocs.io>`_
- `PQDM <https://pqdm.readthedocs.io>`_
- `SimpleJSON <https://simplejson.readthedocs.io>`_

Sub-dependencies
----------------

- `Tables <https://www.pytables.org/usersguide/tutorials.html>`_
- `Dask <https://dask.org>`_
- `NetCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`_
- `Bottleneck <https://pypi.org/project/Bottleneck>`_
- `SciPy <https://www.scipy.org>`_
- `Lxml <https://lxml.de>`_
- `Descartes <https://pypi.org/project/descartes>`_
- `GeoJSON <https://pypi.org/project/geojson>`_
- `ArcGIS2GeoJSON <https://github.com/chris48s/arcgis2geojson>`_

Optional
--------

- `NetworkX <https://networkx.github.io>`_ for flow accumulation
- `CartoPy <http://scitools.org.uk/cartopy>`_ for plotting

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
