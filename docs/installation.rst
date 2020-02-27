.. highlight:: shell

============
Installation
============

Required dependencies
---------------------

- Python (3.6 or later)
- `numpy <http://www.numpy.org/>`_
- `pandas <http://pandas.pydata.org/>`__
    - `tables <https://www.pytables.org/usersguide/tutorials.html`_
- `xarray <https://xarray.pydata.org/en/stable/>`_
    - `dask <https://dask.org/`_
    - `netCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html`_
    - `bottleneck <https://pypi.org/project/Bottleneck/`_
    - `scipy <https://www.scipy.org/`_
    - `lxml <https://lxml.de/`_
- `requests <https://requests.readthedocs.io/en/master/>`_
- `owslib <https://geopython.github.io/OWSLib/>`_
- `geopandas <https://geopandas.org/`_
    - `descartes <https://pypi.org/project/descartes/`_
    - `geojson <https://pypi.org/project/geojson/`_
- `rasterio <https://github.com/mapbox/rasterio>`_
- `rasterstats <https://pythonhosted.org/rasterstats/`_
- `matplotlib <http://matplotlib.org/>`_
- `shapely <https://shapely.readthedocs.io/en/latest/>`_

Optional dependencies
---------------------

- `cartopy <http://scitools.org.uk/cartopy/>`_


Stable release
--------------

After installing dependecies, to install Hydrodata, run this command in your terminal:

.. code-block:: console

    $ pip install hydrodata

This is the preferred method to install Hydrodata, as it will always install the most recent stable release.

Alternatively, Hydrodata and all it's dependencies can be installed using `Conda <https://conda.io/en/latest/>`_ as follows:

.. code-block:: console

    $ wget https://raw.githubusercontent.com/cheginit/hydrodata/master/environment.yml
    $ conda env create -f environment.yml
    $ conda activate hydrodata

From sources
------------

The sources for Hydrodata can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/cheginit/hydrodata

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/cheginit/hydrodata/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate hydrodata
    $ python setup.py install


.. _Github repo: https://github.com/cheginit/hydrodata
.. _tarball: https://github.com/cheginit/hydrodata/tarball/master
