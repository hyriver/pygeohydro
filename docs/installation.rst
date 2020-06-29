.. highlight:: shell

============
Installation
============

Dependencies
------------

- `Python 3.6+ <https://www.python.org/downloads>`_
- `NumPy <http://www.numpy.org>`_
- `Pandas <http://pandas.pydata.org>`_
- `Xarray <https://xarray.pydata.org>`_
- `SciPy <https://www.scipy.org>`_
- `OWSLib <https://geopython.github.io/OWSLib>`_
- `GeoPandas <https://geopandas.org>`_
- `GeoJSON <https://pypi.org/project/geojson>`_
- `RasterIO <https://github.com/mapbox/rasterio>`_
- `Matplotlib <http://matplotlib.org>`_
- `Shapely <https://shapely.readthedocs.io>`_
- `SimpleJSON <https://simplejson.readthedocs.io>`_
- `Folium <https://python-visualization.github.io/folium/>`_
- `Requests <https://requests.readthedocs.io>`_
- `DefusedXML <https://github.com/tiran/defusedxml>`_
- `NetworkX <https://networkx.github.io>`_

Recommended
-----------

- `Dask <https://dask.org>`_
- `NetCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`_
- `Bottleneck <https://pypi.org/project/Bottleneck>`_

Optional
--------
- `CartoPy <http://scitools.org.uk/cartopy>`_ for plotting in various projections

Stable release
--------------

You can install Hydrodata using ``pip`` after installing ``libgdal`` on your system (for example ``libgdal-dev`` in Ubuntu):

.. code-block:: console

    $ pip install hydrodata

Alternatively, it can be installed from ``conda-forge`` using `Conda`_:

.. code-block:: console

    $ conda install -c conda-forge hydrodata

From sources
------------

The sources for Hydrodata can be downloaded from the Github `repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/cheginit/hydrodata

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/cheginit/hydrodata/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install . --no-deps


.. _Conda: https://docs.conda.io/en/latest
.. _repo: https://github.com/cheginit/hydrodata
.. _tarball: https://github.com/cheginit/hydrodata/tarball/master
