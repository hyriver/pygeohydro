.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :align: center

| 

.. image:: https://img.shields.io/pypi/v/hydrodata.svg
    :target: https://pypi.python.org/pypi/hydrodata

.. image:: https://codecov.io/gh/cheginit/hydrodata/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/hydrodata

.. image:: https://travis-ci.com/cheginit/hydrodata.svg?branch=master
    :target: https://travis-ci.com/cheginit/hydrodata

.. image:: https://readthedocs.org/projects/hydrodata/badge/?version=latest
    :target: https://hydrodata.readthedocs.io/en/latest/?badge=latest

.. image:: https://zenodo.org/badge/237573928.svg
    :target: https://zenodo.org/badge/latestdoi/237573928
        
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

-----------------

Hydrodata: A portal to access hydrology and climatology databases
------------------------------------------------------------------

Hydrodata is a python library designed to aid in watershed analysis. It provides access to hydrology and climatology databases with some helper functions for visualization. Currently, the following data retrieval services are supported:

* `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ for getting watershed geometry and flowlines from NHDPlus V2
* `Daymet <https://daymet.ornl.gov/>`__ for climatology data, both single pixel and gridded
* `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`_ for daily actual evapotranspiration
* `NLCD 2016 <https://www.mrlc.gov/>`__ for land cover, land use
* `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ for daily streamflow observations
* `OpenTopography <https://opentopography.org/>`_ for Digital Elevation Model

Support for additional databases can be requested via `issue <https://github.com/cheginit/hydrodata/issues>`_.

Documentation
-------------

Learn more about hydrodata in its official documentation at https://hydrodata.readthedocs.io.


Installation
------------

To install Hydrodata, run this command in your terminal:

.. code-block:: console

    $ pip install hydrodata

Alternatively, it can be installed from source by cloning the repository.

.. code-block:: console

    $ git clone https://github.com/cheginit/hydrodata.git
    $ cd hydrodata
    $ conda env create -f environment.yml
    $ conda activate hydrodata
    $ python setup.py install

Quick Start
-----------

With just a few lines of code, Hydrodata provides easy access to a handful of databases. ``Station`` gathers the USGS site information such as name, contributing drainage area and upstream flowlines and watershed geometry.

.. code-block:: python

    from hydrodata import Station
    import hydrodata.datasets as hds

    lon, lat = -69.32, 45.17
    start, end = '2000-01-01', '2010-01-21'
    wshed = Station(start, end, coords=(lon, lat))
    
DEM can be retrieved for the station's contributing watershed as follows:

.. code-block:: python

    dem = hds.dem_bygeom(wshed.geometry)

Then, we can get climate data and streamflow observation:

.. code-block:: python

    clm_loc = hds.deymet_byloc(wshed.lon, wshed.lat, start=wshed.start, end=wshed.end)
    clm_loc['Q (cms)'] = hds.nwis(wshed.station_id, wshed.start, wshed.end)

The watershed geometry can be used to clip the gridded data:

.. code-block:: python

    variables = ["tmin", "tmax", "prcp"]
    clm_grd = hds.daymet_bygeom(wshed.geometry, start='2005-01-01', end='2005-01-31', variables=variables, pet=True)
    eta_grd = hds.ssebopeta_bygeom(wshed.geometry, start='2005-01-01', end='2005-01-31')

We can also easily find all or within certain distance USGS stations up- or downstream of the watershed outlet:

.. code-block:: python

    stations = wshed.watershed.get_stations()
    stations_upto_150 = wshed.watershed.get_stations(navigation="upstreamMain", distance=150)

All the gridded data are returned as `xarray <https://xarray.pydata.org/en/stable/>`_ datasets with efficient data processing tools. Hydrodata also has a function called ``plot.signatures`` that can plot five hydrologic signatures graphs in one plot. Some example plots are shown below that are produced with the following codes:

.. code-block:: python

    from hydrodata import plot

    plot.signatures(clm_loc['Q (cms)'], wshed.drainage_area, prcp=clm_loc['prcp (mm/day)'], title=wshed.name)
    clm_grd.isel(time=1).tmin.plot(aspect=2, size=8)
    eta_grd.isel(time=4).eta.plot(size=8)
    
    ax = wshed.watershed.basin.plot(color='white', edgecolor='black', zorder=1, figsize = (10, 10))
    wshed.tributaries.plot(ax=ax, label='Tributaries', zorder=2)
    wshed.main_channel.plot(ax=ax, color='green', lw=3, label='Main', zorder=3)
    stations.plot(ax=ax, color='black', label='All stations', marker='s', zorder=4)
    stations_upto_150.plot(ax=ax, color='red', label='Stations up to 150 km upstream of main', marker='*', zorder=5)
    ax.legend(loc='best')

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots.png
        :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots.png

Contributing
------------

Hydrodata offers some limited statistical analysis. It could be more useful to the watershed modeling community to integrate more data exploratory capabilities to the package. Additionally, adding support for more databases such as water quality, phenology, and water level, are very welcome. If you are interested please get in touch. You can find information about contributing to hydrodata at our `Contributing page <https://hydrodata.readthedocs.io/en/latest/contributing.html>`_.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
