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

Features
---------

Hydrodata is a python library designed to aid in watershed analysis. It provides access to hydrology and climatology databases with some helper functions for visualization. Currently, the following data retrieval services are supported:

* `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ for getting watershed geometry and flowlines from NHDPlus V2
* `Daymet <https://daymet.ornl.gov/>`__ for climatology data, both single pixel and gridded
* `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`_ for daily actual evapotranspiration, both single pixel and gridded
* `NLCD 2016 <https://www.mrlc.gov/>`__ for land cover, land use
* `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ for USGS stations' daily streamflow observations
* `OpenTopography <https://opentopography.org/>`_ for Digital Elevation Model

The gridded data can be resampled to coarser or finer resolutions via the ``resolution`` argument (in decimal degree). The **resampling** is carried out using the **bilinear** method for continuous spatial data such as climate data and the **majority** method for discrete data such as land cover.

Additionally, the function for getting Daymet data offers a flag for computing **Potential Evapotranspiration** (PET) using the retrieved climate data. PET is computed based on `FAO-56 <http://www.fao.org/3/X0490E/X0490E00.htm>`_.

Requests for additional databases or functionalities can be submitted via `issues <https://github.com/cheginit/hydrodata/issues>`_.

Documentation
-------------

Learn more about Hydrodata in its official documentation at https://hydrodata.readthedocs.io.


Installation
------------

It's recommended to use `Conda <https://conda.io/en/latest/>`_ as the Python package management tool so the dependencies can be installed easily since Hydrodata is pure Python but its dependencies are not. This can be achieved using the ``environment.yml`` file provided in this repository. You can clone the repository or download the file from `here <https://raw.githubusercontent.com/cheginit/hydrodata/master/environment.yml>`_.

.. code-block:: console

    $ conda env create -f environment.yml

The environment can then be activate by issuing ``conda activate hydrodata``.

Alternatively, you can install the dependencies manually, then install Hydrodata using ``pip``:

.. code-block:: console

    $ pip install hydrodata

Quick Start
-----------

With just a few lines of code, Hydrodata provides easy access to a handful of databases. ``Station`` gathers the USGS site information such as name, contributing drainage area, upstream flowlines and watershed geometry.

.. code-block:: python

    from hydrodata import Station
    import hydrodata.datasets as hds

    lon, lat = -69.32, 45.17
    start, end = '2000-01-01', '2010-01-21'
    wshed = Station(start, end, coords=(lon, lat))

Using the retrieved information such as the watershed geometry we can then use the `datasets` module to access other databases. For example, we can find the USGS stations upstream (or downstream) of the main river channel (or tributatires) up to a certain distance, say 150 km. Also, all the USGS stations inside the watershed can be found:

.. code-block:: python

    stations = wshed.watershed.get_stations()
    stations_upto_150 = wshed.watershed.get_stations(navigation="upstreamMain", distance=150)

DEM can be retrieved for the station's contributing watershed and resampled from the original resolution of 1 arc-second (~30 m) to 30 arc-second (~1 km), as follows:

.. code-block:: python

    dem = hds.dem_bygeom(wshed.geometry, resolution=30.0/3600.0)

The climate data and streamflow observations for the location of interest can be retrieved as well:

.. code-block:: python

    clm_loc = hds.deymet_byloc(wshed.lon, wshed.lat, start=wshed.start, end=wshed.end)
    clm_loc['Q (cms)'] = hds.nwis(wshed.station_id, wshed.start, wshed.end)

Other than point-based data, gridded data can also be accessed at the desired resolution. Furthermore, the watershed geometry can be used to mask the gridded data:

.. code-block:: python

    variables = ["tmin", "tmax", "prcp"]
    clm_grd = hds.daymet_bygeom(wshed.geometry, start='2005-01-01', end='2005-01-31', variables=variables, pet=True)
    eta_grd = hds.ssebopeta_bygeom(wshed.geometry, start='2005-01-01', end='2005-01-31')

We can also find all or within certain distance USGS stations up- or downstream of the watershed outlet:

.. code-block:: python

    stations = wshed.watershed.get_stations()
    stations_upto_150 = wshed.watershed.get_stations(navigation="upstreamMain", distance=150)

All the gridded data are returned as `xarray <https://xarray.pydata.org/en/stable/>`_ datasets that has efficient data processing tools. Hydrodata also has a ``plot`` module that can plot five hydrologic signatures graphs in one plot. Some example plots are shown below that are produced with the following codes:

.. code-block:: python

    from hydrodata import plot

    plot.signatures(clm_loc['Q (cms)'], wshed.drainage_area, prcp=clm_loc['prcp (mm/day)'], title=wshed.name)
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
