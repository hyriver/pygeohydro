Hydrodata
=========


.. image:: https://img.shields.io/pypi/v/hydrodata.svg
        :target: https://pypi.python.org/pypi/hydrodata

.. image:: https://codecov.io/gh/cheginit/hydrodata/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/cheginit/hydrodata

.. image:: https://travis-ci.com/cheginit/hydrodata.svg?branch=master
        :target: https://travis-ci.com/cheginit/hydrodata.svg?branch=master

.. image:: https://readthedocs.org/projects/hydrodata/badge/?version=latest
        :target: https://hydrodata.readthedocs.io/en/latest/?badge=latest

.. image:: https://zenodo.org/badge/237573928.svg
        :target: https://zenodo.org/badge/latestdoi/237573928

Hydrodata is a python library designed to aid in NHDPlus watershed analysis. Hydrodata is capable of downloading,
preprocessing, and visualizing climatological, hydrological, and geographical datasets pertaining to a given watershed.
Supported datasets include: Daymet climate, USGS streamflow, and NLCD data with further additions planned.

* Documentation: https://hydrodata.readthedocs.io

Features
--------

* Download watersheds geometry and flowlines (NHDPlus V2) using `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ service.
* Download daily climate data by point and gridded from `Daymet <https://daymet.ornl.gov/>`__ database.
* Compute potential evapotranspiration for both point and gridded data based on `FAO-56 <http://www.fao.org/3/X0490E/X0490E00.htm>`__.
* Download actual evapotranspiration from `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`_ service.
* Download land use, land cover data from `NLCD 2016 <https://www.mrlc.gov/>`__ database.
* Download daily streamflow observations from `USGS NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ database.
* Plot hydrological signature graphs.


Usage
-----

With just a few lines of code Hydrodata provides easy access to a handful of databases. ``Station`` gathers the USGS site inforamtion such as name, contributing drainage area and upstream flowlines and watershed geometry.

.. code-block:: python

    from hydrodata import Station
    import hydrodata.datasets as hds

    lon, lat = -69.32, 45.17
    start, end = '2000-01-01', '2010-01-21'
    wshed = Station(start, end, coords=(lon, lat))
    
Then, we can get climate data and streamflow observation:

.. code-block:: python

    clm_loc = hds.deymet_byloc(wshed.lon, wshed.lat, start=wshed.start, end=wshed.end)
    clm_loc['Q (cms)'] = hds.nwis(wshed.station_id, wshed.start, wshed.end)

The watershed geometry can be used to clip the gridded data:

.. code-block:: python

    variables = ["tmin", "tmax", "prcp"]
    clm_grd = hds.daymet_bygeom(wshed.geometry, start='2005-01-01', end='2005-01-31', variables=variables, pet=True)
    eta_grd = hds.ssebopeta_bygeom(wshed.geometry, start='2005-01-01', end='2005-01-31')

We can also easily find all or within a certain distance USGS stations up- or downstream of the watershed outlet:

.. code-block:: python

    stations = wshed.watershed.get_stations()
    stations_upto_150 = wshed.watershed.get_stations(navigation="upstreamMain", distance=150)

All the gridded data are returned as `xarray <https://xarray.pydata.org/en/stable/>`_ datasets with efficient data processing tools. Hydrodata also has a function called ``plot.signatures`` that can plot five hydrologic signatures graphs in one plot. Some example plots are shown below that are produced with the following codes:

.. code-block:: python

    from hydrodata import plot

    plot.signatures(clm_loc['Q (cms)'], wshed.drainage_area, prcp=clm_loc['prcp (mm/day)'], title=wshed.name, figsize=(12, 12))
    clm_grd.isel(time=1).tmin.plot(aspect=2, size=8)
    eta_grd.isel(time=4).et.plot(size=8)
    
    ax = wshed.watershed.basin.plot(color='white', edgecolor='black', zorder=1, figsize = (10, 10))
    wshed.tributaries.plot(ax=ax, label='Tributaries', zorder=2)
    wshed.main_channel.plot(ax=ax, color='green', lw=3, label='Main', zorder=3)
    stations.plot(ax=ax, color='black', label='All stations', marker='s', zorder=4)
    stations_upto_150.plot(ax=ax, color='red', label='Stations up to 150 km upstream of main', marker='*', zorder=5)
    ax.legend(loc='best')

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/example_plots.png
        :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/example_plots.png

Installation
------------

To install Hydrodata, run this command in your terminal:

.. code-block:: console

    $ pip install hydrodata


Alternatively, it can be installed from source by first using ``create_env.sh`` script, which generates two environments
using Miniconda; one for installing hydrodata and its dependencies and another for running the `nhdplus.R` script
(for downloading a watershed geometry based on station ID or coordinates).

Before running the ``create_env.sh`` script, ensure Miniconda is installed, this can be accomplished by
running the command ``conda`` on the command line. If Miniconda is not installed it can be downloaded on
continuum_'s site and installed on Linux as follows:

.. _continuum: https://repo.anaconda.com/miniconda/

.. code-block:: console

    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ chmod +x Miniconda3-latest-Linux-x86_64.sh
    $ bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${APP_DIR}/miniconda
    $ rm -f Miniconda3-latest-Linux-x86_64.sh

where ``${APP_DIR}`` is the installation folder.

.. code-block:: console

    $ git clone https://github.com/cheginit/hydrodata.git
    $ cd hydrodata
    $ ./create_env.sh
    $ conda activate hydrodata
    $ python setup.py install


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
