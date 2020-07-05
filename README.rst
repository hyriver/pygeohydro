.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :align: center

|

.. image:: https://img.shields.io/pypi/v/hydrodata.svg
    :target: https://pypi.python.org/pypi/hydrodata
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/hydrodata.svg
    :target: https://anaconda.org/conda-forge/hydrodata
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/hydrodata/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/hydrodata
    :alt: CodeCov

.. image:: https://github.com/cheginit/hydrodata/workflows/build/badge.svg
    :target: https://github.com/cheginit/hydrodata/actions?query=workflow%3Abuild
    :alt: Github Actions

.. image:: https://readthedocs.org/projects/hydrodata/badge/?version=latest
    :target: https://hydrodata.readthedocs.io/en/latest/?badge=latest
    :alt: ReadTheDocs

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/hydrodata/develop
    :alt: Binder

|

.. image:: https://pepy.tech/badge/hydrodata
    :target: https://pepy.tech/project/hydrodata
    :alt: Downloads

.. image:: https://www.codefactor.io/repository/github/cheginit/hydrodata/badge/develop
    :target: https://www.codefactor.io/repository/github/cheginit/hydrodata/overview/develop
    :alt: CodeFactor

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

.. image:: https://zenodo.org/badge/237573928.svg
    :target: https://zenodo.org/badge/latestdoi/237573928
    :alt: Zenodo

|

Features
--------

Hydrodata is a python library designed to aid in watershed analysis. It provides easy and consistent access to a handful of hydrology and climatology databases with some helper functions for visualization. Currently, the following data retrieval services are supported:

* `Daymet <https://daymet.ornl.gov/>`__ for climatology data, both single pixel and gridded,
* `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ for NHDPlus V2 indexing data,
* `WaterData GeoServer <https://labs.waterdata.usgs.gov/geoserver/web/wicket/bookmarkable/org.geoserver.web.demo.MapPreviewPage?1>`__ for catchments, HUC8, HUC12, GagesII, NHDPlus V2 flowlines, and water bodies,
* `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ for daily streamflow observations,
* `HCDN 2009 <https://www2.usgs.gov/science/cite-view.php?cite=2932>`_ for identifying sites where human activity affects the natural flow of the watercourse,
* `NLCD 2016 <https://www.mrlc.gov/>`__ for land cover, land use (some utilities are available for analysing and plotting the cover data),
* `3DEP <https://www.usgs.gov/core-science-systems/ngp/3dep>`_ from National Map service for getting data such as Digital Elevation Model, slope, and aspect,
* `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`_ for daily actual evapotranspiration, both single pixel and gridded.

Additionally, the following functionalities are offered:

* **Interactive map** for exploring USGS stations within a bounding box,
* Efficient vector-based **flow accumulation** in a stream network,
* Computing **Potential Evapotranspiration** (PET) using Daymet climate data based on `FAO-56 <http://www.fao.org/3/X0490E/X0490E00.htm>`_,
* High level APIs for easy access to any ArcGIS `RESTful <https://en.wikipedia.org/wiki/Representational_state_transfer>`_-based services as well as `WMS <https://en.wikipedia.org/wiki/Web_Map_Service>`_- and `WFS <https://en.wikipedia.org/wiki/Web_Feature_Service>`_-based services,
* Helpers for plotting land cover data based on **official NLCD cover legends**,
* A **roughness coefficients** lookup table for each land cover type which is useful for overland flow routing among other applications.

You can try using Hydrodata without installing it by clicking on the binder badge below the Hydrodata banner. A Jupyter notebook instance with Hydrodata installed, will be launched on yout web browser. Then, you can check out ``docs/usage.ipynb`` and ``docs/quickguide.ipynb`` notebooks or create a new one and start coding!

Moreover, requests for additional databases or functionalities can be submitted via `issue tracker <https://github.com/cheginit/hydrodata/issues>`_.

Documentation
-------------

Learn more about Hydrodata in its official documentation at https://hydrodata.readthedocs.io.


Installation
------------

You can install Hydrodata using ``pip`` after installing ``libgdal`` on your system (for example ``libgdal-dev`` in Ubuntu):

.. code-block:: console

    $ pip install hydrodata

Alternatively, it can be installed from ``conda-forge`` using `Conda <https://docs.conda.io/en/latest/>`_:

.. code-block:: console

    $ conda install -c conda-forge hydrodata

Quickstart
----------

With just a few lines of code, Hydrodata provides easy access to a handful of databases. We can start by exploring the available USGS stations within a bounding box:

.. code-block:: python

    import hydrodata.datasets as hds

    hds.interactive_map((-70, 44, -69, 46))

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/interactive_map.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/interactive_map.png
    :align: center


Then, we can either specify a station ID or coordinates to the ``Station`` function and gathers the USGS site information such as name, contributing drainage area, and watershed geometry.

.. code-block:: python

    from hydrodata import Station

    dates = ("2000-01-01", "2010-01-21")
    wshed = Station(coords=(-69.32, 45.17), dates=dates)

The generated ``wshed`` object has a property that shows whether the station is in HCDN database i.e., whether it's a natural watershed or is affected by human activity. For this watershed ``wshed.hcdn`` is ``True``, therefore, this is a natural watershed. Moreover, using the retrieved information, ``datasets`` module provides access to other databases within the watershed geometry. For example, we can get the main river channel and its tributaries, the USGS stations upstream (or downstream) of the main river channel (or the tributatires) up to a certain distance, say 150 km or all the stations:

.. code-block:: python

    tributaries = wshed.flowlines()
    main_channel = wshed.flowlines(navigation="upstreamMain")
    catchments = wshed.catchments()
    stations = wshed.nwis_stations(navigation="upstreamMain", distance=150)

For demonstrating the flow accumulation function, lets assume the flow in each river segment is equal to its length. Therefore, it should produce the same results as the ``arbolatesu`` variable in the NHDPlus database.

.. code-block:: python

    from hydrodata import utils

    flw = utils.prepare_nhdplus(tributaries, 0, 0, purge_non_dendritic=False)


    def routing(qin, q):
        return qin + q


    acc = utils.vector_accumulation(
        flw[["comid", "tocomid", "lengthkm"]], routing, "lengthkm", ["lengthkm"]
    )
    flw = flw.merge(acc, on="comid")
    diff = flw.arbolatesu - flw.acc

We can check the validity of the results using ``diff.abs().sum() = 5e-14``. Furthermore, DEM, slope, and aspect can be retrieved for the station's contributing watershed at 30 arc-second (~1 km) resolution:

.. code-block:: python

    from hydrodata import NationalMap

    nm = NationalMap(wshed.geometry, resolution=30)
    dem, slope, aspect = nm.get_dem(), nm.get_slope(), nm.get_aspect()

The point-based climate data and streamflow observations can be retrieved as well. Note the use of ``pet`` flag for computing PET:

.. code-block:: python

    variables = ["tmin", "tmax", "prcp"]
    clm_p = hds.daymet_byloc(wshed.coords, dates=dates, variables=variables, pet=True)
    clm_p["Q (cms)"] = hds.nwis_streamflow(wshed.station_id, dates)

In addition to point-based data, we can get gridded data. The retrieved data are masked with the watershed geometry:

.. code-block:: python

    dates = ("2005-01-01", "2005-01-31")
    clm_g = hds.daymet_bygeom(
        wshed.geometry, dates=dates, variables=variables, pet=True
    )
    eta_g = hds.ssebopeta_bygeom(wshed.geometry, dates=dates)

All the gridded data are returned as `xarray <https://xarray.pydata.org/en/stable/>`_ ``Dataset`` (or ``DataArray``) that offers efficient data processing tools. Some example plots are shown below:

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots.png

Additionally, Hydrodata has a ``plot`` module that plots five hydrologic signatures graphs in one plot:

.. code-block:: python

    from hydrodata import plot

    plot.signatures(clm_p["Q (cms)"], precipitation=clm_p["prcp (mm/day)"])

The ``services`` module can be used to access some other web services as well. For example, we can access `Watershed Boundary Dataset <https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer>`_ via RESTful service, `3D Eleveation Program <https://www.usgs.gov/core-science-systems/ngp/3dep>`_ from WMS and `FEMA National Flood Hazard Layer <https://www.fema.gov/national-flood-hazard-layer-nfhl>`_ via WFS as follows:

.. code-block:: python

    from hydrodata import ArcGISREST, WFS, services

    la_wshed = Station(station_id="11092450")

    wbd8 = ArcGISREST(base_url="https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4")
    wbd8.get_featureids(la_wshed.geometry)
    huc8 = wbd8.get_features()

    url_wms = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"
    hillshade = services.wms_bygeom(
        url_wms,
        geometry=wshed.geometry,
        version="1.3.0",
        layers={"aspect": "3DEPElevation:GreyHillshade_elevationFill"},
        outFormat="image/tiff",
        resolution=1,
    )

    url_wfs = (
        "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"
    )
    wfs = WFS(
        url_wfs,
        layer="public_NFHL:Base_Flood_Elevations",
        outFormat="esrigeojson",
        crs="epsg:4269",
    )
    r = wfs.getfeature_bybox(la_wshed.geometry.bounds, box_crs="epsg:4326")
    flood = utils.json_togeodf(r.json(), "epsg:4269", "epsg:4326")

Contributing
------------

Hydrodata offers some limited statistical analysis. It could be more useful to the watershed modeling community to integrate more data exploratory capabilities to the package. Additionally, adding support for more databases such as water quality, phenology, and water level, are very welcome. If you are interested please get in touch. You can find information about contributing to hydrodata at our `Contributing page <https://hydrodata.readthedocs.io/en/latest/contributing.html>`_.

Credits
-------

This package was created based on the `audreyr/cookiecutter-pypackage`_ project template.

.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
