.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :align: center

|

=========== ===========================================================================
Package     Description
=========== ===========================================================================
Hydrodata_  Access NWIS, HCDN 2009, NLCD, and SSEBop databases
PyGeoOGC_   Query data from any ArcGIS RESTful-, WMS-, and WFS-based services
PyGeoUtils_ Convert responses from PyGeoOGC's supported web services to datasets
PyNHD_      Access NLDI and WaterData web services for navigating the NHDPlus database
Py3DEP_     Access topographic data through the 3D Elevation Program (3DEP) web service
PyDaymet_   Access the Daymet database for daily climate data
=========== ===========================================================================

.. _Hydrodata: https://github.com/cheginit/hydrodata
.. _PyGeoOGC: https://github.com/cheginit/pygeoogc
.. _PyGeoUtils: https://github.com/cheginit/pygeoutils
.. _PyNHD: https://github.com/cheginit/pynhd
.. _Py3DEP: https://github.com/cheginit/py3dep
.. _PyDaymet: https://github.com/cheginit/pydaymet

Hydrodata: Portal to hydrology and climatology data
---------------------------------------------------

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

🚨 **This package is under heavy development and breaking changes are likely to happen.** 🚨

Features
--------

Hydrodata is a stack of Python libraries designed to aid in watershed analysis through
web services. Currently, it only includes hydrology and climatology data within the US.
Hydrodata itself has three modules; ``hydrodata``, ``plot``, and ``helpers``.

The ``hydrodata`` module provides access to the following web services:

* `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ for daily mean streamflow observations,
* `HCDN 2009 <https://www2.usgs.gov/science/cite-view.php?cite=2932>`__ for identifying sites
  where human activity affects the natural flow of the watercourse,
* `NLCD 2016 <https://www.mrlc.gov/>`__ for land cover/land use, imperviousness, and canopy data,
* `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`__ for daily actual
  evapotranspiration, for both single pixel and gridded data.

Also, it has two other functions:

* ``interactive_map``: Interactive map for exploring USGS stations within a bounding box.
* ``cover_statistics``: Compute categorical statistics of land use/land cover data.

The ``plot`` module includes two main functions:

* ``signatures``: Plot five hydrologic signature graphs
* ``cover_legends``: The official NLCD land cover legends for plotting a land cover dataset.

The ``helpers`` module include:

* ``nlcd_helper``: A roughness coefficients lookup table for each land cover type which is
  useful for overland flow routing among other applications.
* ``nwis_error``: A dataframe for finding information about NWIS requests' errors.

You can try using Hydrodata without installation it on you system by clicking on the binder badge
below the Hydrodata banner. A Jupyter notebook instance with the Hydrodata software stack
pre-installed will be launched in your web browser and you can start coding!

Moreover, requests for additional databases or functionalities can be submitted via
`issue tracker <https://github.com/cheginit/hydrodata/issues>`__.

Documentation
-------------

Learn more about Hydrodata in its official documentation at https://hydrodata.readthedocs.io.


Installation
------------

You can install Hydrodata using ``pip`` after installing ``libgdal`` on your system
(for example, in Ubuntu run ``sudo apt install libgdal-dev``):

.. code-block:: console

    $ pip install hydrodata

Alternatively, Hydrodata can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge hydrodata

Quickstart
----------

With just a few lines of code, Hydrodata provides easy access to a handful of databases.
We can start by exploring the available USGS stations within a bounding box:

.. code-block:: python

    import hydrodata.datasets as hds

    hds.interactive_map((-70, 44, -69, 46))

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/interactive_map.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/interactive_map.png
    :align: center

Then, we can either specify a station ID or coordinates to the ``Station`` function and
gathers the USGS site information such as name, contributing drainage area,
and watershed geometry.

.. code-block:: python

    from hydrodata import Station

    dates = ("2000-01-01", "2010-01-21")
    wshed = Station(coords=(-69.32, 45.17), dates=dates)

The generated ``wshed`` object has a property that shows whether the station is in
HCDN database i.e., whether it's a natural watershed or is affected by human activity.
For this watershed ``wshed.hcdn`` is ``True``, therefore, this is a natural watershed.
Moreover, using the retrieved information, ``datasets`` module provides access to other
databases within the watershed geometry. For example, we can get the main river channel and its
tributaries, the USGS stations upstream (or downstream) of the main river channel
(or the tributatires) up to a certain distance, say 150 km or all the stations:

.. code-block:: python

    tributaries = wshed.flowlines()
    main_channel = wshed.flowlines(navigation="upstreamMain")
    catchments = wshed.catchments()
    stations = wshed.nwis_stations(navigation="upstreamMain", distance=150)

For demonstrating the flow accumulation function, lets assume the flow in each river segment
is equal to its length. Therefore, it should produce the same results as the ``arbolatesu``
variable in the NHDPlus database.

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

We can check the validity of the results using ``diff.abs().sum() = 5e-14``.
Furthermore, DEM, slope, and aspect can be retrieved for the station's contributing
watershed at 1 km resolution:

.. code-block:: python

    from hydrodata import NationalMap

    nm = NationalMap(wshed.geometry, resolution=1e3)
    dem, slope, aspect = nm.get_dem(), nm.get_slope(), nm.get_aspect()

The point-based climate data and streamflow observations can be retrieved as well.
Note the use of ``pet`` flag for computing PET:

.. code-block:: python

    variables = ["tmin", "tmax", "prcp"]
    clm_p = hds.daymet_byloc(wshed.coords, dates=dates, variables=variables, pet=True)
    clm_p["Q (cms)"] = hds.nwis_streamflow(wshed.station_id, dates)

In addition to point-based data, we can get gridded data. The retrieved data are masked
with the watershed geometry:

.. code-block:: python

    dates = ("2005-01-01", "2005-01-31")
    clm_g = hds.daymet_bygeom(
        wshed.geometry, dates=dates, variables=variables, pet=True
    )
    eta_g = hds.ssebopeta_bygeom(wshed.geometry, dates=dates)

All the gridded data are returned as `xarray <https://xarray.pydata.org/en/stable/>`__
``Dataset`` (or ``DataArray``) that offers efficient data processing tools.
Some example plots are shown below:

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/example_plots.png

Additionally, Hydrodata has a ``plot`` module that plots five hydrologic signatures
graphs in one plot:

.. code-block:: python

    from hydrodata import plot

    plot.signatures(clm_p["Q (cms)"], precipitation=clm_p["prcp (mm/day)"])

The ``pygeoogc`` library can be used to access some other web services as well.
For example, we can access
`Watershed Boundary Dataset <https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer>`__
via RESTful service,
`National Wetlands Inventory <https://www.fws.gov/wetlands/>`__ from WMS, and
`FEMA National Flood Hazard <https://www.fema.gov/national-flood-hazard-layer-nfhl>`__
via WFS. The output for these functions are of type ``requests.Response`` that
can be converted to ``GeoDataFrame`` or ``xarray.Dataset`` using Hydrodata.

.. code-block:: python

    from pygeoogc import ArcGISREST, WFS, wms_bybox, MatchCRS
    from hydrodata import NLDI, utils

    basin_geom = NLDI().getfeature_byid(
        "nwissite",
        "USGS-11092450",
        basin=True
    ).geometry[0]

    rest_url = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4"
    wbd8 = ArcGISRESTful(rest_url)
    wbd8.get_featureids(basin_geom)
    resp = wbd8.get_features()
    huc8 = utils.json_togeodf(resp)

    url_wms = "https://www.fws.gov/wetlands/arcgis/services/Wetlands_Raster/ImageServer/WMSServer"
    layer = "0"
    r_dict = wms_bybox(
        url_wms,
        layer,
        basin_geom.bounds,
        1e3,
        "image/tiff",
        box_crs="epsg:4326",
        crs="epsg:3857",
    )
    geom = MatchCRS.geometry(basin_geom, "epsg:4326", "epsg:3857")
    wetlands = utils.wms_toxarray(r_dict, geom, "epsg:3857")

    url_wfs = "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"

    wfs = WFS(
        url_wfs,
        layer="public_NFHL:Base_Flood_Elevations",
        outformat="esrigeojson",
        crs="epsg:4269",
    )
    bbox = basin_geom.bounds
    bbox = (bbox[1], bbox[0], bbox[3], bbox[2])
    r = wfs.getfeature_bybox(bbox, box_crs="epsg:4326")
    flood = utils.json2geodf(r.json(), "epsg:4269", "epsg:4326")

Contributing
------------

Hydrodata offers some limited analysis tools. It could be more useful for
the watershed modeling community to integrate more data exploratory and analysis
capabilities to the package. Additionally, adding support for more databases such
as water quality, phenology, and water level, are very welcome. If you are interested
please get in touch. You can find information about contributing to hydrodata at our
`Contributing page <https://hydrodata.readthedocs.io/en/latest/contributing.html>`__.

Credits
-------

This package was created based on the `audreyr/cookiecutter-pypackage`__ project template.

__ https://github.com/audreyr/cookiecutter-pypackage
