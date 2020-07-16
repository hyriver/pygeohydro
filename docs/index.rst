.. image:: _static/hydrodata_logo_text.png
   :width: 50%
   :alt: hydrodata
   :align: left

|

.. image:: https://img.shields.io/pypi/v/hydrodata.svg
    :target: https://pypi.python.org/pypi/hydrodata
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/hydrodata.svg
    :target: https://anaconda.org/conda-forge/hydrodata
    :alt: Conda Version

.. image:: https://pepy.tech/badge/hydrodata
    :target: https://pepy.tech/project/hydrodata
    :alt: Downloads

.. image:: https://github.com/cheginit/hydrodata/workflows/build/badge.svg
    :target: https://github.com/cheginit/hydrodata/actions?query=workflow%3Abuild
    :alt: Github Actions

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/hydrodata/develop
    :alt: Binder

.. image:: https://zenodo.org/badge/237573928.svg
    :target: https://zenodo.org/badge/latestdoi/237573928
    :alt: Zenodo

|

Hydrodata is a stack of Python libraries designed to aid in watershed analysis. This software
stack includes:

- `PyGeoOGC <https://github.com/cheginit/pygeoogc>`__: For easy access to services that are based on
  ArcGIS `RESTful <https://en.wikipedia.org/wiki/Representational_state_transfer>`__,
  `WMS <https://en.wikipedia.org/wiki/Web_Map_Service>`__, and
  `WFS <https://en.wikipedia.org/wiki/Web_Feature_Service>`__ web services.
- `Py3DEP <https://github.com/cheginit/py3dep>`__: For accessing
  `3DEP <https://www.usgs.gov/core-science-systems/ngp/3dep>`__ from the National Map service
  for getting data such as Digital Elevation Model, slope, and aspect,

Hydrodata itself has two main modules; ``datasets`` and ``plot``. The ``datasets`` module provides
easy and consistent access to a handful of hydrology and climatology databases. The ``plot`` module
includes some helper functions for plotting hydrologic signatures and NLCD cover data.
Currently, the following data retrieval services are supported through the ``datasets`` moduel:

* `Daymet <https://daymet.ornl.gov/>`__ for climatology data, both single pixel and gridded,
* `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ for NHDPlus V2 indexing data,
* `WaterData GeoServer <https://labs.waterdata.usgs.gov/geoserver/web/wicket/bookmarkable/org.geoserver.web.demo.MapPreviewPage?1>`__
  for catchments, HUC8, HUC12, GagesII, NHDPlus V2 flowlines, and water bodies,
* `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ for daily streamflow observations,
* `HCDN 2009 <https://www2.usgs.gov/science/cite-view.php?cite=2932>`__ for identifying sites
  where human activity affects the natural flow of the watercourse,
* `NLCD 2016 <https://www.mrlc.gov/>`__ for land cover, land use (some utilities are available for
  analysing and plotting the cover data),
* `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`__ for daily actual
  evapotranspiration, both single pixel and gridded.

Additionally, the following utilities are available:

* **Interactive map** for exploring USGS stations within a bounding box,
* Efficient vector-based **flow accumulation** in a stream network,
* Computing **Potential Evapotranspiration** (PET) using Daymet climate data based on
  `FAO-56 <http://www.fao.org/3/X0490E/X0490E00.htm>`__,
* Helpers for plotting land cover data based on the **official NLCD cover legends**,
* A **roughness coefficients** lookup table for each land cover type which is useful for
  overland flow routing among other applications.
* Functions for converting the returned responses from the supported webservices to data frames;
  ``json_togeodf`` and ``wms_toxarray``.

You can try using Hydrodata without installation it on you system by clicking on the binder badge
below the Hydrodata banner. A Jupyter notebook instance with Hydrodata
pre-installed will be launched in your web browser and you can start coding!

.. image:: _static/example_plots.png
    :align: center

.. toctree::
    :maxdepth: 1
    :caption: Documentation

    installation
    quickguide
    usage
    modules
    contributing
    authors
    history
    license

Index
=====

:ref:`genindex`

:ref:`modindex`
