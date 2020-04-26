.. image:: _static/hydrodata_logo_text.png
   :width: 50%
   :alt: hydrodata
   :align: left

Hydrodata is a portal to access hydrology and climatology data in python and designed to aid in watershed analysis. It provides access to hydrology and climatology databases with some helper functions for visualization.

Introduction
============

Currently, the following data retrieval services are supported directly:

* `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ and `NHDPlus V2 <https://www.usgs.gov/core-science-systems/ngp/national-hydrography/national-hydrography-dataset?qt-science_support_page_related_con=0#qt-science_support_page_related_con>`_ for vector river network, catchments, and other NHDPlus data.
* `Daymet <https://daymet.ornl.gov/>`__ for climatology data, both single pixel and gridded
* `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`_ for daily actual evapotranspiration, both single pixel and gridded
* `NLCD 2016 <https://www.mrlc.gov/>`__ for land cover, land use (some utilities are available for analysing and plotting the cover data)
* `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ for daily streamflow observations
* `HCDN 2009 <https://www2.usgs.gov/science/cite-view.php?cite=2932>`_ for identifying sites where human activity affects the natural flow of the watercourse
* `3DEP <https://www.usgs.gov/core-science-systems/ngp/3dep>`_ for Digital Elevation Model

Additionally, the following functionalities are offered:

* **Interactive map** for exploring USGS stations within a bounding box,
* Efficient vector-based **flow accumulation** in a stream network,
* Computing **Potential Evapotranspiration** (PET) using Daymet data based on `FAO-56 <http://www.fao.org/3/X0490E/X0490E00.htm>`_,
* High level APIs for easy access to all ArcGIS `RESTful <https://en.wikipedia.org/wiki/Representational_state_transfer>`_-based services as well as `WMS <https://en.wikipedia.org/wiki/Web_Map_Service>`_- and `WFS <https://en.wikipedia.org/wiki/Web_Feature_Service>`_-based services,
* Helpers for plotting land cover data based on **official NLCD cover legends**,
* A **roughness coefficients** lookup table for each land cover type which is useful for overland flow routing.

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
