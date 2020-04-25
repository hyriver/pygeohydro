.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :target: https://raw.githubusercontent.com/cheginit/hydrodata/develop/docs/_static/hydrodata_logo_text.png
    :align: center

Hydrodata is a portal to access hydrology and climatology data in python and designed to aid in watershed analysis. It provides access to hydrology and climatology databases with some helper functions for visualization. Currently, the following data retrieval services are supported:

* `NLDI <https://labs.waterdata.usgs.gov/about-nldi/>`_ for accessing NHDPlus V2 database
* `Daymet <https://daymet.ornl.gov/>`__ for climatology data, both single pixel and gridded
* `SSEBop <https://earlywarning.usgs.gov/ssebop/modis/daily>`_ for daily actual evapotranspiration, both single pixel and gridded
* `NLCD 2016 <https://www.mrlc.gov/>`__ for land cover, land use (some utilities are available for analysing and plotting the cover data)
* `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ for USGS stations' daily streamflow observations
* `3DEP <https://www.usgs.gov/core-science-systems/ngp/3dep>`_ for Digital Elevation Model

Additionally, the following functionalities are offered:

* Efficient vector-based **flow accumulation** in a stream network
* Computing **Potential Evapotranspiration** (PET) using Daymet data based on `FAO-56 <http://www.fao.org/3/X0490E/X0490E00.htm>`_.
* **Interactive map** for exploring USGS stations within a bounding box
* High level APIs for easy access to all ArcGIS `RESTful <https://en.wikipedia.org/wiki/Representational_state_transfer>`_-based services as well as `WMS <https://en.wikipedia.org/wiki/Web_Map_Service>`_- and `WFS <https://en.wikipedia.org/wiki/Web_Feature_Service>`_-based services.

License
=======

HydroData is licensed under `MIT`__.

__ https://github.com/cheginit/hydrodata/blob/master/LICENSE
