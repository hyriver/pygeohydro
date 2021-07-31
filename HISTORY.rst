=======
History
=======

0.11.1 (unreleased)
-------------------

The highlight of this release is adding support for NLCD 2019 and singnificant improvements
in NWIS support.

New Features
~~~~~~~~~~~~
- Add support for the recently released version of NLCD (2019), including the impervious
  descriptor layer. Highlights of the new database are:

    NLCD 2019 now offers land cover for years 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019,
    and impervious surface and impervious descriptor products now updated to match each date
    of land cover. These products update all previously released versions of landcover and
    impervious products for CONUS (NLCD 2001, NLCD 2006, NLCD 2011, NLCD 2016) and are not
    directly comparable to previous products. NLCD 2019 land cover and impervious surface product
    versions of previous dates must be downloaded for proper comparison. NLCD 2019 also offers an
    impervious surface descriptor product that identifies the type of each impervious surface pixel.
    This product identifies types of roads, wind tower sites, building locations, and energy
    production sites to allow deeper analysis of developed features.

    -- `MRLC <https://www.mrlc.gov>`__

- Add support for all the supported regions of NLCD database (CONUS, AK, HI, and PR).
- Add support for passing multiple years to the NLCD function, like so ``{"cover": [2016, 2019]}``.
- Add ``plot.descriptor_legends`` function to plot the legend for the impervious descriptor layer.
- New features in ``NWIS`` class are:
  * Remove ``query_*`` methods since it's not convenient to pass
  them directly as a dictionary.
  * Add a new function called ``get_parameter_codes`` to query parameters and get information
  about them.
  * To decrease complexity of ``get_streamflow`` method add a new private function to handle some
  of the tasks.
  * For handling more of NWIS's services make ``retrieve_rdb`` more general.
- Add a new argument called ``nwis_kwds`` to ``interactive_map`` so any NWIS
  specific keywords can be passed for filtering stations.
- Improve exception handling in ``get_info`` method and simplify and improve
  its performance for getting HCDN.

Internal Changes
~~~~~~~~~~~~~~~~
- Migrate to using ``AsyncRetriever`` for handling communications with web services.

0.11.0 (2021-06-19)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.6 since many of the dependencies such as ``xarray`` and ``pandas``
  have done so.
- Remove ``get_nid`` and ``get_nid_codes`` functions since NID now has a ArcGISRESTFul service.

New Features
~~~~~~~~~~~~
- Add a new class called ``NID`` for accessing the recently released National Inventory of Dams
  web service. This service is based on ArcGIS's RESTful service. So now the user just need to
  instantiate the class like so ``NID()`` and with three methods of ``AGRBase`` class, the
  user can retrieve the data. These methods are: ``bygeom``, ``byids``, and ``bysql``. Moreover,
  it has an ``attrs`` property that includes descriptions of the database fields with their units.
- Refactor ``NWIS.get_info`` to be more generic by accepting any valid queries that are
  documented at
  `USGS Site Web Service <https://waterservices.usgs.gov/rest/Site-Service.html#outputDataTypeCd>`__.
- Allow for passing a list of queries to ``NWIS.get_info`` and use ``async_retriever`` that
  significantly improves the network response time.
- Add two new flags to ``interactive_map`` for limiting the stations to those with
  daily values (``dv=True``) and/or instantaneous values (``iv=True``). This function
  now includes a link to stations webpage on USGS website.

Internal Changes
~~~~~~~~~~~~~~~~
- Use persistent caching for all send/receive requests that can significantly improve the
  network response time.
- Explicitly include all the hard dependencies in ``setup.cfg``.
- Refactor ``interactive_map`` and ``NWIS.get_info`` to make them more efficient and reduce
  their code complexity.

0.10.2 (2021-03-27)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add announcement regarding the new name for the softwate stack, HyRiver.
- Improve ``pip`` installation and release workflow.

0.10.1 (2021-03-06)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add ``lxml`` to deps.

0.10.0 (2021-03-06)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- The official first release of PyGeoHydro with a new name and logo.
- Replace cElementTree with ElementTree since it's been deprecated by ``defusedxml``.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible
  bugs.
- Speed up CI testing by using ``mamba`` and caching.


0.9.2 (2021-03-02)
------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Rename ``hydrodata`` package to ``pygeohydro`` for publication on JOSS.
- In ``NWIS.get_info``, drop rows that don't have mean daily discharge data instead of slicing.
- Speed up Github Actions by using ``mamba`` and caching.
- Improve ``pip`` installation by adding ``pyproject.toml``.

New Features
~~~~~~~~~~~~

- Add support for the National Inventory of Dams (NID) via ``get_nid`` function.

0.9.1 (2021-02-22)
------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Fix an issue with ``NWIS.get_info`` method where stations with False values as their
  ``hcdn_2009`` value were returned as ``None`` instead.

0.9.0 (2021-02-14)
------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Bump versions of packages across the stack to the same version.
- Use the new pynhd function for getting basins, ``NLDI.get_basisn``.
- Made ``mypy`` checks more strict and added all the missing type annotations.

0.8.0 (2020-12-06)
------------------

- Fixed the issue with WaterData due to the recent changes on the server side.
- Updated the examples based on the latest changes across the stack.
- Add support for multipolygon.
- Remove the ``fill_hole`` argument.
- Fix a warning in ``nlcd`` regarding performing division on nan values.

0.7.2 (2020-8-18)
-----------------

Enhancements
~~~~~~~~~~~~
- Replaced ``simplejson`` with ``orjson`` to speed-up JSON operations.
- Explicitly sort the time dimension of the ``ssebopeta_bygeom`` function.

Bug fixes
~~~~~~~~~
- Fix an issue with the ``nlcd`` function where high resolution requests fail.

0.7.1 (2020-8-13)
-----------------

New Features
~~~~~~~~~~~~
- Added a new argument to ``plot.signatures`` for controlling the vertical position of the
  plot title, called ``title_ypos``. This could be useful for multi-line titles.

Bug fixes
~~~~~~~~~
- Fixed an issue with the ``nlcd`` function where none layers are not dropped and cause the
  function to fails.

0.7.0 (2020-8-12)
-----------------

This version divides PyGeoHydro into six standalone Python libraries. So many of the changes
listed below belong to the modules and functions that are now a separate package. This decision
was made for reducing the complexity of the code base and allow the users to only install
the packages that they need without having to install all the PyGeoHydro dependencies.

Breaking changes
~~~~~~~~~~~~~~~~
- The ``services`` module is now a separate package called PyGeoOGCC and is set as a requirement
  for PyGeoHydro. PyGeoOGC is a leaner package with much less dependencies and is suitable for
  people who might only need an interface to web services.
- Unified function names for getting feature by ID and by box.
- Combined ``start`` and ``end`` arguments into a ``tuple`` argument
  called ``dates`` across the code base.
- Rewrote NLDI function and moved most of its classmethods to Station
  so now Station class has more cohesion.
- Removed exploratory functionality of ``ArcGISREST``, since it's more convenient
  to do so from a browser. Now, ``base_url`` is a required arguments.
- Renamed ``in_crs`` in ``datasets`` and ``services`` functions to ``geo_crs`` for geometry and
  ``box_crs`` for bounding box inputs.
- Re-wrote the ``signatures`` function from scratch using ``NamedTuple`` to improve readability
  and efficiency. Now, the ``daily`` argument should be just a ``pandas.DataFrame`` or
  ``pandas.Series`` and the column names are used for legends.
- Removed ``utils.geom_mask`` function and replaced it with ``rasterio.mask.mask``.
- Removed ``width`` as an input in functions with raster output since ``resolution`` is almost
  always the preferred way to request for data. This change made the code more readable.
- Renamed two functions: ``ArcGISRESTful`` and ``wms_bybox``. These function now return
  ``requests.Response`` type output.
- ``onlyipv4`` is now a class method in ``RetrySession``.
- The ``plot.signatures`` function now assumes that the input time series are in mm/day.
- Added a flag to ``get_streamflow`` function in the ``NWIS`` class to convert from cms
  to mm/day which is useful for plotting hydrologic signatures using the ``signatures``
  functions.

Enhancements
~~~~~~~~~~~~
- Remove soft requirements from the env files.
- Refactored ``requests`` functions into a single class and a separate file.
- Made all the classes available directly from ``pygeohydro``.
- Added `CodeFactor <https://www.codefactor.io/>`_ to the Github pipeline and addressed
  the some of issues
  that CodeFactor found.
- Added `Bandit <https://bandit.readthedocs.io/en/latest/>`_ to check the code for
  secutiry issue.
- Improved docstrings and documentations.
- Added customized exceptions for better exception handling.
- Added ``pytest`` fixtures to improve the tests speed.
- Refactored ``daymet`` and ``nwis_siteinfo`` functions to reduce code complexity
  and improve readability.
- Major refactoring of the code base while adding type hinting.
- The input geometry (or bounding box) can be provided in any projection
  and the necessary re-projections are done under the hood.
- Refactored the method for getting object IDs in ``ArcGISREST`` class to improve
  robustness and efficiency.
- Refactored ``Daymet`` class to improve readability.
- Add `Deepsource <https://deepsource.io/>`_ for further code quality checking.
- Automatic handling of large WMS requests (more than 8 million pixel i.e., width x height)
- The ``json_togeodf`` function now accepts both a single (Geo)JSON or a list of them
- Refactored ``plot.signatures`` using ``add_gridspec`` for a much cleaner code.

New Features
~~~~~~~~~~~~
- Added access to WaterData's Geoserver databases.
- Added access to the remaining NLDI database (Water Quality Portal and Water Data Exchange).
- Created a Binder for launching a computing environment on the cloud and testing PyGeoHydro.
- Added a URL repository for the supported services called ``ServiceURL``
- Added support for `FEMA <https://hazards.fema.gov/femaportal/wps/portal/NFHLWMS>`_ web services
  for flood maps and `FWS <https://www.fws.gov/wetlands/Data/Web-Map-Services.html>`_ for wetlands.
- Added a new function called ``wms_toxarray`` for converting WMS request responses to
  ``xarray.DataArray`` or ``xarray.Dataset``.

Bug fixes
~~~~~~~~~
- Re-projection issues for function with input geometry.
- Start and end variables not being initialized when coords was used in ``Station``.
- Geometry mask for ``xarray.DataArray``
- WMS output re-projections

0.6.0 (2020-06-23)
------------------

- Refactor requests session
- Improve overall code quality based on CodeFactor suggestions
- Migrate to Github Actions from TravisCI

0.5.5 (2020-06-03)
------------------

- Add to conda-forge
- Remove pqdm and arcgis2geojson dependencies

0.5.3 (2020-06-07)
------------------

- Added threading capability to the flow accumulation function
- Generalized WFS to include both by bbox and by featureID
- Migrate RTD to pip from conda
- Changed HCDN database source to GagesII database
- Increased robustness of functions that need network connections
- Made the flow accumulation output a pandas Series for better handling of time
  series input
- Combined DEM, slope, and aspect in a class called NationalMap.
- Installation from pip installs all the dependencies

0.5.0 (2020-04-25)
------------------

- An almost complete re-writing of the code base and not backward-compatible
- New website design
- Added vector accumulation
- Added base classes and function accessing any ArcGIS REST, WMS, WMS service
- Standalone functions for creating datasets from responses and masking the data
- Added threading using pqdm to speed up the downloads
- Interactive map for exploring USGS stations
- Replaced OpenTopography with 3DEP
- Added HCDN database for identifying natural watersheds

0.4.4 (2020-03-12)
------------------

- Added new databases: NLDI, NHDPLus V2, OpenTopography, gridded Daymet, and SSEBop
- The gridded data are returned as xarray DataArrays
- Removed dependency on StreamStats and replaced it by NLDI
- Improved overall robustness and efficiency of the code
- Not backward comparable
- Added code style enforcement with isort, black, flake8 and pre-commit
- Added a new shiny logo!
- New installation method
- Changed OpenTopography base url to their new server
- Fixed NLCD legend and statistics bug

0.3.0 (2020-02-10)
------------------

- Clipped the obtained NLCD data using the watershed geometry
- Added support for specifying the year for getting NLCD
- Removed direct NHDPlus data download dependency buy using StreamStats and USGS APIs
- Renamed get_lulc function to get_nlcd

0.2.0 (2020-02-09)
------------------

- Simplified import method
- Changed usage from ``rst`` format to ``ipynb``
- Autoo-formatting with the black python package
- Change docstring format based on Sphinx
- Fixed pytest warnings and changed its working directory
- Added an example notebook with datafiles
- Added docstring for all the functions
- Added Module section to the documentation
- Fixed py7zr issue
- Changed 7z extractor from pyunpack to py7zr
- Fixed some linting issues.

0.1.0 (2020-01-31)
------------------

- First release on PyPI.
