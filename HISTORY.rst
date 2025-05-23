=======
History
=======

0.19.4 (2025-05-23)
-------------------

Bug Fixes
~~~~~~~~~
- Fix the download URL for ``get_camels`` function according to the changes
  in the Hydroshare web service.
- Use the new links for the eHydro web service.
- Fix the download issue with the Census data in ``get_us_states`` function.

0.19.3 (2025-03-07)
-------------------

New Features
~~~~~~~~~~~~
- Add support for POLARIS soil dataset. The new function is called ``soil_polaris``.
  The function returns soil properties from the POLARIS dataset for a given location.
  The dataset includes soil properties such as soil texture, bulk density, organic
  carbon, pH, and soil moisture. The dataset is available for CONUS at 30m resolution.

Internal Changes
~~~~~~~~~~~~~~~~
- Make ``matplotlib`` and ``folium`` optional dependencies instead
  of required dependencies. This is to reduce the size of the package
  and make it more lightweight. They are now required only if the
  ``plot`` module is used.
- Move the plotting functionality of PyGeoHydro for hydrologic signatures
  to HydroSignatures package. For now, the plot module is exported from
  HydroSignatures package to maintain backward compatibility.

0.19.0 (2025-01-17)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Update all dependencies on HyRiver libraries to the latest versions
  and modify the code to be compatible with the latest versions of
  the libraries.

0.18.0 (2024-10-05)
-------------------

Bug Fixes
~~~~~~~~~
- Bump the minimum version of ``aiohttp-client-cache>=0.12.3`` to fix an
  issue with the latest version of ``aiohttp``. (:issue_hydro:`124`)

Breaking Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.8 since its end-of-life date is October 2024.
- Remove all exceptions from the main module and raise them from the
  ``exceptions`` module. This is to declutter the public API and make
  it easier to maintain.

0.17.1 (2024-09-14)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.8 since its end-of-life date is October 2024.

Bug Fixes
~~~~~~~~~
- Update the ``nlcd`` module to reflect the changes in the MRLC web service.
  There have been some breaking changes in the NLCD web service, and the module
  is updated to reflect these changes. Thus, previous versions of the module
  will not work with the new NLCD web service. (:issue_hydro:`122`)
- Update the ``nid`` module based on the latest changes to the NID web service.
  The changes include the addition of new fields to the NID dataset and the
  removal of some fields. The module is updated to reflect these changes.
- Update the ``nfhl`` module to reflect the changes in the NFHL web service.
  There have been some breaking changes in the NFHL web service, and the module
  is updated to reflect these changes. Thus, previous versions of the module
  will not work with the new NFHL web service.

0.17.0 (2024-05-07)
-------------------

New Features
~~~~~~~~~~~~
- Add support for the National Levee Dataset (NLD) from the USACE. The new
  class is called ``NLD`` and gives users the ability to subset the NLD
  dataset by geometry, ID, or SQL queries. The class has three methods:
  ``bygeom``, ``byids``, and ``bysql``.

Enhancements
~~~~~~~~~~~~
- Add a new argument to ``EHydro`` for passing a directory to store the
  raw downloaded data. This is useful since most times the raw data is
  needed for further processing and reuse. So, by storing them in a folder
  other than its previous default location, i.e., ``./cache``, users can
  easily access and manage them.

Internal Changes
~~~~~~~~~~~~~~~~
- Add the ``exceptions`` module to the high-level API to declutter
  the main module. In the future, all exceptions will be raised from
  this module and not from the main module. For now, the exceptions
  are raised from both modules for backward compatibility.
- Switch to using the ``src`` layout instead of the ``flat`` layout
  for the package structure. This is to make the package more
  maintainable and to avoid any potential conflicts with other
  packages.
- Add artifact attestations to the release workflow.
- Move ``NID`` class to the ``nid`` module to make the package more
  organized and the main module less cluttered.

0.16.5 (2024-05-26)
-------------------

New Features
~~~~~~~~~~~~
- Add new function called ``soil_soilgrids`` to get soil data from the
  SoilGrids dataset. The signature of the function is the same as of the
  ``soil_gnatsgo`` function, so they can be used interchangeably.
  For more information on the SoilGrids dataset, visit
  `ISRIC <https://www.isric.org/explore/soilgrids/faq-soilgrids#What_do_the_filename_codes_mean>`__.

0.16.4 (2024-05-20)
-------------------

Bug Fixes
~~~~~~~~~
- Fix an issue in ``NID.stage_nid_inventory`` where the function was failing
  when the response status code was 206 (partial content). This issue is fixed
  by checking the response status code and if it's 206, the function will continue
  reading the headers and the get the modified date from the response headers.
  Also, the function incorrectly didn't check if the local database was up-to-date
  with the remote database when the processed database already existed. Now, the
  function will check changes in the remote database and re-download the data even if
  necessary even if the processed database exists.

0.16.3 (2024-05-16)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- More robust handling of failed download links for eHydro data.
  For example, sometimes, eHydro web service uses placeholder as actual
  links. There are also cases where links are in the database but they
  are dead.
- Add the ``exceptions`` module to the high-level API to declutter
  the main module. In the future, all exceptions will be raised from
  this module and not from the main module. For now, the exceptions
  are raised from both modules for backward compatibility.

Bug Fixes
~~~~~~~~~
- In ``EHydro`` class, sometimes the requested surveys are not in the same CRS,
  so they couldn't be combined into a single ``GeoDataFrame``. This issue is fixed
  by reprojecting all the requested surveys to 5070 CRS before combining them.

0.16.1 (2024-04-24)
-------------------

Bug Fixes
~~~~~~~~~
- In ``nlcd_helper`` function the roughness value for class 82 was set to 0.16
  instead of 0.037.

New Features
~~~~~~~~~~~~
- Converted all methods of ``NWIS`` class to ``classmethod`` so the class can be used
  without instantiating it. This change makes the class more flexible and easier to use.
- In ``NID`` class, the ``stage_nid_inventory`` method now checks if the remote NID
  database has been modified since the last download and only downloads the new data
  if it has been modified. This change makes the method more efficient and reduces the
  network traffic while ensuring that the local database is always up-to-date.

0.16.0 (2024-01-03)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Bump the minimum supported version of ``shapely`` to 2.

Internal Changes
~~~~~~~~~~~~~~~~
- Update the link to NWIS error codes tables in the ``nwis_errors`` function.
- Update ``NWIS`` class based on the latest changes to the NWIS web service.
- Use the default tiles for the ``interactive_map`` function.

0.15.2 (2023-09-22)
-------------------

New Features
~~~~~~~~~~~~
- Add a new attribute to ``EHydro`` class called ``survey_grid``.
  It's a ``geopandas.GeoDataFrame`` that includes the survey grid
  of the eHydro dataset which is a 35-km hexagonal grid.
- Add support for getting point cloud and survey outline data from
  eHydro. You can set ``data_type`` in ``EHydro`` to ``bathymetry``,
  ``points``, ``outlines``, or ``contours`` to get the corresponding
  data. The default is ``points`` since this is the recommended data
  type by USACE.
- Add ``NFHL`` class within ``nfhl`` module to access FEMA's
  National Flood Hazard Layer (NFHL) using six different ArcGISRESTFul
  services. Contributed by
  `Fernando Aristizabal <https://github.com/fernando-aristizabal>`__.
  (:pull_hydro:`108`)

Internal Changes
~~~~~~~~~~~~~~~~
- Remove dependency on ``dask``.
- Move all NLCD related functions to a separate module called ``nlcd``.
  This doesn't affect the API since the functions are still available
  under ``pygeohydro`` namespace.

0.15.1 (2023-08-02)
-------------------
This release provides access to three new datasets:

- USACE Hydrographic Surveys (eHydro) and
- USGS Short-Term Network (STN) Flood Event Data,
  contributed by `Fernando Aristizabal <https://github.com/fernando-aristizabal>`__.
  (:pull_hydro:`108`)
- NLCD 2021

New Features
~~~~~~~~~~~~
- Add support for getting topobathymetry data from USACE Hydrographic
  Surveys (eHydro). The new class is called ``EHydro`` and gives users
  the ability to subset the eHydro dataset by geometry, ID, or SQL queries.
- Add new ``stnfloodevents`` module with ``STNFloodEventData`` class for
  retrieving flood event data from the
  `USGS Short-Term Network (STN) <https://stn.wim.usgs.gov/STNWeb/#/>`__
  RESTful Service. This Python API abstracts away RESTful principles and
  produces analysis ready data in geo-referenced GeoDataFrames, DataFrames,
  lists, or dictionaries as desired. The core class methods available are
  ``data_dictionary``, ``get_all_data``, and ``get_filtered_data``.
  These class methods retrieve the data dictionaries by type, get all the
  available data by type, and make filtered requests for data by type as well,
  respectively. The four types of data include ``instruments``, ``peaks``,
  ``hwms``, and ``sites``.
  Contributed by `Fernando Aristizabal <https://github.com/fernando-aristizabal>`__.
- Add a wrapper function for the ``STNFloodEventData`` class called
  ``stn_flood_event``.
- Add support for the new NLCD data (2021) for the three supported layers.

0.15.0 (2023-05-07)
-------------------
From release 0.15 onward, all minor versions of HyRiver packages
will be pinned. This ensures that previous minor versions of HyRiver
packages cannot be installed with later minor releases. For example,
if you have ``py3dep==0.14.x`` installed, you cannot install
``pydaymet==0.15.x``. This is to ensure that the API is
consistent across all minor versions.

New Features
~~~~~~~~~~~~
- Add a new option to ``NWIS.get_info``, called ``nhd_info``, for
  retrieving NHDPlus related info on the sites. This will two new
  service calls that might slow down the function, so it's disabled
  by default.
- Update links in ``NID`` to the latest CSV and GPKG versions of
  the NID dataset.
- Add two new properties to ``NID`` to access the entire NID dataset.
  You can use ``NID.df`` to access the CSV version as a
  ``pandas.DataFrame`` and ``NID.gdf`` to access the GPKG version
  as a ``geopandas.GeoDataFrame``. Installing ``pyogrio`` is highly
  recommended for much faster reading of the GPKG version.
- Refactor ``NID.bygeom`` to use the new ``NID.gdf`` property for
  spatial querying of the dataset. This change should make the query
  much faster.
- For now, retain compatibility with ``shapely<2`` while supporting
  ``shapley>=2``.

0.14.0 (2023-03-05)
-------------------

New Features
~~~~~~~~~~~~
- Add a new function, called ``nlcd_area_percent``, for computing the
  percentages or natural, developed, and impervious areas within geometries
  of a given ``GeoDataFrame``. This function uses imperviousness and land
  use/land cover data from NLCD to compute the area percentages of the natural,
  developed, and impervious areas. For more information please refer to the
  function's documentation.
- Add a new column to the dataframe returned by ``NWIS.get_info``, called
  ``nhd_comid``, and rename ``drain_sqkm`` to ``nhd_areasqkm``. The new
  drainage area is the best available estimates of stations' drainage area
  that have been extracted from the NHDPlus. The new ``nhd_comid`` column
  makes it easier to link stations to NHDPlus.
- In ``get_camels``, return ``qobs`` with negatives values set to ``NaN``.
  Also, Add a new variable called ``Newman_2017`` to both datasets for
  identifying the 531 stations that were used in
  `Newman et al. (2017) <https://doi.org/10.1175/JHM-D-16-0284.1>`__.
- Add a new function, called ``streamflow_fillna``, for filling missing
  streamflow values (``NAN``) with day-of-year average values.

Breaking Changes
~~~~~~~~~~~~~~~~
- Bump the minimum required version of ``shapely`` to 2.0,
  and use its new API.

Internal Changes
~~~~~~~~~~~~~~~~
- Sync all minor versions of HyRiver packages to 0.14.0.
- Improve performance of all NLCD functions by merging two methods of
  the ``NLCD`` and also reducing the memory footprint of the functions.

0.13.12 (2023-02-10)
--------------------

New Features
~~~~~~~~~~~~
- Add initial support for `SensorThings API <https://labs.waterdata.usgs.gov/api-docs/about-sensorthings-api/index.html/>`__
  Currently, the ``SensorThings`` class only supports ``Things`` endpoint.
  Users need to provide a valid Odata filter. The class has a ``odata_helper``
  function that can be used to generate and validate Odata filters.
  Additionally, using ``sensor_info`` and ``sensor_property`` functions
  users can request for information about sensors themselves or their properties.

Internal Changes
~~~~~~~~~~~~~~~~
- Simplify geometry validation by using ``pygeoutils.geo2polygon``
  function in ``ssebopeta_bygeom``.
- Fully migrate ``setup.cfg`` and ``setup.py`` to ``pyproject.toml``.
- Convert relative imports to absolute with ``absolufy-imports``.
- Sync all patch versions of HyRiver packages to x.x.12.

0.13.10 (2023-01-09)
--------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- The NID service has changed some of its endpoints to use Federal ID
  instead of Dam ID. This change affects the ``NID.inventory_byid``
  function. This function now accepts Federal IDs instead of dam IDs.

New Features
~~~~~~~~~~~~
- Refactor the ``show_versions`` function to improve performance and
  print the output in a nicer table-like format.

Internal Changes
~~~~~~~~~~~~~~~~
- Use the new ``pygeoogc.streaming_download`` function in ``huc_wb_full``
  to improve performance and reduce code complexity.
- Skip 0.13.9 version so the minor version of all HyRiver packages become
  the same.
- Modify the codebase based on the latest changes in ``geopandas`` related
  to empty dataframes.
- Use ``pyright`` for static type checking instead of ``mypy`` and address
  all typing issues that it raised.

0.13.8 (2022-12-09)
-------------------

New Features
~~~~~~~~~~~~
- Add a function called ``huc_wb_full`` that returns the full watershed
  boundary ``GeoDataFrame`` of a given HUC level. If only a subset of HUCs
  is needed the ``pygeohydro.WBD`` class should be used. The full dataset
  is downloaded from the National Maps'
  `WBD staged products <https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/WBD/HU2/Shape/>`__.
- Add a new function called ``irrigation_withdrawals`` for retrieving estimated
  monthly water use for irrigation by 12-digit hydrologic unit in the
  CONUS for 2015 from `ScienceBase <https://doi.org/10.5066/P9FDLY8P>`__.
- Add a new property to ``NID``, called ``data_units`` for indicating the
  units of NID dataset variables.
- The ``get_us_states`` now accepts ``conus`` as a ``subset_key`` which is
  equivalent to ``contiguous``.

Internal Changes
~~~~~~~~~~~~~~~~
- Add ``get_us_states`` to ``__init__`` file, so it can be loaded directly,
  e.g., ``gh.get_us_states("TX")``.
- Modify the codebase based on `Refurb <https://github.com/dosisod/refurb>`__
  suggestions.
- Significant performance improvements in ``NWIS.get_streamflow`` especially
  for large requests by refactoring the timezone handling.

Bug Fixes
~~~~~~~~~
- Fix the dam types and purposes mapping dictionaries in ``NID`` class.

0.13.7 (2022-11-04)
-------------------

New Features
~~~~~~~~~~~~
- Add a two new function for retrieving soil properties across the US:

  * ``soil_properties``: Porosity, available water capacity, and field capacity,
  * ``soil_gnatsgo``: Soil properties from the gNATSGO database.

- Add a new help function called ``state_lookup_table`` for getting
  a lookup table of US states and their counties. This can be particularly
  useful for mapping the digit ``state_cd`` and ``county_cd`` that NWIS
  returns to state names/codes.
- Add support for getting individual state geometries using ``get_us_states``
  function by passing their two letter state code. Also, use TIGER 2022
  data for the US states and counties instead of TIGER 2021.

Internal Changes
~~~~~~~~~~~~~~~~
- Remove ``proplot`` as a dependency and use ``matplotlib`` instead.

0.13.6 (2022-08-30)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add the missing PyPi classifiers for the supported Python versions.

0.13.5 (2022-08-29)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Append "Error" to all exception classes for conforming to PEP-8 naming conventions.
- Deprecate ``ssebopeta_byloc`` since it's been replaced with ``ssebopeta_bycoords``
  since version 0.13.0.

Internal Changes
~~~~~~~~~~~~~~~~
- Bump the minimum versions of ``pygeoogc`` and ``pygeoutils`` to 0.13.5 and that of
  ``async-retriever`` to 0.3.5.

0.13.3 (2022-07-31)
-------------------

New Features
~~~~~~~~~~~~
- Add a new argument to ``NID.inventory_byid`` class for staging the entire NID dataset
  prior to inventory queries. There a new public method called ``NID.stage_nid_inventory``
  that can be used to download the entire NID dataset and save it as a ``feather`` file.
  This is useful inventory queries with large number of IDs and is much more efficient
  than querying the NID web service.

Bug Fixes
~~~~~~~~~
- The background value in ``cover_statistics`` function should have been 127 not 0.
  Also, dropped the background value from the return statistics.

0.13.2 (2022-06-14)
-------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Set the minimum supported version of Python to 3.8 since many of the
  dependencies such as ``xarray``, ``pandas``, ``rioxarray`` have dropped support
  for Python 3.7.

Internal Changes
~~~~~~~~~~~~~~~~
- Remove ``USGS`` prefixes from the input station IDs in ``NWIS.get_streamflow``
  function. Also, check if the remaining parts of the IDs are all digits and throw
  an exception if otherwise. Additionally, make sure that IDs have at least 8 chars by
  adding leading zeros (:issue_hydro:`99`).
- Use `micromamba <https://github.com/marketplace/actions/provision-with-micromamba>`__
  for running tests
  and use `nox <https://github.com/marketplace/actions/setup-nox>`__
  for linting in CI.

0.13.1 (2022-06-11)
-------------------

New Features
~~~~~~~~~~~~
- Add a new function called ``get_us_states`` to the ``helpers`` module for obtaining
  a GeoDataFrame of the US states. It has an optional argument for returning the
  ``contiguous`` states, ``continental`` states, ``commonwealths`` states, or
  US ``territories``. The data are retrieved from the Census' Tiger 2021 database.
- In the ``NID`` class keep the ``valid_fields`` property as a ``pandas.Series``
  instead of a ``list``, so it can be searched easier via its ``str`` accessor.

Internal Changes
~~~~~~~~~~~~~~~~
- Refactor the ``plot.signatures`` function to use ``proplot`` instead of ``matplotlib``.
- Improve performance of ``NWIS.get_streamflow`` by not validating the layer name
  when instantiating the ``WaterData`` class. Also, make the function more robust
  by checking if streamflow data is available for each station and throw a warning
  if not.

Bug Fixes
~~~~~~~~~
- Fix an issue in ``NWIS.get_streamflow`` where ``-9999`` values were not being
  filtered out. According to NWIS, these values are reserved for ice-affected
  data. This fix sets these values to ``numpy.nan``.

0.13.0 (2022-04-03)
-------------------

New Features
~~~~~~~~~~~~
- Add a new flag to ``nlcd_*`` functions called ``ssl`` for disabling SSL verification.
- Add a new function called ``get_camels`` for getting the
  `CAMELS <https://ral.ucar.edu/solutions/products/camels>`__ dataset. The function
  returns a ``geopandas.GeoDataFrame`` that includes basin-level attributes
  for all 671 stations in the dataset and a ``xarray.Dataset`` that contains
  streamflow data for all 671 stations and their basin-level attributes.
- Add a new function named ``overland_roughness`` for getting the overland
  roughness values from land cover data.
- Add a new class called ``WBD`` for getting watershed boundary (HUC) data.

.. code-block:: python

    from pygeohydro import WBD

    wbd = WBD("huc4")
    hudson = wbd.byids("huc4", ["0202", "0203"])

Breaking Changes
~~~~~~~~~~~~~~~~
- Remove caching-related arguments from all functions since now they
  can be set globally via three environmental variables:

  * ``HYRIVER_CACHE_NAME``: Path to the caching SQLite database.
  * ``HYRIVER_CACHE_EXPIRE``: Expiration time for cached requests in seconds.
  * ``HYRIVER_CACHE_DISABLE``: Disable reading/writing from/to the cache file.

  You can do this like so:

.. code-block:: python

    import os

    os.environ["HYRIVER_CACHE_NAME"] = "path/to/file.sqlite"
    os.environ["HYRIVER_CACHE_EXPIRE"] = "3600"
    os.environ["HYRIVER_CACHE_DISABLE"] = "true"

Internal Changes
~~~~~~~~~~~~~~~~
- Write ``nodata`` attribute using ``rioxarray`` in ``nlcd_bygeom`` since the
  clipping operation of ``rioxarray`` uses this value as the fill value.


0.12.4 (2022-02-04)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Return a named tuple instead of a ``dict`` of percentages in the
  ``cover_statistics`` function. It makes accessing the values easier.
- Add ``pycln`` as a new ``pre-commit`` hooks for removing unused imports.
- Remove time zone info from the inputs to ``plot.signatures`` to avoid
  issues with the ``matplotlib`` backend.

Bug Fixes
~~~~~~~~~
- Fix an issue in ``plot.signatures`` where the new ``matplotlib``
  version requires a ``numpy`` array instead of a ``pandas.DataFrame``.

0.12.3 (2022-01-15)
-------------------

Bug Fixes
~~~~~~~~~
- Replace no data values of data in ``ssebopeta_bygeom`` with ``np.nan`` before
  converting it to mm/day.
- Fix an inconsistency issue with CRS projection when using UTM in ``nlcd_*``.
  Use ``EPSG:3857`` for all reprojections and get the data from NLCD in the same
  projection. (:issue_hydro:`85`)
- Improve performance of ``nlcd_*`` functions by reducing number of service calls.

Internal Changes
~~~~~~~~~~~~~~~~
- Add type checking with ``typeguard`` and fix type hinting issues raised by
  ``typeguard``.
- Refactor ``show_versions`` to ensure getting correct versions of all
  dependencies.

0.12.2 (2021-12-31)
-------------------

New Features
~~~~~~~~~~~~
- The ``NWIS.get_info`` now returns a ``geopandas.GeoDataFrame`` instead of a
  ``pandas.DataFrame``.

Bug Fixes
~~~~~~~~~
- Fix a bug in ``NWIS.get_streamflow`` where the drainage area might not be
  computed correctly if target stations are not located at the outlet of
  their watersheds.

0.12.1 (2021-12-31)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use the three new ``ar.retrieve_*`` functions instead of the old ``ar.retrieve``
  function to improve type hinting and to make the API more consistent.

Bug Fixes
~~~~~~~~~
- Fix an in issue with ``NWIS.get_streamflow`` where time zone of the data
  was not being correctly determined when it was US specific abbreviations
  such as ``CST``.

0.12.0 (2021-12-27)
-------------------

New Features
~~~~~~~~~~~~
- Add support for getting instantaneous streamflow from NWIS in addition to
  the daily streamflow by adding ``freq`` argument to ``NWIS.get_streamflow``
  that can be either ``iv`` or ``dv``. The default is ``dv`` to retain the previous
  behavior of the function.
- Convert the time zone of the streamflow data to UTC.
- Add attributes of the requested stations as ``attrs`` parameter to the returned
  ``pandas.DataFrame``. (:issue_hydro:`75`)
- Add a new flag to ``NWIS.get_streamflow`` for returning the streamflow as
  ``xarray.Dataset``. This dataset has two dimensions; ``time`` and ``station_id``.
  It has ten variables which includes ``discharge`` and nine other station attributes.
  (:issue_hydro:`75`)
- Add ``drain_sqkm`` from GagesII to ``NWIS.get_info``.
- Show ``drain_sqkm`` in the interactive map generated by ``interactive_map``.
- Add two new functions for getting NLCD data; ``nlcd_bygeom`` and ``nlcd_bycoords``.
  The new ``nlcd_bycoords`` function returns a ``geopandas.GeoDataFrame`` with the NLCD
  layers as columns and input coordinates, which should be a list of ``(lon, lat)`` tuples,
  as the ``geometry`` column. Moreover, The new ``nlcd_bygeom`` function now accepts a
  ``geopandas.GeoDataFrame`` as the input. In this case, it returns a ``dict`` with keys as
  indices of the input ``geopandas.GeoDataFrame``. (:issue_hydro:`80`)
- The previous ``nlcd`` function is being deprecated. For now, it calls ``nlcd_bygeom``
  internally and retains the old behavior. This function will be removed in future versions.

Breaking Changes
~~~~~~~~~~~~~~~~
- The ``ssebop_byloc`` is being deprecated and replaced by ``ssebop_bycoords``.
  The new function accepts a ``pandas.DataFrame`` as input that should include
  three columns: ``id``, ``x``, and ``y``. It returns a ``xarray.Dataset`` with
  two dimensions: ``time`` and ``location_id``. The ``id`` columns from the input
  is used as the ``location_id`` dimension. The ``ssebop_byloc`` function still
  retains the old behavior and will be removed in future versions.
- Set the request caching's expiration time to never expire. Add two flags to all
  functions to control the caching: ``expire_after`` and ``disable_caching``.
- Replace ``NID`` class with the new RESTful-based web service of National Inventory
  of Dams. The new NID service is very different from the old one, so this is considered
  a breaking change.

Internal Changes
~~~~~~~~~~~~~~~~
- Improve exception handling in ``NWIS.get_info`` when NWIS returns an error message
  rather than 500s web service error.
- The ``NWIS.get_streamflow`` function now checks if the site info dataset contains
  any duplicates. Therefore, all the remaining station numbers will be unique. This
  prevents an issue with setting ``attrs`` where duplicate indexes cause an exception
  when being converted to a dict. (:issue_hydro:`75`)
- Add all the missing types so ``mypy --strict`` passes.

0.11.4 (2021-11-24)
-------------------

New Features
~~~~~~~~~~~~
- Add support for the
  `Water Quality Portal <http://www.waterqualitydata.us>`__ Web Services. (:issue_hydro:`72`)
- Add support for two versions of NID web service. The original NID web service is considered
  version 2 and the new NID is considered version 3. You can pass the version number to the
  ``NID`` like so ``NID(2)``. The default version is 2.

Bug Fixes
~~~~~~~~~
- Fix an issue with background percentage calculation in ``cover_statistics``.

0.11.3 (2021-11-12)
-------------------

New Features
~~~~~~~~~~~~
- Add a `new <https://ags03.sec.usace.army.mil/server/rest/services/Dams_Public/MapServer/0>`__
  map service for National Inventory of Dams (NID).

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``importlib-metadata`` for getting the version instead of ``pkg_resources``
  to decrease import time as discussed in this
  `issue <https://github.com/pydata/xarray/issues/5676>`__.

0.11.2 (2021-07-31)
-------------------

Bug Fixes
~~~~~~~~~
- Refactor ``cover_statistics`` to address an issue with wrong category names and also
  improve performance for large datasets by using ``numpy``'s functions.
- Fix an issue with detecting wrong number of stations in ``NWIS.get_streamflow``.
  Also, improve filtering stations that their start/end date don't match the user requested
  interval.

0.11.1 (2021-07-31)
-------------------

The highlight of this release is adding support for NLCD 2019 and significant improvements
in NWIS support.

New Features
~~~~~~~~~~~~
- Add support for the recently released version of NLCD (2019), including the impervious
  descriptor layer. Highlights of the new database are:

    NLCD 2019 now offers land cover for years 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019,
    and impervious surface and impervious descriptor products now updated to match each date
    of land cover. These products update all previously released versions of land cover and
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

  * Remove ``query_*`` methods since it's not convenient to pass them directly as a dictionary.
  * Add a new function called ``get_parameter_codes`` to query parameters and get information
    about them.
  * To decrease complexity of ``get_streamflow`` method add a new private function to handle
    some tasks.
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
  it has a ``attrs`` property that includes descriptions of the database fields with their units.
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
- Add announcement regarding the new name for the software stack, HyRiver.
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
- Replace ``cElementTree`` with ``ElementTree`` since it's been deprecated by ``defusedxml``.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible
  bugs.
- Speed up CI testing by using ``mamba`` and caching.


0.9.2 (2021-03-02)
------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Rename ``hydrodata`` package to ``PyGeoHydro`` for publication on JOSS.
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
- Use the new PyNHD function for getting basins, ``NLDI.get_basisn``.
- Made ``mypy`` checks more strict and added all the missing type annotations.

0.8.0 (2020-12-06)
------------------

- Fixed the issue with WaterData due to the recent changes on the server side.
- Updated the examples based on the latest changes across the stack.
- Add support for multipolygon.
- Remove the ``fill_hole`` argument.
- Fix a warning in ``nlcd`` regarding performing division on ``nan`` values.

0.7.2 (2020-8-18)
-----------------

Enhancements
~~~~~~~~~~~~
- Replaced ``simplejson`` with ``orjson`` to speed-up JSON operations.
- Explicitly sort the time dimension of the ``ssebopeta_bygeom`` function.

Bug Fixes
~~~~~~~~~
- Fix an issue with the ``nlcd`` function where high resolution requests fail.

0.7.1 (2020-8-13)
-----------------

New Features
~~~~~~~~~~~~
- Added a new argument to ``plot.signatures`` for controlling the vertical position of the
  plot title, called ``title_ypos``. This could be useful for multi-line titles.

Bug Fixes
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
  for PyGeoHydro. PyGeoOGC is a leaner package with much fewer dependencies and is suitable for
  people who might only need an interface to web services.
- Unified function names for getting feature by ID and by box.
- Combined ``start`` and ``end`` arguments into a ``tuple`` argument
  called ``dates`` across the code base.
- Rewrote NLDI function and moved most of its ``classmethods`` to ``Station`` so now ``Station``
  class has more cohesion.
- Removed exploratory functionality of ``ArcGISREST``, since it's more convenient
  to do so from a browser. Now, ``base_url`` is a required argument.
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
- Made all the classes available directly from ``PyGeoHydro``.
- Added `CodeFactor <https://www.codefactor.io/>`_ to the Github pipeline and addressed
  some issues that ``CodeFactor`` found.
- Added `Bandit <https://bandit.readthedocs.io/en/latest/>`_ to check the code for
  security issue.
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
- Automatic handling of large WMS requests (more than 8 million pixels i.e., width x height)
- The ``json_togeodf`` function now accepts both a single (Geo)JSON or a list of them
- Refactored ``plot.signatures`` using ``add_gridspec`` for a much cleaner code.

New Features
~~~~~~~~~~~~
- Added access to WaterData's GeoServer databases.
- Added access to the remaining NLDI database (Water Quality Portal and Water Data Exchange).
- Created a Binder for launching a computing environment on the cloud and testing PyGeoHydro.
- Added a URL repository for the supported services called ``ServiceURL``
- Added support for `FEMA <https://hazards.fema.gov/femaportal/wps/portal/NFHLWMS>`_ web services
  for flood maps and `FWS <https://www.fws.gov/wetlands/Data/Web-Map-Services.html>`_ for wetlands.
- Added a new function called ``wms_toxarray`` for converting WMS request responses to
  ``xarray.DataArray`` or ``xarray.Dataset``.

Bug Fixes
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
- Migrate RTD to ``pip`` from ``conda``.
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
- Added threading using ``pqdm`` to speed up the downloads
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
- Added code style enforcement with ``isort``, black, flake8 and pre-commit
- Added a new shiny logo!
- New installation method
- Changed OpenTopography base url to their new server
- Fixed NLCD legend and statistics bug

0.3.0 (2020-02-10)
------------------

- Clipped the obtained NLCD data using the watershed geometry
- Added support for specifying the year for getting NLCD
- Removed direct NHDPlus data download dependency by using StreamStats and USGS APIs
- Renamed ``get_lulc`` function to ``get_nlcd``

0.2.0 (2020-02-09)
------------------

- Simplified import method
- Changed usage from ``rst`` format to ``ipynb``
- Auto-formatting with the black python package
- Change ``docstring`` format based on Sphinx
- Fixed ``pytest`` warnings and changed its working directory
- Added an example notebook with data files
- Added ``docstring`` for all the functions
- Added Module section to the documentation
- Fixed py7zr issue
- Changed 7z extractor from ``pyunpack`` to py7zr
- Fixed some linting issues.

0.1.0 (2020-01-31)
------------------

- First release on PyPI.
