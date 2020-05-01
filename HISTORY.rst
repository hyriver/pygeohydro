=======
History
=======

0.5.2 (2020-05-01)
------------------

* Added threading capability to the flow accumulation function
* Generalized WFS to include both by bbox and by featureID
* Migrate RTD to pip from conda
* Changed HCDN database source to GagesII database
* Increased robustness of functions that need network connections

0.5.0 (2020-04-25)
------------------

* An almost complete re-writing of the code base and not backward-compatible
* New website design
* Added vector accumulation
* Added base classes and function accesing any ArcGIS REST, WMS, WMS service
* Standalone functions for creating datasets from responses and masking the data
* Added threading using pqdm to speed up the downloads
* Interactive map for exploring USGS stations
* Replaced OpenTopography with 3DEP
* Added HCDN database for identifying natural watersheds

0.4.4 (2020-03-12)
------------------

* Added new databases: NLDI, NHDPLus V2, OpenTopography, gridded Daymet, and SSEBop
* The gridded data are returned as xarray DataArrays
* Removed dependecy on StreamStats and replaced it by NLDI
* Improved overal robustness and efficiency of the code
* Not backward comparible
* Added code style enforcement with isort, black, flake8 and pre-commit
* Added a new shiny logo!
* New installation method
* Changed OpenTopography base url to their new server
* Fixed NLCD legend and statistics bug

0.3.0 (2020-02-10)
------------------

* Clipped the obtained NLCD data using the watershed geometry
* Added support for specifying the year for getting NLCD
* Removed direct NHDPlus data download dependency buy using StreamStats and USGS APIs
* Renamed get_lulc function to get_nlcd

0.2.0 (2020-02-09)
------------------

* Simplified import method
* Changed usage from `rst` format to `ipynb`
* Autoo-formatting with the black python package
* Change docstring format based on Sphinx
* Fixed pytest warnings and changed its working directory
* Added an example notebook with datafiles
* Added docstring for all the functions
* Added Module section to the documentation
* Fixed py7zr issue
* Changed 7z extractor from pyunpack to py7zr
* Fixed some linting issues.

0.1.0 (2020-01-31)
------------------

* First release on PyPI.
