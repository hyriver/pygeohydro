.. .. image:: https://raw.githubusercontent.com/cheginit/pygeohydro/master/docs/_static/pygeohydro_logo_text.png
..     :target: https://raw.githubusercontent.com/cheginit/pygeohydro/master/docs/_static/pygeohydro_logo_text.png
..     :align: center

.. |

.. |pygeohydro| image:: https://github.com/cheginit/pygeohydro/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeohydro/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoogc| image:: https://github.com/cheginit/pygeoogc/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeoogc/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |pygeoutils| image:: https://github.com/cheginit/pygeoutils/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeoutils/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |pynhd| image:: https://github.com/cheginit/pynhd/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pynhd/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |py3dep| image:: https://github.com/cheginit/py3dep/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/py3dep/actions?query=workflow%3Apytest
    :alt: Github Actions

.. |pydaymet| image:: https://github.com/cheginit/pydaymet/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pydaymet/actions?query=workflow%3Apytest
    :alt: Github Actions

=========== ==================================================================== ============
Package     Description                                                          Status
=========== ==================================================================== ============
PyGeoHydro_ Access NWIS, NID, HCDN 2009, NLCD, and SSEBop databases              |pygeohydro|
PyGeoOGC_   Send queries to any ArcGIS RESTful-, WMS-, and WFS-based services    |pygeoogc|
PyGeoUtils_ Convert responses from PyGeoOGC's supported web services to datasets |pygeoutils|
PyNHD_      Navigate and subset NHDPlus (MR and HR) using web services           |pynhd|
Py3DEP_     Access topographic data through National Map's 3DEP web service      |py3dep|
PyDaymet_   Access Daymet for daily climate data both single pixel and gridded   |pydaymet|
=========== ==================================================================== ============

.. _PyGeoHydro: https://github.com/cheginit/pygeohydro
.. _PyGeoOGC: https://github.com/cheginit/pygeoogc
.. _PyGeoUtils: https://github.com/cheginit/pygeoutils
.. _PyNHD: https://github.com/cheginit/pynhd
.. _Py3DEP: https://github.com/cheginit/py3dep
.. _PyDaymet: https://github.com/cheginit/pydaymet


A Portal To Hydrology And Climatology Data Through Python
=========================================================

.. image:: https://img.shields.io/pypi/v/pygeohydro.svg
    :target: https://pypi.python.org/pypi/pygeohydro
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/pygeohydro.svg
    :target: https://anaconda.org/conda-forge/pygeohydro
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/pygeohydro/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/pygeohydro
    :alt: CodeCov

.. image:: https://readthedocs.org/projects/pygeohydro/badge/?version=latest
    :target: https://pygeohydro.readthedocs.io/en/latest/?badge=latest
    :alt: ReadTheDocs

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/pygeohydro/master?urlpath=lab/tree/docs/examples
    :alt: Binder

|

.. image:: https://pepy.tech/badge/hydrodata
    :target: https://pepy.tech/project/hydrodata
    :alt: Downloads

.. image:: https://www.codefactor.io/repository/github/cheginit/pygeohydro/badge/develop
    :target: https://www.codefactor.io/repository/github/cheginit/pygeohydro/overview/develop
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

**NOTE**

This software stack was formerly named `hydrodata <https://pypi.org/project/hydrodata>`__
and since a R package with the same name already exists, we decided to
renamed the project. Therefore, we renamed ``hydrodata`` to
`pygeohydro <https://pypi.org/project/pygeohydro>`__.
Installing ``hydrodata`` will install ``pygeohydro`` from now on.

Features
--------

This stack of six Python libraries are designed to aid in watershed analysis through
web services. Currently, they only includes hydrology and climatology data within the US.
Some of the major capabilities these packages are:

* Easy access to many web services for subsetting data and returning the requests as masked
  xarrays or GeoDataFrames.
* Splitting large requests into smaller chunks under-the-hood since web services usually limit
  the number of items per request. So the only bottleneck for subsetting the data
  is the local machine memory.
* Navigating and subsetting NHDPlus database (both meduim- and high-resolution) using web services.
* Cleaning up the vector NHDPlus data, fixing some common issues, and computing vector-based
  accumulation through a river network.
* A URL inventory for some of the popular (and tested) web services.
* Some utilities for manipulating the data and visualization.

You can visit `examples <https://pygeohydro.readthedocs.io/en/master/examples.html>`__
webpage to see some example notebooks. You can also try this project without installing
it on you system by clicking on the binder badge below the PyGeoHydro banner. A Jupyter notebook
instance with the PyGeoHydro software stack pre-installed will be launched in your web browser
and you can start coding!

Please note that this project is in early development stages, while the provided
functionaities should be stable, changes in APIs are possible in new releases. But we
appreciate it if you give this project a try and provide feedback. Contributions are most welcome.

Moreover, requests for additional databases and functionalities can be submitted via
`issue tracker <https://github.com/cheginit/pygeohydro/issues>`__.

.. image:: https://raw.githubusercontent.com/cheginit/pygeohydro/master/docs/_static/example_plots.png
    :target: https://raw.githubusercontent.com/cheginit/pygeohydro/master/docs/_static/example_plots.png


Documentation
=============

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    installation
    examples

.. toctree::
    :maxdepth: 1
    :caption: Help & Reference

    history
    modules
    contributing
    authors
    license
