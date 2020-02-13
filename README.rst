Hydrodata
=========


.. image:: https://img.shields.io/pypi/v/hydrodata.svg
        :target: https://pypi.python.org/pypi/hydrodata

.. image:: https://travis-ci.com/cheginit/hydrodata.svg?branch=master
        :target: https://travis-ci.com/cheginit/hydrodata.svg?branch=master

.. image:: https://readthedocs.org/projects/hydrodata/badge/?version=latest
        :target: https://hydrodata.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/237573928.svg
   :target: https://zenodo.org/badge/latestdoi/237573928

Hydrodata is a python library designed to aid in NHDPlus watershed analysis. Hydrodata is capable of downloading,
preprocessing, and visualizing climatological, hydrological, and geographical datasets pertaining to a given watershed.
Supported datasets include: Daymet climate, USGS streamflow, and NLCD data with further additions planned.

* Documentation: https://hydrodata.readthedocs.io

Features
--------

* Download daily climate data from the `Daymet <https://daymet.ornl.gov/>`__ database.
* Download watersheds geometry and characteristics using `StreamStats <https://www.usgs.gov/mission-areas/water-resources/science/streamstats-streamflow-statistics-and-spatial-analysis-tools?qt-science_center_objects=0#qt-science_center_objects>`_ service.
* Compute potential evapotranspiration using `ETo <https://eto.readthedocs.io/en/latest/>`__ package.
* Download land use, land cover, canopy and impervious data from `NLCD 2016 <https://www.mrlc.gov/>`__ database.
* Download daily streamflow data from the `USGS NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ database.
* Plot hydrological signature graphs.

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/Observed_01467087.png
        :target: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/Observed_01467087.png

.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/NLCD.png
        :target: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/NLCD.png

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
