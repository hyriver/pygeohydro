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

Hydrodata downloads climate data for USGS stations as well as land use, land cover data for the corresponding watershed.


* Free software: MIT license
* Documentation: https://hydrodata.readthedocs.io.


Features
--------

* Download daily climate data from the `Daymet <https://daymet.ornl.gov/>`__ database.
* Download daily streamflow data from the `NWIS <https://nwis.waterdata.usgs.gov/nwis>`__ database.
* Compute potential evapotranspiration using `ETo <https://eto.readthedocs.io/en/latest/>`__ package.
* Download land use, land cover data from `NLCD 2016 <https://www.mrlc.gov/>`__ database.
* Plot hydrological signature graphs.


Installation
------------

To install Hydrodata, run this command in your terminal:

.. code-block:: console

    $ pip install hydrodata


Alternatively, it can be installed from source by first using `create_env.sh` script to generate two environments using Miniconda framework; one for installing hydrodata and its dependencies and one for running the `nhdplus.R` script (for downloading a watershed geometry).

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
