---
title: 'Hydrodata: A portal to hydrology and climatology data through Python'
tags:
  - Python
  - hydrology
  - climate
  - web services
authors:
  - name: Taher Chegini
    orcid: 0000-0002-5430-6000
    affiliation: 1
  - name: Hong-Yi Li
    orcid: 0000-0002-9807-3851
    affiliation: 1
affiliations:
 - name: University of Houston
   index: 1
date: 26 February 2021
bibliography: paper.bib
---

# Summary

Over the last decade, the increasing availability of web services that offer hydrology and
climatology data has facilitated publishing reproducible scientific researches in these fields.
Such web services allow researchers to subset big databases and perform some of the common data
processing operations on the server-side. However, implementing such services increases the
technical complexity of code development as it requires sufficient understanding of their
underlying protocols to generate valid queries and filters. `Hydrodata` tries to bridge this gap
by providing a unified and simple Application Programming Interface (API) to web services that are
based on three of the most commonly used protocols for geospatial-temporal data publication:
REpresentational State Transfer (RESTful), Web Feature Services (WFS), and Web Map Services (WMS).
`Hydrodata` is a software stack and includes the following Python packages:

* [Hydrodata](https://github.com/cheginit/hydrodata): Provides access to NWIS (National Water
  Information System), NID (National Inventory of Dams), HCDN-2009 (Hydro-Climatic Data Network),
  NLCD (National Land Cover Database), and SSEBop (operational Simplified Surface Energy Balance)
  databases. Moreover, it can generate an interactive map for exploring NWIS stations within a
  bounding box, compute categorical statistics of land use/land cover data, and plot five
  hydrologic signature graphs. There are several helper functions one of which  returns a roughness
  coefficients lookup table for each NLCD land cover type. These coefficients can be
  useful for overland flow routing among other applications
* [PyGeoOGC](https://github.com/cheginit/pygeoogc): Generates valid queries for retrieving data
  from supported RESTful-, WMS-, and WFS-based services. Although these web services limit
  the number of features in a single query, under-the-hood, `PyGeoOGC` takes care of breaking down
  a large query into smaller queries according to the service specifications. Additionally, this
  package offers several notable utilities: data re-projection, sending asynchronous data retrieval
  requests, and traversing a JSON (JavaScript Object Notation) object.
* [PyGeoUtils](https://github.com/cheginit/pygeoutils): Converts responses from PyGeoOGC's
  supported web services to geo-dataframes (vector data type) or datasets (raster data type).
  Moreover, for gridded data, it can mask the output dataset based on any given geometry.
* [PyNHD](https://github.com/cheginit/pynhd): Provides the ability to navigate and subset NHDPlus
  (National Hydrography Database), at medium- and high-resolution, using NLDI (Hydro
  Network-Linked Data Index), WaterData, and TNM (The National Map) web services. Additionally,
  it can retrieve over 30 catchment-scale attributes from
  [ScienceBase](https://www.sciencebase.gov/catalog/item/5669a79ee4b08895842a1d47)
  that are linked to the NHDPlus database via Common Identifiers (ComIDs). `PyNHD` has some
  additional river network tools that use NHDPlus data for routing through a river network.
  These tools sort the river network topologically from upstream to downstream, then based on
  a user defined function transport the specified attribute through the network.
* [Py3DEP](https://github.com/cheginit/py3dep): Gives access to topographic data through 3DEP (3D
  Elevation Program) service. There are 12 topographic data that this package can pull from the
  3DEP service such as Digital Elevation Model, slope, aspect, and hillshade.
* [PyDaymet](https://github.com/cheginit/pydaymet): Retrieves daily climate data as well as monthly
  and annual summaries from the Daymet dataset. It is possible to request data for a single
  location as well as a grid (any valid geometrical shape) at 1-km spatial resolution.

Furthermore, `PyGeoOGC` and `PyGeoUtils` are low-level engines of this software stack that the
other four packages utilize for providing access to some of the most popular databases in the
hydrology community. These two low-level packages are generic and developers can use them for
connecting and sending queries to any other web services that are based on the protocols that
`Hydrodata` supports.

# Statement of need

Preparing input data for conducting hydrology and climatology studies is often one of
the most time-consuming steps in such studies. The difficulties for processing such input data stem
from diverse data sources and types as well as their sizes. For example, hydrological modeling
of watersheds might require climate data such as precipitation and temperature, topology
data such as Digital Elevation Model, and river network. Climate and topology data
are available in raster format, and river network could be from a vector data type. Additionally,
these data are available from different sources, for example, we can retrieve climate data
from Daymet [@Thornton_2020], topology data from 3D Elevation Program [@Thatcher_2020],
and river network from National Hydrography Database [@Buto_2020]. The diversity in data sources
and their large size may hinder reproducible publication of such studies. `Hydrodata` software
stack provides access to such databases through plethora of web services that public and private
entities have made available.

There are several open-source packages that offer similar functionalities. For example,
[hydrofunctions](https://github.com/mroberge/hydrofunctions) is a Python package that retrieves
streamflow data from NWIS and [ulmo](https://github.com/ulmo-dev/ulmo) is another Python package
that provides access to several public hydrology and climatology data. `Dataretrieval` gives
access to some of the USGS (United States Geological Survey) databases and has two versions in
[R](https://github.com/USGS-R/dataRetrieval) and [Python](https://github.com/USGS-python/dataretrieval).
There is also a R Package with the same name, [HydroData](https://github.com/mikejohnson51/HydroData),
and provide access to 15 earth system datasets. Although these packages offer similar
functionalities to `Hydrodata`, none of the Python packages offer access to datasets from diverse
sources and post-processing functionalities that `Hydrodata` provides.

# Acknowledgements

We acknowledge contributions from Austin Raney and Emilio Mayorga to this project.

# References
