#!/usr/bin/env python3

# Source specific imports
import io
import socket
import zipfile
import rasterio as rio
from unittest.mock import patch

# local imports
from meta_source import DataSource

class ssebopeta(DataSource):

    def __init__(self):
        # This is probably not the right url, just as an example
        super().__init__(
            "SSEBop Database",
            "https://earlywarning.usgs.gov/ssebop/modis",
            "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/daily/downloads"
            )

    @classmethod
    def bygeom(
        cls,
        geometry,
        start=None,
        end=None,
        years=None,
        resolution=None,
        fill_holes=False
        ):
        """Gridded data from the SSEBop database.

        Note
        ----
        Since there's still no web service available for subsetting, the data first
        needs to be downloads for the requested period then the data is masked by the
        region interest locally. Therefore, it's not as fast as other functions and
        the bottleneck could be the download speed.

        Parameters
        ----------
        geometry : Geometry
            The geometry for downloading clipping the data. For a box geometry,
            the order should be as follows:
            geom = box(minx, miny, maxx, maxy)
        start : string or datetime
            Starting date
        end : string or datetime
            Ending date
        years : list
            List of years
        fill_holes : bool, optional
            Wether to fill the holes in the geometry's interior, defaults to False.

        Returns
        -------
        xarray.DataArray
            The actual ET for the requested region.
        """

        handled_date_input = super.bygeom(geometry, start=start, end=end, years=years)

        if fill_holes:
            geometry = Polygon(geometry.exterior)

        f_list = [
        (day, f"{cls.__class__._base_url}/det{d.strftime('%Y%j')}.modisSSEBopETactual.zip")
        for day in handled_date_input
        ]

        def _getaddrinfoIPv4(host, port, family=0, type=0, proto=0, flags=0):
            return orig_getaddrinfo(
                host=host,
                port=port,
                family=socket.AF_INET,
                type=type,
                proto=proto,
                flags=flags,
            )

        orig_getaddrinfo = socket.getaddrinfo
        session = utils.retry_requests()

        # disable IPv6 to speedup the download
        with patch("socket.getaddrinfo", side_effect=_getaddrinfoIPv4):
            # find the mask using the first dataset
            date, url = f_list[0]

            res = utils.get_url(session, url)

            _zip_file = zipfile.ZipFile(io.BytesIO(res.content))
            with rio.MemoryFile() as memfile:
                memfile.write(_zip_file.read(_zip_file.filelist[0].filename))

                with memfile.open() as src:
                    ras_msk, _ = rio.mask.mask(src, [geometry])
                    nodata = src.nodata

                    with xr.open_rasterio(src) as ds:
                        ds.data = ras_msk
                        msk = ds < nodata if nodata > 0.0 else ds > nodata
                        ds = ds.where(msk, drop=True)
                        ds = ds.expand_dims(dict(time=[dt]))
                        ds = ds.squeeze("band", drop=True)
                        ds.name = "eta"
                        data = ds * 1e-3

            # apply the mask to the rest of the data and merge
            for dt, url in f_list[1:]:
                res = utils.get_url(session, url)

                _zip_file = zipfile.ZipFile(io.BytesIO(res.content))
                with rio.MemoryFile() as memfile:
                    memfile.write(_zip_file.read(_zip_file.filelist[0].filename))

                    with memfile.open() as src:
                        with xr.open_rasterio(src) as ds:
                            ds = ds.where(msk, drop=True)
                            ds = ds.expand_dims(dict(time=[dt]))
                            ds = ds.squeeze("band", drop=True)
                            ds.name = "eta"
                            data = xr.merge([data, ds * 1e-3])

        data["eta"].attrs["units"] = "mm/day"
        return data

if __name__ == "__main__":
    s = ssebopeta()