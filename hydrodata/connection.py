#!/usr/bin/env python
"""requests wrapper with retry"""

import socket
from unittest.mock import patch

from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException, RetryError, Timeout
from urllib3 import Retry


class RetrySession:
    """Configures the passed-in session to retry on failed requests.

    The fails can be due to connection errors, specific HTTP response
    codes and 30X redirections. The original code is taken from:
    https://github.com/bustawin/retry-requests
    """

    def __init__(
        self,
        retries=5,
        backoff_factor=0.5,
        status_to_retry=(500, 502, 504),
        prefixes=("http://", "https://"),
    ):
        """Initialize the clss

        Parameters
        ----------
        retries : int
            The number of maximum retries before raising an exception.
        backoff_factor : float
            A factor used to compute the waiting time between retries.
        status_to_retry : tuple of ints
            A tuple of status codes that trigger the reply behaviour.

        Returns
        -------
        requests.Session
            A session object with retry configurations.
        """

        self.session = Session()

        r = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_to_retry,
            method_whitelist=False,
        )
        adapter = HTTPAdapter(max_retries=r)
        for prefix in prefixes:
            self.session.mount(prefix, adapter)
        self.session.hooks = {
            "response": lambda r, *args, **kwargs: r.raise_for_status()
        }

    def get(self, url, payload=None):
        """Retrieve data from a url by GET using a requests session"""
        try:
            return self.session.get(url, params=payload)
        except (ConnectionError, HTTPError, RequestException, RetryError, Timeout):
            raise

    def post(self, url, payload=None):
        """Retrieve data from a url by POST using a requests session"""
        try:
            return self.session.post(url, data=payload)
        except (ConnectionError, HTTPError, RequestException, RetryError, Timeout):
            raise


def onlyIPv4():
    """disable IPv6 and only use IPv4"""

    orig_getaddrinfo = socket.getaddrinfo

    def getaddrinfoIPv4(host, port, family=0, ptype=0, proto=0, flags=0):
        return orig_getaddrinfo(
            host=host,
            port=port,
            family=socket.AF_INET,
            type=ptype,
            proto=proto,
            flags=flags,
        )

    return patch("socket.getaddrinfo", side_effect=getaddrinfoIPv4)
