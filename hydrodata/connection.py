"""requests wrapper with retry."""

import socket
from typing import Any, Mapping, MutableMapping, Optional, Tuple
from unittest.mock import _patch, patch

from requests import Response, Session
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException, RetryError, Timeout
from urllib3 import Retry


class RetrySession:
    """Configures the passed-in session to retry on failed requests.

    The fails can be due to connection errors, specific HTTP response
    codes and 30X redirections. The original code is taken from:
    https://github.com/bustawin/retry-requests

    Parameters
    ----------
    retries : int, optional
        The number of maximum retries before raising an exception, defaults to 5.
    backoff_factor : float, optional
        A factor used to compute the waiting time between retries, defaults to 0.5.
    status_to_retry : tuple, optional
        A tuple of status codes that trigger the reply behaviour, defaults to (500, 502, 504).
    prefixes : tuple, optional
        The prefixes to consider, defaults to ("http://", "https://")
    """

    def __init__(
        self,
        retries: int = 5,
        backoff_factor: float = 0.5,
        status_to_retry: Tuple[int, ...] = (500, 502, 504),
        prefixes: Tuple[str, ...] = ("http://", "https://"),
    ) -> None:

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
        self.session.hooks = {"response": [lambda r, *args, **kwargs: r.raise_for_status()]}

    def get(self, url: str, payload: Optional[Mapping[str, Any]] = None,) -> Response:
        """Retrieve data from a url by GET and return the Response."""
        try:
            return self.session.get(url, params=payload)
        except (ConnectionError, HTTPError, RequestException, RetryError, Timeout):
            raise

    def post(self, url: str, payload: Optional[MutableMapping[str, Any]] = None,) -> Response:
        """Retrieve data from a url by POST and return the Response."""
        try:
            return self.session.post(url, data=payload)
        except (ConnectionError, HTTPError, RequestException, RetryError, Timeout):
            raise


def onlyIPv4() -> _patch:
    """disable IPv6 and only use IPv4."""

    orig_getaddrinfo = socket.getaddrinfo

    def getaddrinfoIPv4(host, port, family=0, ptype=0, proto=0, flags=0):
        return orig_getaddrinfo(
            host=host, port=port, family=socket.AF_INET, type=ptype, proto=proto, flags=flags,
        )

    return patch("socket.getaddrinfo", side_effect=getaddrinfoIPv4)
