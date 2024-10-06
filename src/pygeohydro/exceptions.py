"""Customized PyGeoHydro exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import async_retriever.exceptions as ar
import pygeoogc.exceptions as ogc

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


class ServiceError(ar.ServiceError):
    """Exception raised when the requested data is not available on the server.

    Parameters
    ----------
    err : str
        Service error message.
    """


class MissingColumnError(Exception):
    """Exception raised when a required column is missing from a dataframe.

    Parameters
    ----------
    missing : list
        List of missing columns.
    """

    def __init__(self, missing: list[str]) -> None:
        self.message = f"The following columns are missing:\n{', '.join(missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class MissingCRSError(Exception):
    """Exception raised when input GeoDataFrame is missing CRS."""

    def __init__(self) -> None:
        self.message = "The input GeoDataFrame is missing CRS."
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class ServiceUnavailableError(ogc.ServiceUnavailableError):
    """Exception raised when the service is not available.

    Parameters
    ----------
    url : str
        The server url
    """


class DataNotAvailableError(Exception):
    """Exception raised for requested data is not available.

    Parameters
    ----------
    data_name : str
        Data name requested.
    """

    def __init__(self, data_name: str) -> None:
        self.message = f"{data_name.capitalize()} is not available for the requested query."
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class InputValueError(Exception):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    given : str, optional
        The given input, defaults to None.
    """

    def __init__(
        self,
        inp: str,
        valid_inputs: Sequence[str | int] | Generator[str | int, None, None],
        given: str | int | None = None,
    ) -> None:
        if given is None:
            self.message = f"Given {inp} is invalid. Valid options are:\n"
        else:
            self.message = f"Given {inp} ({given}) is invalid. Valid options are:\n"
        self.message += "\n".join(str(i) for i in valid_inputs)
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class InputRangeError(Exception):
    """Exception raised when a function argument is not in the valid range.

    Parameters
    ----------
    database : str
        Data base name.
    rng : tuple
        Tuple of valid range.
    """

    def __init__(self, database: str, rng: tuple[str, str]) -> None:
        self.message = f"{database.capitalize()} is available from {rng[0]} to {rng[1]}."
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class InputTypeError(Exception):
    """Exception raised when a function argument type is invalid.

    Parameters
    ----------
    arg : str
        Name of the function argument
    valid_type : str
        The valid type of the argument
    example : str, optional
        An example of a valid form of the argument, defaults to None.
    """

    def __init__(self, arg: str, valid_type: str, example: str | None = None) -> None:
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class ZeroMatchedError(ValueError):
    """Exception raised when a function argument is missing.

    Parameters
    ----------
    msg : str
        The exception error message
    """

    def __init__(self, msg: str | None = None) -> None:
        if msg is None:
            self.message = "Service returned no features."
        else:
            self.message = f"Service returned no features with the following error message:\n{msg}"
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class DependencyError(Exception):
    """Exception raised when a dependencies are not met.

    Parameters
    ----------
    libraries : tuple
        List of valid inputs
    """

    def __init__(self, func: str, libraries: str | list[str] | Generator[str, None, None]) -> None:
        libraries = [libraries] if isinstance(libraries, str) else libraries
        self.message = f"The following dependencies are missing for running {func}:\n"
        self.message += ", ".join(libraries)
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the error message."""
        return self.message
