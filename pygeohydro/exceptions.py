"""Customized PyGeoHydro exceptions."""
from typing import Generator, List, Optional, Tuple, Union


class DataNotAvailable(Exception):
    """Exception raised for requested data is not available.

    Parameters
    ----------
    data_name : str
        Data name requested.
    """

    def __init__(self, data_name: str) -> None:
        self.message = f"{data_name.capitalize()} is not available for the query requested."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputValue(Exception):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """

    def __init__(
        self, inp: str, valid_inputs: Union[List[str], Generator[str, None, None]]
    ) -> None:
        self.message = f"Given {inp} is invalid. Valid {inp}s are:\n" + ", ".join(
            str(i) for i in valid_inputs
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputRange(Exception):
    """Exception raised when a function argument is not in the valid range.

    Parameters
    ----------
    database : str
        Data base name.
    rng : tuple
        Tuple of valid range.
    """

    def __init__(self, database: str, rng: Tuple[str, str]) -> None:
        self.message = f"{database.capitalize()} is available from {rng[0]} to {rng[1]}."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputType(Exception):
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

    def __init__(self, arg: str, valid_type: str, example: Optional[str] = None) -> None:
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ZeroMatched(ValueError):
    """Exception raised when a function argument is missing.

    Parameters
    ----------
    msg : str
        The exception error message
    """

    def __init__(self, msg: Optional[str] = None) -> None:
        if msg is None:
            self.message = "Service returned no features."
        else:
            self.message = f"Service returned no features with the following error message:\n{msg}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
