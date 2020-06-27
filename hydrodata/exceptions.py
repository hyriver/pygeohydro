"""Customized Hydrodata exceptions"""


class ServerError(Exception):
    """Exception raised when the requested data is not available on the server.

    Parameters
    ----------
    url : str
        The server url
    """

    def __init__(self, url):
        self.message = f"The requested server is no available in the URL:\n{url}"
        super().__init__(self.message)

    def __str__(self):
        return self.message


class MissingItems(Exception):
    """Exception raised when a required item is missing.

    Parameters
    ----------
    missing : tuple
        The server url
    """

    def __init__(self, missing):
        self.message = (
            "The following items are missing:\n" + f"{', '.join(m for m in missing)}"
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message


class ZeroMatched(ValueError):
    """Exception raised when a function argument is missing"""

    pass


class InvalidInputValue(Exception):
    """Exception raised for invalid input

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """

    def __init__(self, inp, valid_inputs):
        self.message = f"Given {inp} is invalid. Valid {inp}s are:\n" + ", ".join(
            str(i) for i in valid_inputs
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InvalidInputRange(ValueError):
    """Exception raised when a function argument is not in the valid range"""

    pass


class InvalidInputType(Exception):
    """xception raised when a function argument type is invalid

    Parameters
    ----------
    arg : str
        Name of the function argument
    valid_type : str
        The valid type of the argument
    example : str, optional
        An example of a valid form of the argument, defaults to None.
    """

    def __init__(self, arg, valid_type, example=None):
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self):
        return self.message


class MissingInputs(ValueError):
    """Exception raised when there are missing function arguments"""

    pass
