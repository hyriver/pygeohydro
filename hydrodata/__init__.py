"""Top-level package for Hydrodata."""

from .datasets import NLDI, NationalMap, Station, WaterData
from .exceptions import (
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingInputs,
    MissingItems,
    ServerError,
    ZeroMatched,
)
from .utils import MatchCRS

__author__ = """Taher Chegini"""
__email__ = "cheginit@gmail.com"
__version__ = "0.6.0"
