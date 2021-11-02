"""Configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pygeohydro namespace for doctest."""
    import pygeohydro as gh

    doctest_namespace["gh"] = gh
