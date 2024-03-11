"""Configuration for pytest."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _add_standard_imports(doctest_namespace):
    """Add pygeohydro namespace for doctest."""
    import pygeohydro as gh

    doctest_namespace["gh"] = gh
