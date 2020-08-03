#!/usr/bin/env python
"""The setup script."""
from setuptools import setup  # isort:skip
import distutils.text_file
from pathlib import Path


def parse_requirements(req):
    """Read requirements file and return it as a list."""
    return distutils.text_file.TextFile(Path(__file__).with_name(req)).readlines()


with open(Path(__file__).with_name("README.rst")) as f:
    long_description = f.read()

setup(
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    long_description=long_description,
    long_description_content_type="text/x-rst",
    install_requires=parse_requirements("requirements.txt"),
)
