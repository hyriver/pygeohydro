#!/usr/bin/env python
"""The setup script."""


import distutils.text_file
from pathlib import Path

from setuptools import setup


def parse_requirements(req):
    return distutils.text_file.TextFile(Path(__file__).with_name(req)).readlines()


setup(version="0.5.3", install_requires=parse_requirements("requirements.txt"))
