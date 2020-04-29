#!/usr/bin/env python
"""The setup script."""


from setuptools import setup
import distutils.text_file
from pathlib import Path


def parse_requirements(req):
    return distutils.text_file.TextFile(Path(__file__).with_name(req)).readlines()

setup(version="0.5.0", install_requires=parse_requirements('requirements.txt'))