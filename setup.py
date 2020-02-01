#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'ETo>=1.1.0',
    'OWSLib>=0.18.0',
    'pyunpack>=0.1.2',
    'numpy>=1.18.1',
    'rasterstats>=0.14.0',
    'tables>=3.6.1',
    'daymetpy>=1.0.0',
    'numba>=0.48.0',
    'tqdm>=4.42.0',
    'requests>=2.22.0',
    'Shapely>=1.6.4.post2',
    'pandas>=1.0.0',
    'matplotlib>=3.1.1',
    'geopandas>=0.6.1',
    'h5py>=2.10.0',
    'patool>=1.12',
    'lxml>=4.5']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Taher Chegini",
    author_email='cheginit@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Hydrodata downloads climate data for a USGS station as well as land use, land cover data for the corresponding watershed.",
    entry_points={
        'console_scripts': [
            'hydrodata=hydrodata.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='hydrodata',
    name='hydrodata',
    packages=find_packages(include=['hydrodata', 'hydrodata.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/cheginit/hydrodata',
    version='0.1.1',
    zip_safe=False,
)
