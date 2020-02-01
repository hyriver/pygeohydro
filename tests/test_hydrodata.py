#!/usr/bin/env python

"""Tests for `hydrodata` package."""

import pytest


from hydrodata.hydrodata import Dataloader


@pytest.fixture
def get_data():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    start, end = '2000-01-01', '2015-12-31'
    station_id = '01467087'
    frankford = Dataloader(start, end, station_id=station_id)
    frankford.get_climate()
    frankford.get_lulc()
    
    fishing = Dataloader(start, end, coords=(-76.43, 41.08))
    fishing.get_climate()

    p = 0
    frankford.plot()
    p += 1
    Q_dict = {'Frankford1': frankford.climate['qobs (cms)'],
              'Frankford2': frankford.climate['qobs (cms)']}
    frankford.plot(Q_dict=Q_dict)
    p += 1
    Q_dict = {'Frankford': frankford.climate['qobs (cms)'],
              'Fishing': fishing.climate['qobs (cms)']}
    frankford.plot_discharge(Q_dict=Q_dict)
    p += 1
    
    df = frankford.climate.copy()
    df["pr (mm/day)"], df["ps (mm/day)"] = frankford.separate_snow(
        df["prcp (mm/day)"].values,
        df["tmean (C)"].values,
        tcr=0.0)
    
    return frankford.climate.loc['2010-01-01', 'qobs (cms)'], \
           fishing.climate.loc['2010-01-01', 'qobs (cms)'], \
           df.loc['2010-01-01', 'pr (mm/day)'], \
           p
    
def test_content(get_data):
    """Sample pytest test function with the pytest fixture as an argument."""
    q_id, q_co, pr, p = get_data
    assert abs(q_id - 1.4838) < 1e-5 and \
           abs(q_co - 11.9214) < 1e-5 and \
           abs(pr - 5.0) < 1e-2 and \
           p == 3
