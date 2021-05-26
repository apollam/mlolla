import pytest
import pandas as pd


@pytest.fixture(scope='function')
def input_base_data():
    data = {
        'Recebida em': {
            '1-2017-1': '2017-01-01 00:00:00',
            '1-2017-2': '2017-02-01 00:00:00',
            '1-2017-3': '2017-03-01 00:00:00'},
        'desempenho_Admissão': {
            '1-2017-1': '2007-07-16 00:00:00',
            '1-2017-2': '2007-07-16 00:00:00',
            '1-2017-3': '2007-07-16 00:00:00'},
        'Dat_DataNasc': {
            '1-2017-1': '1985-01-01 00:00:00',
            '1-2017-2': '1988-02-01 00:00:00',
            '1-2017-3': '1967-03-01 00:00:00'}
    }
    data = pd.DataFrame(data)
    return data


@pytest.fixture(scope='function')
def expected_data_age():
    data = {
        'desempenho_Admissão': {
            '1-2017-1': '2007-07-16 00:00:00',
            '1-2017-2': '2007-07-16 00:00:00',
            '1-2017-3': '2007-07-16 00:00:00'},
        'time_difference': {
            '1-2017-1': 32.42163661581137,
            '1-2017-2': 29.384188626907072,
            '1-2017-3': 50.66019417475728}
    }
    return pd.DataFrame(data)


@pytest.fixture(scope='function')
def expected_data_tenure():
    data = {
        'Dat_DataNasc': {
            '1-2017-1': '1985-01-01 00:00:00',
            '1-2017-2': '1988-02-01 00:00:00',
            '1-2017-3': '1967-03-01 00:00:00'},
        'time_difference': {
            '1-2017-1': 9.589459084604716,
            '1-2017-2': 9.675450762829403,
            '1-2017-3': 9.753120665742024}
    }
    return pd.DataFrame(data)


@pytest.fixture(scope='function')
def expected_data_tenure_years():
    data = {
        'Dat_DataNasc': {
            '1-2017-1': '1985-01-01 00:00:00',
            '1-2017-2': '1988-02-01 00:00:00',
            '1-2017-3': '1967-03-01 00:00:00'},
        'time_difference': {
            '1-2017-1': 9.589459084604716,
            '1-2017-2': 9.675450762829403,
            '1-2017-3': 9.753120665742024}
    }
    return pd.DataFrame(data)


@pytest.fixture(scope='function')
def expected_data_tenure_days():
    data = {
        'Dat_DataNasc': {
            '1-2017-1': '1985-01-01 00:00:00',
            '1-2017-2': '1988-02-01 00:00:00',
            '1-2017-3': '1967-03-01 00:00:00'},
        'time_difference': {
            '1-2017-1': 3457.0,
            '1-2017-2': 3488.0,
            '1-2017-3': 3516.0}
    }
    return pd.DataFrame(data)


@pytest.fixture(scope='function')
def input_base_data_format():
    """Passing a different date format 2017-31-01."""

    data = {
        'Recebida em': {
            '1-2017-1': '01-01-2017 00:00:00',
            '1-2017-2': '01-02-2017 00:00:00',
            '1-2017-3': '01-03-2017 00:00:00'},
        'desempenho_Admissão': {
            '1-2017-1': '16-07-2007 00:00:00',
            '1-2017-2': '16-07-2007 00:00:00',
            '1-2017-3': '16-07-2007 00:00:00'},
        'Dat_DataNasc': {
            '1-2017-1': '01-01-1985 00:00:00',
            '1-2017-2': '01-02-1988 00:00:00',
            '1-2017-3': '01-03-1967 00:00:00'}
    }
    data = pd.DataFrame(data)
    return data


@pytest.fixture(scope='function')
def expected_data_tenure_days_format():
    data = \
        {'Dat_DataNasc': {'1-2017-1': '01-01-1985 00:00:00', '1-2017-2': '01-02-1988 00:00:00',
                          '1-2017-3': '01-03-1967 00:00:00'},
         'time_difference': {'1-2017-1': 3457.0, '1-2017-2': 3488.0, '1-2017-3': 3516.0}}

    return pd.DataFrame(data)
