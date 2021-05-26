from .context import mlolla  # line needed so won't have import problems
import pandas as pd
from mlolla.data.transformers.time_difference import TimeDifference
from .fixtures.time_difference_fixtures import input_base_data, expected_data_age, \
    expected_data_tenure, expected_data_tenure_days, expected_data_tenure_days_format, \
    input_base_data_format


def test_transform_age(input_base_data, expected_data_age):
    time_diference = TimeDifference(
        start_column='Dat_DataNasc',
        end_column='Recebida em',
        scale_by='years')

    transformed_data = time_diference.transform(input_base_data)
    pd.testing.assert_frame_equal(transformed_data, expected_data_age)


def test_transform_tenure(input_base_data, expected_data_tenure):
    time_diference = TimeDifference(
        start_column='desempenho_Admissão',
        end_column='Recebida em',
        scale_by='years')

    transformed_data = time_diference.transform(input_base_data)
    pd.testing.assert_frame_equal(transformed_data, expected_data_tenure)


def test_transform_tenure_days(input_base_data, expected_data_tenure_days):
    time_diference = TimeDifference(
        start_column='desempenho_Admissão',
        end_column='Recebida em',
        scale_by='days')

    transformed_data = time_diference.transform(input_base_data)
    pd.testing.assert_frame_equal(transformed_data, expected_data_tenure_days)


def test_date_format(input_base_data_format, expected_data_tenure_days_format):
    time_diference = TimeDifference(
        start_column='desempenho_Admissão',
        end_column='Recebida em',
        scale_by='days', date_format='%d-%m-%Y %H:%M:%S')

    transformed_data = time_diference.transform(input_base_data_format)
    pd.testing.assert_frame_equal(transformed_data, expected_data_tenure_days_format)
