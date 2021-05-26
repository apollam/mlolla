import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope='function')
def input():
    input = pd.DataFrame({'float_col': {0: 0.5,
                                           1: np.nan,
                                           2: 1.4,
                                           3: np.nan,
                                           4: 1.4,
                                           5: 2.4},
                             'categorical_col': {0: 'AUX ATEND CLIENT',
                                                 1: 'MEDICO AUDITOR',
                                                 2: 'AUX ATEND CLIENT',
                                                 3: np.nan,
                                                 4: 'ENFERMEIRO',
                                                 5: 'AUX ATEND CLIENT'},
                             'bool_col': {0: False,
                                          1: False,
                                          2: False,
                                          3: np.nan,
                                          4: True,
                                          5: np.nan},
                             'int_col': {0: int(1),
                                         1: int(3),
                                         2: int(12),
                                         3: np.nan,
                                         4: int(44),
                                         5: np.nan},
                             'flag_col': {0: 1,
                                          1: np.nan,
                                          2: 0,
                                          3: 1,
                                          4: 1,
                                          5: 0}
                             })

    return input


@pytest.fixture(scope='function')
def expected():
    expected = pd.DataFrame({'float_col': {0: 0.5,
                                           1: 1.4,
                                           2: 1.4,
                                           3: 1.4,
                                           4: 1.4,
                                           5: 2.4},
                             'categorical_col': {0: 'AUX ATEND CLIENT',
                                                 1: 'MEDICO AUDITOR',
                                                 2: 'AUX ATEND CLIENT',
                                                 3: 'AUX ATEND CLIENT',
                                                 4: 'ENFERMEIRO',
                                                 5: 'AUX ATEND CLIENT'},
                             'bool_col': {0: False,
                                          1: False,
                                          2: False,
                                          3: False,
                                          4: True,
                                          5: False},
                             'int_col': {0: 1,
                                         1: 3,
                                         2: 12,
                                         3: 7.5,
                                         4: 44,
                                         5: 7.5},
                             'flag_col': {0: 1.0,
                                          1: 1.0,
                                          2: 0.0,
                                          3: 1.0,
                                          4: 1.0,
                                          5: 0.0}
                             })

    return expected
