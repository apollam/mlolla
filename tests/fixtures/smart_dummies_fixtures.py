import pytest
import pandas as pd


@pytest.fixture(scope='function')
def input():
    input = pd.DataFrame({
                'id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
                'GRAU_INSTRUCAO': {0: str(1), 1: str(2), 2: str(3), 3: str(4), 4: str(5), 5: str(6)},
                'Score': {0: 5, 1: 2, 2: 2, 3: 4, 4: 3, 5: 7},
                'job_title': {0: 'JobTitle1', 1: 'JobTitle2',
                              2: 'JobTitle1', 3: 'JobTitle1',
                              4: 'JobTitle3', 5: 'JobTitle1'}
                })

    return input


@pytest.fixture(scope='function')
def expected():
    data = {'id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
            'Score': {0: 5, 1: 2, 2: 2, 3: 4, 4: 3, 5: 7},
            'GRAU_INSTRUCAO_1': {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'GRAU_INSTRUCAO_2': {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
            'GRAU_INSTRUCAO_3': {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0},
            'GRAU_INSTRUCAO_4': {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0},
            'GRAU_INSTRUCAO_5': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0},
            'GRAU_INSTRUCAO_6': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
            'job_title_JobTitle1': {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
            'job_title_JobTitle2': {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
            'job_title_JobTitle3': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0}}
    expected = pd.DataFrame(data)

    return expected
