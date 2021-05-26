import pytest
import pandas as pd


@pytest.fixture
def input_to_drop():
    data = {
        'A': [None, None, None, 0.2, 0.1],
        'B': [1, 2, 3, 4, 5],
        'B_correlated': [2, 4, 6, 8, 10],
        'C': [2, 4, 1, 2, 3],
        'single': [1, 1, 1, 1, 1]
    }
    data = pd.DataFrame(data)
    return data



