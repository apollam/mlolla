from .context import mlolla  # line needed so won't have import problems
from mlolla.data.transformers.smart_imputer import SmartImputer
import pandas as pd
from .fixtures.smart_imputer_fixtures import input, expected


def test_smart_imputer(input, expected):
    transformer = SmartImputer()
    original = transformer.fit_transform(input)
    pd.testing.assert_frame_equal(original, expected)
