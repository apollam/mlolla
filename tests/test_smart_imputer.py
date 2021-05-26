from mlolla.data.transformers import SmartImputer
import pandas as pd


def test_smart_imputer(input, expected):
    transformer = SmartImputer()
    original = transformer.fit_transform(input)
    pd.testing.assert_frame_equal(original, expected)
