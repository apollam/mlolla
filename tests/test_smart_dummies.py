from mlolla.transformers.smart_dummies import SmartDummies
import pandas as pd


def test_smart_dummies(input, expected):
    transformer = SmartDummies()
    original = transformer.fit_transform(input)
    pd.testing.assert_frame_equal(expected, original)
