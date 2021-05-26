from .context import mlolla  # line needed so won't have import problems
from mlolla.data.transformers.smart_dummies import SmartDummies
import pandas as pd
from .fixtures.smart_dummies_fixtures import input, expected


def test_smart_dummies(input, expected):
    transformer = SmartDummies()
    original = transformer.fit_transform(input)
    pd.testing.assert_frame_equal(expected, original)
