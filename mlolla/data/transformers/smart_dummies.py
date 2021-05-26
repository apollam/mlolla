from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import pandas as pd


class SmartDummies(BaseEstimator, TransformerMixin):
    """Applies get_dummies on every categorical column in X."""

    def __init__(self, no_cols=[]):
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self.no_cols = no_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = X.columns.tolist()
        dtypes = X.dtypes
        for i, col in enumerate(cols):
            if dtypes[i] == 'O' and col not in self.no_cols:
                dummies = pd.get_dummies(X[col], prefix=col, dtype='int64')
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(col, axis=1)

        return X

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X)
        return self.transform(X)
