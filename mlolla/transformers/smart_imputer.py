import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class SmartImputer(BaseEstimator, TransformerMixin):
    """Imputs values on the missing rows of a DataFrame.

    If the column is categorical, imputs the mode.
    If it's numerical, imputs the median.

    Parameters
    ----------
    no_cols : array
        List of columns not to be imputed
    """

    def __init__(self, no_cols=[]):
        self.values_to_imput_dict = dict()
        self.no_cols = no_cols
        self._cols = []

    def fit(self, X, y=None):
        """Saves the modes and medians of train dataset.

        Parameters
        ----------
        X : Dataframe
            Features dataframe
        y : Dataframe, array
            Label column, not used here

        Returns
        -------
            self
        """
        self._cols = X.columns.tolist()
        self._cols = [e for e in self._cols if e not in self.no_cols]

        for col in self._cols:
            if len(X[col].value_counts()) == 2:
                self.values_to_imput_dict[col] = X[col].mode()[0]

            if X[col].dtype == 'float64' or X[col].dtype == 'int64':
                self.values_to_imput_dict[col] = X[col].median()

            else:
                self.values_to_imput_dict[col] = X[col].mode()[0]

        return self

    def transform(self, X, y=None):
        """Transforms the imputed dataframe by imputing values on the
        missing rows.

        Parameters
        ----------
        X : Dataframe
            Features dataframe
        y : Dataframe, array
            Label column

        Returns
        -------
        Dataframe
            The transformed dataframe
        """

        for col in self._cols:
            X[col] = X[col].fillna(self.values_to_imput_dict[col])

        return pd.DataFrame(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
