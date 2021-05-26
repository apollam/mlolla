from sklearn.base import TransformerMixin, BaseEstimator


class FillMissing(BaseEstimator, TransformerMixin):
    """Fill nan values on defined columns.

    Parameters
    ----------
    to_fill : dict
        A dict with the value to be imputed on nan values in the column. You can pass a value or you can pass one of the
        functions bellow to imput the value returned from the function: 'max', 'min', 'median', 'mean'.
    """

    def __init__(self, to_fill=None):
        self.to_fill = to_fill
        self._values = dict()

    def fit(self, X, y=None):
        """Saves the value which will be imputed on transform.

        Parameters
        ----------
        X : DataFrame
            Transformed DataFrame
        y : Array-like or DataFrame
            Label column, not used here
        Returns
        -------
        DataFrame
            X
        """

        for key, value in self.to_fill.items():
            for col in X.columns:
                if key in col:
                    if isinstance(value, str):
                        self._values[col] = X[col].agg(value)
                    else:
                        self._values[col] = value

        return self

    def transform(self, X, y=None):
        """Transforms X by applying a value stated on the value variable
        on the nan values.

        Parameters
        ----------
        X : Array-like
            Transformed DataFrame
        y : Array-like or DataFrame
            Label column, not used here
        Returns
        -------
        DataFrame
            X
        """

        X = X.fillna(self._values)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Run fit and transform.

        Parameters
        ----------
        X : DataFrame
            DataFrame to be transformed
        y : Array-like or DataFrame
            Label column, not used here

        Returns
        -------
            Transformed DataFrame
        """

        self.fit(X)
        return self.transform(X)
