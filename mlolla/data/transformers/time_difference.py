from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np


class TimeDifference(BaseEstimator, TransformerMixin):
    """This transformer calculates the time difference by subtracting the date_end and date_start columns.

    Parameters
    ----------
    start_column : str
        Name of the column which contains start date
    end_column: str
        Name of the column which contains end date
    scale_by : str, default 'days'
        Scale fo the results you can use {'days', 'year', 'month', 'week',
        'semester', 'quarters'}
    final_column: str, default 'time_difference'
        Name of the final column
    date_format: str, default '%Y-%m-%dT%H:%M:%S'
         The strftime to parse time, eg '%d/%m/%Y', of the start_column and end_column
    remove_date_columns: bool, default True
         True to drop start_column and end_column
    """

    def __init__(self, start_column='date_start', end_column='date_end', scale_by='days',
                 final_column='time_difference', remove_date_columns=True, date_format='%Y-%m-%d %H:%M:%S'):

        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self.start_column = start_column
        self.end_column = end_column
        self.final_column = final_column
        self.scale_by = scale_by
        self.remove_date_columns = remove_date_columns
        self.data_format = date_format

    def get_feature_names(self):
        """Get the Dataframe transformation feature names.

        Returns
        -------
        Array-like
            An array with all the transformation feature names

        """
        return self.final_column

    def _set_scale(self, by):
        """Function to return a number to scale the results in year, month, weeks, days, etc."""

        if by == 'years':
            return 360.5
        elif by == 'months':
            return 30.5
        elif by == 'weeks':
            return 7
        elif by == 'semesters':
            return 75.5
        elif by == 'quarters':
            return 91.5
        else:
            return 1

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Performs transformation on X by adding a new column containing the tenure and deleting the old date columns.

        Parameters
        ----------
        X : Dataframe
            A dataframe containing resume data with date_end and date_start
            columns
        y : Array-like or Dataframe
            Label column, not used here

        Returns
        -------
        Dataframe
            A Dataframe containing a new tenure column without old date columns
        """

        X[self.end_column] = pd.to_datetime(X[self.end_column], format=self.data_format, utc=True, errors='coerce')
        X[self.start_column] = pd.to_datetime(X[self.start_column], format=self.data_format, utc=True, errors='coerce')

        X[self.final_column] = X[self.end_column] - X[self.start_column]
        X[self.final_column] /= np.timedelta64(1, 'D')
        time_scale = self._set_scale(self.scale_by)
        X[self.final_column] /= time_scale

        if self.remove_date_columns:
            X = X.drop([self.end_column, self.start_column], axis=1)

        return X

    def fit_transform(self, X, y=None, **kwargs):
        X = self.transform(X)

        return X
