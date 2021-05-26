from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Return a DataFrame without the specified columns.

    Parameters
    ----------
    columns_to_drop: list
        List of columns to drop
    correlation_to_drop: float, dict
        You can pass dictionary using the name of the column as key and some threshold as value,
        for example {'tenure': 0.6}. The transformer will drop all columns that have a Person's correlation more
        than 0.6 with the defined column. If a float was passed it will remove all collinear features up to the
        threshold passed
    missing_threshold: float
        Percentage value which the NaN percentage cannot be greater than.
    drop_uniques_values: Bool
        If True it will  remove columns that have the same value in all lines

    Attributes
    -------
    record_missing: list
        list of missing columns removed
    record_single_unique: list
        list of single unique columns removed
    record_correlation: list
        list of collinear columns removed
    """

    def __init__(self, columns_to_drop=None, correlation_to_drop=None, missing_threshold=None,
                 drop_uniques_values=False):
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self.columns_to_drop = columns_to_drop
        self.correlation_to_drop = correlation_to_drop
        self.missing_threshold = missing_threshold
        self.drop_uniques_values = drop_uniques_values
        self.record_missing = None
        self.record_single_unique = None
        self.record_correlation = None

    def _get_columns_missing(self, data):
        """Find the features with a fraction of missing values above `missing_threshold`"""

        # Calculate the fraction of missing in each column
        missing_series = data.isnull().sum() / data.shape[0]

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > self.missing_threshold]).reset_index().\
            rename(columns={'index': 'feature', 0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        self.record_missing = to_drop
        self.columns_to_drop += to_drop

        print('%d features with greater than %0.2f missing values.\n' % (len(to_drop), self.missing_threshold))

    def _get_columns_correlation(self, data):
        """Get the columns which have high correlation with the defined column."""
        data_corr = data.corr()
        self.record_correlation = self.record_correlation if self.record_correlation else list()
        for column, threshold in self.correlation_to_drop.items():
            to_drop = data_corr[column].loc[data_corr[column] >= threshold].index.tolist()
            to_drop.remove(column)
            self.columns_to_drop += to_drop
            self.record_correlation += to_drop

    def _get_all_collinear(self, data):
        """Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.
        Using code adapted from:
        https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/"""
        self.correlation_to_drop = self.correlation_to_drop
        corr_matrix = data.corr()

        self.corr_matrix = corr_matrix

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > self.correlation_to_drop)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > self.correlation_to_drop])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > self.correlation_to_drop])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index=True)

        self.record_correlation = to_drop
        self.columns_to_drop += to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(to_drop), self.correlation_to_drop))

    def _get_correlation_to_drop(self, data):
        """Redirect to the right function to remove correlation columns"""
        if isinstance(self.correlation_to_drop, dict):
            self._get_columns_correlation(data)
        elif isinstance(self.correlation_to_drop, float):
            self._get_all_collinear(data)

    def _get_single_unique(self, data):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = data.nunique()

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={'index': 'feature', 0: 'nunique'})

        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.columns_to_drop += to_drop

        print('%d features with a single unique value.\n' % len(to_drop))

    def fit(self, X, y=None):
        """On fit, set the global variables self.correlation_to_drop and self.nan_threshold.

        Parameters
        ----------
        X : DataFrame
            DataFrame to be transformed
        y : Array-like or DataFrame
            Label column, not used here

        Returns
        -------
            self
        """

        self.columns_to_drop = self.columns_to_drop if self.columns_to_drop else list()
        if self.correlation_to_drop:
            self._get_correlation_to_drop(X)
        if self.missing_threshold:
            self._get_columns_missing(X)
        if self.drop_uniques_values:
            self._get_single_unique(X)
        return self

    def transform(self, X, y=None):
        """On transform, drop the columns.

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

        X = X.drop(columns=self.columns_to_drop)

        return X

    def fit_transform(self, X, y=None, **kwargs):
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
