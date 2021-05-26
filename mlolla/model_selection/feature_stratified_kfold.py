from sklearn.model_selection import StratifiedKFold
from mlolla.model.utils.train_utils import get_random_state


class FeatureStratifiedKFold(StratifiedKFold):
    """Creates a cross-validation generator based on sklearn StratifiedKFold for a particular feature in X.

    This will provide train/test indices to split data in train/test sets preserving the percentage of samples
    for each class in y and each feature value in `feature_arr` parameter.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    feature_arr : nd.array, pd.DataFrame
        Selected feature column in X. Eg. X['feature']
    """

    def __init__(self, n_splits=5, shuffle=False, feature_arr=None, random_state=None):
        super(StratifiedKFold, self).__init__(n_splits, shuffle, get_random_state())
        self.feature_arr = feature_arr
        self.random_state = random_state if random_state else get_random_state()

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """

        new_y = self.feature_arr.astype('str') + '_' + y.astype('str')

        return super().split(X, new_y)
