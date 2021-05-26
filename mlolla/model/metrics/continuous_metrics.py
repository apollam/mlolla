import numpy as np
import scipy.stats
from sklearn.metrics import median_absolute_error, mean_squared_error, explained_variance_score, r2_score, \
    make_scorer, mean_absolute_error


def _max_error_wp(y_true, y_pred):
    """Wrapper that will return the maximum error."""

    errors = abs(y_true - y_pred)

    return np.max(errors)


def _min_error_wp(y_true, y_pred):
    """Wrapper that will return the minimum error."""

    errors = abs(y_true - y_pred)

    return np.min(errors)


def _std_error_wp(y_true, y_pred):
    """Wrapper that will return the minimum error."""

    errors = abs(y_true - y_pred)

    return np.std(errors)


def _quantile_error_wp(y_true, y_pred, quantile=.75):
    """Wrapper that will return the quantile of the error."""

    errors = abs(y_true - y_pred)

    return np.quantile(errors, [quantile])


def _rmse_wp(y_true, y_pred):
    """Wrapper that will return tne root mean squared error that answer 'How similar, on average,
    are the numbers in true values to predict values?'"""

    return np.sqrt(mean_squared_error(y_true, y_pred))


def _mean_confidence_interval_wp(y_true, y_pred, confidence=0.95):
    """Wrapper that will return the confidence interval with t."""

    errors = abs(y_true - y_pred)
    a = 1.0 * np.array(errors)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    return h


def get_continuous_metrics(labels=None):
    """Get continuous metrics to be use on regression evaluation."""

    metrics = {
        'median_absolute_error': make_scorer(median_absolute_error, greater_is_better=False),
        'mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
        'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
        'explained_variance': make_scorer(explained_variance_score),
        'r2': make_scorer(r2_score),
        'max_error': make_scorer(_max_error_wp, greater_is_better=False),
        'min_error': make_scorer(_min_error_wp, greater_is_better=False),
        'quantile_error_75': make_scorer(_quantile_error_wp, greater_is_better=False),
        'quantile_error_25': make_scorer(_quantile_error_wp, greater_is_better=False, quantile=.25),
        'quantile_error_95': make_scorer(_quantile_error_wp, greater_is_better=False, quantile=.95),
        'rmse': make_scorer(_rmse_wp, greater_is_better=False),
        'mean_confidence_interval': make_scorer(_mean_confidence_interval_wp, greater_is_better=False)
    }
    refit = 'r2'

    return metrics, refit
