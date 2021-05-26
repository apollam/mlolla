"""Fairness Metrics

Metrics taken from AIF360 Library. For information regarding the code, access the link below.
https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/metrics/metrics.py
"""

from aif360.sklearn.metrics.metrics import statistical_parity_difference, disparate_impact_ratio, \
    equal_opportunity_difference, average_odds_difference, average_odds_error, generalized_entropy_error
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd


def _get_biased_ys(y, y_pred, biased_df):
    """Merge the DataFrame containing the sensitive attribute and with y and y_pred and set the
    sensitive attribute as index since this is needed to acquire the fairness metrics."""

    y_pred = pd.DataFrame(y_pred, index=y.index)
    y = pd.merge(y, biased_df['sens_attr'], how='inner', left_index=True,
                 right_index=True, validate='one_to_one').set_index('sens_attr')
    y_pred = pd.merge(y_pred, biased_df['sens_attr'], how='inner', left_index=True,
                      right_index=True, validate='one_to_one').set_index('sens_attr')

    return y, y_pred


def disparate_impact_ratio_wrapper(y, y_pred, biased_df):
    """Wrapper for disparate_impact_ratio AIF360 function. Since this metric is a ratio, AIF360 library suggests this
    calculation to be made over the score.

    Metric meaning
    ---------------
        Comes from the Disparate Impact effect from United States labor law.
        Ratio of selection rates. Proportion of unprivileged with positive outcome divided by proportion of privileged
        with positive outcome. A value greater than 0.8 satisfies the Disparate Impact law and greater than 1 indicates
        bias for the unprivileged.
    """

    y, y_pred = _get_biased_ys(y, y_pred, biased_df)
    ratio = disparate_impact_ratio(y, y_pred)
    eps = np.finfo(float).eps
    ratio_inverse = 1 / ratio if ratio > eps else eps

    return min(ratio, ratio_inverse)


def statistical_parity_difference_wrapper(y, y_pred, biased_df):
    """Wrapper for statistical_parity_difference AIF360 function. Since this metric is a difference, AIF360
    library suggests this calculation to be made over the score.

    Metric meaning
    ---------------
        A relaxed version of Statistical Parity fairness definition.
        Difference in selection rates. Proportion of unprivileged with positive outcome minus proportion of privileged
        with positive outcome. A value close to 0 is expected and less than 0 indicates bias for the unprivileged.
    """

    y, y_pred = _get_biased_ys(y, y_pred, biased_df)
    diff = statistical_parity_difference(y, y_pred)

    return abs(diff)


def equal_opportunity_difference_wrapper(y, y_pred, biased_df):
    """Wrapper for equal_opportunity_difference AIF360 function. Since this metric is a difference, AIF360
    library suggests this calculation to be made over the score.

    Metric meaning
    ---------------
        A relaxed version of Equality of Opportunity fairness definition.
        Returns the difference in recall scores (TPR) between the unprivileged and privileged groups. A value of 0
        indicates equality of opportunity.
    """

    y, y_pred = _get_biased_ys(y, y_pred, biased_df)
    diff = equal_opportunity_difference(y, y_pred)

    return abs(diff)


def average_odds_difference_wrapper(y, y_pred, biased_df):
    """Wrapper for average_odds_difference AIF360 function. Since this metric is a difference, AIF360
    library suggests this calculation to be made over the score.

    Metric meaning
    ---------------
        A relaxed version of Equality of Opportunity fairness definition.
        Returns the average of the difference in FPR (fall-out) and TPR (recall) for the unprivileged and privileged
        groups. A value of 0 indicates equality of odds.
    """

    y, y_pred = _get_biased_ys(y, y_pred, biased_df)
    diff = average_odds_difference(y, y_pred)

    return abs(diff)


def average_odds_error_wrapper(y, y_pred, biased_df):
    """Wrapper for average_odds_error AIF360 function. Since this metric is a ratio, AIF360 library suggests this
    calculation to be made over the score.

    Metric meaning
    ---------------
        A relaxed version of Equality of Opportunity fairness definition.
        Returns the average of the absolute difference in FPR (fall-out) and TPR (recall) for the unprivileged and
        privileged groups.
    """

    y, y_pred = _get_biased_ys(y, y_pred, biased_df)
    ratio = average_odds_error(y, y_pred)
    eps = np.finfo(float).eps
    ratio_inverse = 1 / ratio if ratio > eps else eps

    return min(ratio, ratio_inverse)


def get_fairness_metrics(biased_df):
    """Instantiate the default metrics to be used on fairness evaluation as a scorer function.

    Notes
    ------
        For a more complete understanding of how the IBM AIF360 library and bias mitigation works, check the following
    article:
    .. [1] Bellamy, R. K. E.; Dey, K.; Hind, M.; Hoffman, S. C.; Houde, S.; Kannan, K.;
           Lohia, P.; Martino, J.; Mehta, S.; Mojsilovic, A.; Nagar, S.; Ramamurthy,K. N.;
           Richards, J.; Saha, D.; Sattigeri, P.; Singh, M.; Varshney, K. R.; Zhang,Y,
           "AI Fairness 360: An Extensible Toolkit for Detecting, Understanding, and
           Mitigating Unwanted Algorithmic Bias", 2018
    """

    metrics = {
        'disparate_impact_ratio': make_scorer(disparate_impact_ratio_wrapper, biased_df=biased_df),
        'statistical_parity_difference': make_scorer(statistical_parity_difference_wrapper, greater_is_better=False,
                                                     biased_df=biased_df),
        'equal_opportunity_difference': make_scorer(equal_opportunity_difference_wrapper, greater_is_better=False,
                                                    biased_df=biased_df),
        'average_odds_difference': make_scorer(average_odds_difference_wrapper, greater_is_better=False,
                                               biased_df=biased_df),
        'average_odds_error': make_scorer(average_odds_error_wrapper, biased_df=biased_df),
    }
    refit_metric = 'disparate_impact_ratio'

    return metrics, refit_metric
