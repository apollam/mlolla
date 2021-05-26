import numpy as np
import scipy.stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, confusion_matrix, make_scorer


def confusion_matrix_by_position(y_true, y_pred, line=0, column=0):
    """Returns the value of one position (line,Column) from confusion matrix"""
    confusion_matrix_values = confusion_matrix(y_true, y_pred)
    return confusion_matrix_values[line, column]


def get_values_function(y_true, y_pred, labels=None, function=None):
    """Returns a list with the results for metric function for each label of labels"""
    values = list()
    for label in labels:
        values.append(function(y_true, y_pred, labels=[label], average=None))
    return values


def mean_function(y_true, y_pred, labels=None, function=None):
    """Returns the mean of a metric function"""
    values = get_values_function(y_true, y_pred, labels=labels, function=function)
    return sum(values)/len(values)


def confidence_interval(y_true, y_pred, confidence=0.95, labels=None, function=None):
    """Return the confidence interval with t from metric function"""
    values = get_values_function(y_true, y_pred, labels=labels, function=function)
    a = 1.0 * np.array(values)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def get_multiclass_metrics(labels=None):
    """Default metrics to be use on multicass classification evaluation"""

    # General Metrics
    metrics={
        'accuracy': make_scorer(accuracy_score),
        'mean_precision': make_scorer(mean_function, function=precision_score, labels=labels),
        'mean_recall': make_scorer(mean_function, function=recall_score, labels=labels),
        'mean_f1': make_scorer(mean_function, function=f1_score, labels=labels),
        'mean_precision_confidence_interval': make_scorer(confidence_interval, function=precision_score, labels=labels),
        'mean_recall_confidence_interval': make_scorer(confidence_interval, function=recall_score, labels=labels),
        'mean_f1_confidence_interval': make_scorer(confidence_interval, function=f1_score, labels=labels),
    }
    # Metrics for each label
    for i, label in enumerate(labels):
        metrics.update(
            {
                'precision_class_{}'.format(label): make_scorer(precision_score, labels=[label], average=None),
                'recall_class_{}'.format(label): make_scorer(recall_score, labels=[label], average=None),
                'f1_class_{}'.format(label): make_scorer(f1_score, labels=[label], average=None)
        })
        for position in range(len(labels)):
            metrics.update(
            {
                'confusion_matrix_by_position_{}_{}'.format(labels[i], labels[position]):
                    make_scorer(confusion_matrix_by_position, line=i, column=position),
            })

    refit_metric = 'mean_f1'

    return metrics, refit_metric
