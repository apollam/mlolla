from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score, confusion_matrix, make_scorer, brier_score_loss


def _true_negatives_wp(y_true, y_pred):
    """Wrapper that will return the true negatives from confusion matrix."""

    return confusion_matrix(y_true, y_pred)[0, 0]


def _false_positives_wp(y_true, y_pred):
    """Wrapper that will return the false positives from confusion matrix."""

    return confusion_matrix(y_true, y_pred)[0, 1]


def _false_negatives_wp(y_true, y_pred):
    """Wrapper that will return the false negatives from confusion matrix."""

    return confusion_matrix(y_true, y_pred)[1, 0]


def _true_positives_wp(y_true, y_pred):
    """Wrapper that will return the true positives from confusion matrix."""

    return confusion_matrix(y_true, y_pred)[1, 1]


def get_binary_metrics(labels=None):
    """Return classification metrics to be use on classification evaluation."""

    metrics = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'false_negatives': make_scorer(_false_negatives_wp, greater_is_better=False),
        'false_positives': make_scorer(_false_positives_wp, greater_is_better=False),
        'true_negatives': make_scorer(_true_negatives_wp),
        'true_positives': make_scorer(_true_positives_wp),
        'brier_score_loss': make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
    }
    refit = 'f1'

    return metrics, refit
