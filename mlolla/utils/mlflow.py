from typing import AnyStr

import mlflow
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def init_mlflow(uri="http://localhost:5000"):
    """Initialize mlflow with tracked configuration."""
    mlflow.set_tracking_uri(uri)


def log_inmemory_artifact(name: str, content: AnyStr, is_binary: bool = False):
    """Log artifact with specific content (needed for when not running local)

    :param name: [str], Name of the artifact that should be stored.
    :param content: [str], Content that should be written to artifact.
    """

    fname = '/tmp/%s' % name

    if is_binary:
        f = open(fname, 'wb')
    else:
        f = open(fname, 'w')

    f.write(content)
    f.close()
    mlflow.log_artifact(fname)


def log_metrics(metrics, prefix):
    """Log metrics to MLFlow

    :param metrics: [dict], dict in the format {metric_name: metric_value}
        (usually outputted from `training_pipeline.evaluate()` function)
    :param prefix: [str], a prefix of the metric name (an advice is to use
        'train' or 'test')
    """

    for metric_name, metric_value in metrics.items():
        if metric_name == 'confusion_matrix':
            log_inmemory_artifact(f"{prefix}_{metric_name}.txt",
                                  str(metric_value).replace('\n', ' '))
        elif metric_name == 'classification_report':
            log_inmemory_artifact(f"{prefix}_{metric_name}.txt", str(metric_value))
        else:
            mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)


def log_hyperparams(hyperparams):
    """Log model's hyperparameters into MLFlow

    :param hyperparams: [dict], dict containing the model's hyperparameters in
        the format {hyperparam_name: hyperparam_value}
    """

    for param_name, param_value in hyperparams.items():
        mlflow.log_param(param_name, param_value)


def send_experiment_to_mlflow(experiment_name, train_metrics, test_metrics,
                              hyperparams, uri):
    """Function to be used on `app_train.py` responsible for logging an
    experiment's training information

    :param experiment_name: [str], name of your experiment (usually your
        model's name)
    :param train_metrics: [dict], dict containing the training metrics
        in the format {metric_name: metric_value} (usually outputted from
        `training_pipeline.evaluate()` function)
    :param test_metrics: [dict], dict containing the test metrics
        in the format {metric_name: metric_value} (usually outputted from
        `training_pipeline.evaluate()` function)
    :param hyperparams: [dict], dict containing the model's hyperparameters in
        the format {hyperparam_name: hyperparam_value}
    """

    logging.info(f"Storing experiment {experiment_name} to MLFlow")

    init_mlflow(uri)
    ml_exp = mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        for prefix, metrics in zip(['train', 'test'],
                                   [train_metrics, test_metrics]):
            log_metrics(metrics, prefix)
        log_hyperparams(hyperparams)
