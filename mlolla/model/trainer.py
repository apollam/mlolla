from .train_factory import TrainFactory
from mlolla.utils.config_json_treatment import get_pipeline_steps, get_training_params, get_x_y
from mlolla.utils.train_utils import create_biased_df, get_random_state, get_class_from_string
from mlolla.utils.validator import Validator
from mlolla.model.metrics import get_fairness_metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from mlolla.model.kfolds import FeatureStratifiedKFold


class Trainer(object):
    """Trainer object used to configure your training based on config.json
    files parameters."""

    def __init__(self, config_json):
        self.config_json = config_json
        self._score_metrics = None
        self._refit_metric = None
        self._kfold = None
        self._training_params = None
        self._pipeline_steps = None
        self._random_state = None
        self._learning_type = None
        self._sens_attr = None

    def _set_config_variables(self, y):
        """Set variables related to training configuration"""

        self._sens_attr = self.config_json["sens_attr_name"] \
            if "sens_attr_name" in self.config_json.keys() else None
        self._training_params = get_training_params(self.config_json)
        self._pipeline_steps = get_pipeline_steps(self.config_json)
        self._random_state = get_random_state()
        self._learning_type = type_of_target(y)

    def _set_metrics(self, y):
        """Set metrics depending on type of label"""

        get_metrics = get_class_from_string(
            module_name=f'mlolla.metrics.{self._learning_type}_metrics',
            class_name=f'get_{self._learning_type}_metrics'
        )
        self._score_metrics, self._refit_metric = \
            get_metrics(labels=y.unique().tolist())

    def _set_kfold(self):
        """Set cross validation strategy depending on type of label"""

        if (self._learning_type == 'binary') or \
                (self._learning_type == 'multiclass'):
            self._kfold = StratifiedKFold(n_splits=5,
                                          shuffle=True,
                                          random_state=self._random_state)
        elif self._learning_type == 'continuous':
            self._kfold = KFold(n_splits=5,
                                shuffle=True,
                                random_state=self._random_state)

    def _set_fairness_attributes(self, X, y):
        """Separates the sensitive attribute from X and set parameters
        related to fairness"""

        self._kfold = FeatureStratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self._random_state,
            feature_arr=X[self._sens_attr]
        )
        X, biased_df = create_biased_df(X, y, self._sens_attr)
        fair_metrics, _ = get_fairness_metrics(biased_df)
        self._score_metrics.update(fair_metrics)

    def _set_params(self, X, y):
        """Set train parameters"""

        self._set_config_variables(y)
        self._set_metrics(y)
        self._set_kfold()
        self._set_fairness_attributes(X, y)

    def train(self):
        """Train a model using the parameters specified in config.json file."""

        X, y = get_x_y(self.config_json, type='train')

        Validator(self.config_json, X, y).validate()

        self._set_params(X, y)

        train = TrainFactory(self._pipeline_steps, self._training_params)
        search = getattr(train, name=self.config_json['train_type'])
        fitted_obj = search(X, y, self._kfold, self._score_metrics, self._refit_metric)

        return fitted_obj
