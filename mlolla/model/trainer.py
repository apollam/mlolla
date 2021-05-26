from .train_factory import TrainFactory
from mlolla.utils.config_json_treatment import get_pipeline_steps, get_training_params, get_x_y
from mlolla.utils.train_utils import create_biased_df, get_random_state, get_class_from_string
from mlolla.metrics.fairness_metrics import get_fairness_metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from mlolla.model_selection.feature_stratified_kfold import FeatureStratifiedKFold


class Trainer(object):
    """Trainer object used to configure your training based on config.json
    files parameters."""

    def __init__(self, config_json):
        self.config_json = config_json
        self.score_metrics = None
        self.refit_metric = None
        self.kfold = None
        self.training_params = None
        self.pipeline_steps = None
        self.random_state = None
        self.learning_type = None
        self._sens_attr = None

    def _set_config_variables(self, y):
        """Set variables related to training configuration"""

        self._sens_attr = self.config_json["sens_attr_name"] \
            if "sens_attr_name" in self.config_json.keys() else None
        self.training_params = get_training_params(self.config_json)
        self.pipeline_steps = get_pipeline_steps(self.config_json)
        self.random_state = get_random_state()
        self.learning_type = type_of_target(y)

    def _set_metrics(self, y):
        """Set metrics depending on type of label"""

        get_metrics = get_class_from_string(
            module_name=f'mlolla.metrics.{self.learning_type}_metrics',
            class_name=f'get_{self.learning_type}_metrics'
        )
        self.score_metrics, self.refit_metric = \
            get_metrics(labels=y.unique().tolist())

    def _set_kfold(self):
        """Set cross validation strategy depending on type of label"""

        if (self.learning_type == 'binary') or \
                (self.learning_type == 'multiclass'):
            self.kfold = StratifiedKFold(n_splits=5,
                                         shuffle=True,
                                         random_state=self.random_state)
        elif self.learning_type == 'continuous':
            self.kfold = KFold(n_splits=5,
                               shuffle=True,
                               random_state=self.random_state)
        else:
            raise KeyError(f'Sorry, type of target '
                           f'{self.learning_type} is not supported yet.')

    def _set_fairness(self, X, y):
        """Separates the sensitive attribute from X and set parameters
        related to fairness"""

        if self._sens_attr:
            if self._sens_attr in X.columns:
                self.kfold = FeatureStratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.random_state,
                    feature_arr=X[self._sens_attr]
                )
                X, biased_df = create_biased_df(X, y, self._sens_attr)
                fair_metrics, _ = get_fairness_metrics(biased_df)
                self.score_metrics.update(fair_metrics)
            else:
                raise KeyError(f"The sensitive attribute '{self._sens_attr}' "
                               f"is not present in your DataFrame.")

    def _set_params(self, X, y):
        """Set train parameters"""

        self._set_config_variables(y)
        self._set_metrics(y)
        self._set_kfold()
        self._set_fairness(self, X, y)

    def train(self):
        """Train a model using the parameters specified in config.json file."""

        X, y = get_x_y(self.config_json, type='train')

        self._set_params(X, y)

        train = TrainFactory(self.pipeline_steps, self.training_params)
        search = getattr(train, self.config_json['train_type'])
        fitted_obj = search(X, y, self.kfold, self.score_metrics, self.refit_metric)

        return fitted_obj
