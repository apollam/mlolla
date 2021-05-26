from .train_factory import TrainFactory
from mlolla.utils import get_pipeline_steps, get_training_params, get_x_y
from mlolla.utils import create_biased_df, get_random_state, get_class_from_string
from mlolla.metrics.fairness_metrics import get_fairness_metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from mlolla.model_selection.feature_stratified_kfold import FeatureStratifiedKFold


class Trainer(object):
    def __init__(self, config_json):
        self.config_json = config_json
        self.score_metrics = None
        self.refit_metric = None
        self.kfold = None
        self.training_params = None
        self.pipeline_steps = None

    def _set_params(self, X, y):
        """Set train parameters using functions from get_train_params file."""

        self.training_params = get_training_params(self.config_json)
        self.pipeline_steps = get_pipeline_steps(self.config_json)
        random_state = get_random_state()

        learning_type = type_of_target(y)
        get_metrics = get_class_from_string(module_name='mlolla.metrics.{}_metrics'.format(learning_type),
                                            class_name='get_{}_metrics'.format(learning_type))

        self.score_metrics, self.refit_metric = get_metrics(labels=y.unique().tolist())

        if (learning_type == 'binary') | (learning_type == 'multiclass'):
            self.kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        elif learning_type == 'continuous':
            self.kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        else:
            raise KeyError('Sorry, type of target {} is not supported yet.'.format(learning_type))

        # Separates the sensitive attribute from X
        if "sens_attr_name" in self.config_json.keys():
            if self.config_json["sens_attr_name"] in X.columns:
                self.kfold = FeatureStratifiedKFold(n_splits=5, shuffle=True, random_state=random_state,
                                                    feature_arr=X[self.config_json["sens_attr_name"]])
                X, biased_df = create_biased_df(X, y, self.config_json["sens_attr_name"])
                fair_metrics, _ = get_fairness_metrics(biased_df)
                self.score_metrics.update(fair_metrics)
            else:
                raise KeyError("The sensitive attribute '{}' is not present in your DataFrame."\
                               .format(self.config_json["sens_attr_name"]))

    def train(self):
        """Train a model using the parameters specified in config.json file."""

        X, y = get_x_y(self.config_json, type='train')

        self._set_params(X, y)

        train = TrainFactory(self.pipeline_steps, self.training_params)
        search = getattr(train, self.config_json['train_type'])
        fitted_obj = search(X, y, self.kfold, self.score_metrics, self.refit_metric)

        return fitted_obj
