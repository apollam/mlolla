from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from imblearn.pipeline import Pipeline
from .utils.train_utils import get_random_state


class TrainFactory(object):
    """Receives train parameters to create a fitted train object using specified type of model selection."""

    def __init__(self, pipeline_steps, training_params=[]):
        """Constructor.

        Parameters
        ----------
        pipeline_steps: list
            List of steps to be passed to a Pipeline
        training_params: list
            Parameters to be passed in grid search or randomized search
        """

        self.training_params = training_params
        self.pipeline_steps = pipeline_steps
        self.random_state = get_random_state()

    def get_pipeline(self):
        """Returns a Pipeline object containing the steps specified in the config.json file."""

        return Pipeline(self.pipeline_steps)

    def train_grid_search(self, X, y, kfold=None, score_metrics=None, refit_metric=True, estimator=None, **kwargs):
        """Fit and return a trained gid search with specified parameters

        Parameters
        ----------
        X : pd.DataFrame, array-like
            Training vector

        y : pd.DataFrame, array-like
            Target relative to X for classification or regression;
            None for unsupervised learning.

        kfold : cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            For classification use StratifiedKFold, for regression use KFold

        score_metrics : string, callable, list/tuple, dict or None, default: None
            A single string or a callable to evaluate the predictions on the test set.

            For evaluating multiple metrics, either give a list of (unique) strings
            or a dict with names as keys and callables as values.

        refit_metric : boolean, string, or callable, default=True
            Refit an estimator using the best found parameters on the whole dataset.

        estimator : obj or estimator
            Pipeline object or estimator

        **kwargs : optional
            Optional keyword arguments can be passed to GridSearch

        Return
        ------
            A fitted GridSearch object
        """

        pipeline = estimator if estimator is not None else self.get_pipeline()
        grid = GridSearchCV(pipeline,
                            param_grid=self.training_params,
                            verbose=True,
                            cv=kfold,
                            return_train_score=False,
                            scoring=score_metrics,
                            refit=refit_metric,
                            **kwargs)

        grid = grid.fit(X, y)

        return grid

    def train_randomized_search(self, X, y, kfold=None, score_metrics=None, refit_metric=True,
                                n_iter=10, estimator=None, **kwargs):
        """Fit and return a trained randomized search with specified parameters

        Parameters
        ----------
        X : pd.DataFrame, array-like
            Training vector.

        y : pd.DataFrame, array-like
            Target relative to X for classification or regression;
            None for unsupervised learning.

        kfold : cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            For classification use StratifiedKFold, for regression use KFold

        score_metrics : string, callable, list/tuple, dict or None, default: None
            A single string or a callable to evaluate the predictions on the test set.

            For evaluating multiple metrics, either give a list of (unique) strings
            or a dict with names as keys and callables as values.

        refit_metric : boolean, string, or callable, default=True
            Refit an estimator using the best found parameters on the whole dataset.

        n_iter : int, default=10
            Number of parameter settings that are sampled. n_iter trades
            off runtime vs quality of the solution.

        estimator : obj or estimator
            Pipeline object or estimator.

        **kwargs : optional
            Optional keyword arguments can be passed to RandomizedSearch.

        Return
        ------
            A fitted RandomizedSearch object
        """

        pipeline = estimator if estimator is not None else self.get_pipeline()
        random = RandomizedSearchCV(pipeline,
                                    param_distributions=self.training_params,
                                    verbose=1,
                                    n_iter=n_iter,
                                    cv=kfold,
                                    return_train_score=False,
                                    scoring=score_metrics,
                                    refit=refit_metric,
                                    random_state=self.random_state,
                                    error_score=0,
                                    **kwargs)
        random = random.fit(X, y)

        return random
