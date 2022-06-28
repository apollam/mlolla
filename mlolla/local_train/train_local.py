import pandas as pd

from mlolla.utils.decorators import timeit
from mlolla.utils import get_random_state

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, cross_validate

RANDOM_STATE = get_random_state()


@timeit
def train_model_cv(model, X, y, fit_params, score_metrics, return_estimator=False):
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = cross_validate(estimator=model,
                             X=X,
                             y=y,
                             cv=kfold,
                             n_jobs=-1,
                             verbose=-1,
                             fit_params=fit_params,
                             scoring=score_metrics,
                             return_estimator=return_estimator,
                             return_train_score=True,
                             error_score="raise")
    results = pd.DataFrame.from_dict(results)

    return results


@timeit
def train_model_grid_search(steps, training_params, X, y, score_metrics,
                            refit_metric, fit_params={}, desc=""):
    pipe = Pipeline(steps=steps)

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(pipe,
                        param_grid=training_params,
                        verbose=True,
                        cv=kfold,
                        return_train_score=True,
                        scoring=score_metrics,
                        n_jobs=-1,
                        refit=refit_metric,
                        iid=True)

    grid.fit(X, y, **fit_params)
    print("Best parameter (CV score=%0.3f):" % grid.best_score_)

    results = pd.DataFrame(grid.cv_results_, index=[desc])

    return results, grid


