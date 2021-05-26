import importlib
import pandas as pd


def get_class_from_string(module_name, class_name):
    """Receives a module name and the class name to be imported from this module
    as strings and returns them as an instantiable class.

    Parameters
    ----------
    module_name: str
        Name of the module to import a class from
    class_name: str
        Name of the class to be imported from the passed module

    Returns
    -------
        Instantiable class in the format `<class '<module-name>.<class-name>'>`
    """

    try:
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_

    except ImportError as e:
        print('Import Error with module {} and class or method {}'.format(module_name, class_name), e)


def transform(self, X, y=None):
    """Function that execute the predict method of a class. This function was
    created to be assign to a class without a transform method."""

    return self.predict(X).reshape(1, -1)


def predict(self, X, y=None):
    """Function that execute the transfrom method of a class. This function was
    created to be assign to a class without a predict method. """

    return self.transform(X)


def create_biased_df(train_X, train_y, sens_att_name):
    biased_df = pd.merge(train_y, train_X[sens_att_name], left_index=True, right_index=True)
    biased_df = biased_df.rename({sens_att_name: 'sens_attr'}, axis=1)
    train_X = train_X.drop(columns=sens_att_name)

    return train_X, biased_df


def get_random_state():
    return 420
