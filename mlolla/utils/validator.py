import logging
from sklearn.utils.multiclass import type_of_target


class Validator(object):
    """Validator object to be used to validate anything you want. The
    idea is to group all validations into one single file."""

    def __init__(self, config_json, X, y):
        self.config_json = config_json
        self.X = X
        self.y = y

    def validate_fairness(self):
        sens_attr = self.config_json["sens_attr_name"]
        if sens_attr:
            if sens_attr in self.X.columns:
                raise KeyError(f"The sensitive attribute '{sens_attr}' "
                               f"is not present in your DataFrame.")
        else:
            logging.info(f"No sensitive attribute was passed")

    def validate_input(self):
        learning_type = type_of_target(self.y)
        if learning_type not in ['binary', 'multiclass', 'continuous']:
            raise KeyError(f'Sorry, type of target {learning_type} '
                           f'is not supported yet.')

    def validate(self):
        self.validate_fairness()
        self.validate_input()
